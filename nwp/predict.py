"""Use a trained model: predict the next word, or generate a continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import NextWordLSTM
from .preprocess import normalize_sentence
from .vocab import Vocabulary


class Predictor:
    def __init__(self, model: NextWordLSTM, vocab: Vocabulary, context_size: int, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.context_size = context_size
        self.device = device
        # Never suggest the special tokens.
        self._banned = torch.tensor([vocab.pad_id, vocab.unk_id], device=device)

    @classmethod
    def load(cls, artifacts_dir: Path, device: str = "cpu") -> "Predictor":
        artifacts_dir = Path(artifacts_dir)
        cfg = json.loads((artifacts_dir / "config.json").read_text(encoding="utf-8"))
        vocab = Vocabulary.load(artifacts_dir / "vocab.json")
        state = torch.load(artifacts_dir / "model.pt", map_location=device, weights_only=True)
        dummy = np.zeros((len(vocab), cfg["embedding_dim"]), dtype=np.float32)
        model = NextWordLSTM(dummy, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"], vocab.pad_id)
        model.load_state_dict(state)
        return cls(model, vocab, cfg["context_size"], device)

    def _encode_context(self, tokens: list[str]) -> torch.Tensor:
        ids = self.vocab.encode(tokens)[-self.context_size:]
        ids = [self.vocab.pad_id] * (self.context_size - len(ids)) + ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _probabilities(self, tokens: list[str], temperature: float = 1.0) -> torch.Tensor:
        logits = self.model(self._encode_context(tokens))[0]
        logits[self._banned] = float("-inf")
        return torch.softmax(logits / temperature, dim=-1)

    def predict_next(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k most likely next words with their probabilities."""
        if not 1 <= k <= len(self.vocab) - 2:
            raise ValueError("k must be between 1 and the number of ordinary vocabulary words")
        probs = self._probabilities(normalize_sentence(text))
        top = probs.topk(k)
        return [(self.vocab.idx2word[i], round(p, 4)) for p, i in zip(top.values.tolist(), top.indices.tolist())]

    def generate(self, text: str, n_words: int = 10, temperature: float = 0.0, seed: int | None = None) -> str:
        """Append ``n_words`` words. temperature=0 is greedy; higher is more varied."""
        tokens = normalize_sentence(text)
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        for _ in range(n_words):
            probs = self._probabilities(tokens, temperature if temperature > 0 else 1.0)
            if temperature <= 0:
                idx = int(probs.argmax())
            else:
                idx = int(torch.multinomial(probs, 1, generator=gen))
            tokens.append(self.vocab.idx2word[idx])
        return " ".join(tokens)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict the next word with a trained model.")
    parser.add_argument("text", help="the beginning of a sentence")
    parser.add_argument("-k", type=int, default=5, help="how many suggestions to show")
    parser.add_argument("--generate", type=int, default=0, metavar="N", help="also generate N words")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    predictor = Predictor.load(args.artifacts)
    print(f"Input: {args.text!r}")
    for word, p in predictor.predict_next(args.text, args.k):
        print(f"  {word:<15} {p:.2%}")
    if args.generate:
        print("Generated:", predictor.generate(args.text, args.generate, args.temperature, seed=0))


if __name__ == "__main__":
    main()
