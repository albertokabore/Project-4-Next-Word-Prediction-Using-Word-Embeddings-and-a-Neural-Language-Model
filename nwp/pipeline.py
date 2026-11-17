"""Run the whole project end to end: collect -> preprocess -> vocab -> sequences
-> Word2Vec -> train -> evaluate -> save.

    python -m nwp.pipeline            # full run
    python -m nwp.pipeline --epochs 3 # quicker run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import collect_dataset, read_text
from .embeddings import build_embedding_matrix, train_word2vec
from .model import NextWordLSTM
from .predict import Predictor
from .preprocess import preprocess_corpus
from .sequences import NextWordDataset, make_sequences, split_sentences
from .trainer import evaluate, train
from .vocab import Vocabulary

log = logging.getLogger("nwp")

DEMO_PROMPTS = [
    "my dear watson",
    "i have no doubt that",
    "sherlock holmes was",
    "it was a dark",
    "what do you",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def unigram_baseline(train_targets: torch.Tensor, test_targets: torch.Tensor) -> dict:
    """Accuracy of always guessing the most common words: the bar the model must beat."""
    top5 = [w for w, _ in Counter(train_targets.tolist()).most_common(5)]
    test = test_targets.tolist()
    return {
        "top1_acc": sum(t == top5[0] for t in test) / len(test),
        "top5_acc": sum(t in top5 for t in test) / len(test),
    }


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out = cfg.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. Collect
    raw_path = collect_dataset(cfg.kaggle_dataset, cfg.raw_filename, cfg.data_dir)
    raw_text = read_text(raw_path)

    # 2-5. Clean, lowercase, remove punctuation, tokenize
    sentences = preprocess_corpus(raw_text)
    n_tokens = sum(map(len, sentences))
    log.info("Preprocessed %d sentences, %d tokens", len(sentences), n_tokens)
    log.info("Example sentence: %s", " ".join(sentences[min(10, len(sentences) - 1)]))

    train_s, val_s, test_s = split_sentences(sentences, cfg.train_frac, cfg.val_frac)

    # 6. Vocabulary (from training text only)
    vocab = Vocabulary.build(train_s, cfg.min_freq)
    vocab.save(out / "vocab.json")
    log.info("Vocabulary: %d words (min_freq=%d)", len(vocab), cfg.min_freq)

    # 7. Training sequences
    splits = {name: make_sequences(s, vocab, cfg.context_size)
              for name, s in (("train", train_s), ("val", val_s), ("test", test_s))}
    for name, (x, _) in splits.items():
        log.info("%-5s sequences: %d", name, len(x))
    x0, y0 = splits["train"][0][100], splits["train"][1][100]
    log.info("Example sequence: %s -> %s", vocab.decode(x0.tolist()), vocab.idx2word[int(y0)])

    # 8. Word2Vec embeddings
    w2v = train_word2vec(train_s, cfg.embedding_dim, cfg.w2v_window, cfg.min_freq,
                         cfg.w2v_sg, cfg.w2v_negative, cfg.w2v_epochs, cfg.seed)
    w2v.save(str(out / "word2vec.model"))
    matrix, covered = build_embedding_matrix(w2v, vocab, cfg.seed)
    log.info("Word2Vec covers %d / %d vocabulary words", covered, len(vocab) - 2)
    neighbours = {}
    for word in ("holmes", "watson", "door", "said", "money"):
        if word in w2v.wv:
            neighbours[word] = [w for w, _ in w2v.wv.most_similar(word, topn=5)]
            log.info("Nearest to %-7s %s", word + ":", neighbours[word])

    # 9. Train the neural network
    loaders = {
        name: DataLoader(NextWordDataset(x, y), batch_size=cfg.batch_size, shuffle=(name == "train"),
                         generator=torch.Generator().manual_seed(cfg.seed))
        for name, (x, y) in splits.items()
    }
    model = NextWordLSTM(matrix, cfg.hidden_size, cfg.num_layers, cfg.dropout, vocab.pad_id)
    log.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    history = train(model, loaders["train"], loaders["val"], device, cfg.epochs, cfg.learning_rate,
                    cfg.weight_decay, cfg.grad_clip, cfg.patience, cfg.freeze_embeddings_epochs)

    # Evaluate on text the model has never seen
    test = evaluate(model, loaders["test"], device)
    baseline = unigram_baseline(splits["train"][1], splits["test"][1])
    log.info("TEST  loss %.3f | perplexity %.1f | top1 %.3f | top5 %.3f",
             test.loss, test.perplexity, test.top1, test.top5)
    log.info("BASELINE (most frequent word) top1 %.3f | top5 %.3f", baseline["top1_acc"], baseline["top5_acc"])

    model.cpu()
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    predictor = Predictor(model, vocab, cfg.context_size)
    demos = {p: predictor.predict_next(p, 5) for p in DEMO_PROMPTS}
    for prompt, preds in demos.items():
        log.info("%-22r -> %s", prompt, ", ".join(f"{w} ({p:.0%})" for w, p in preds))

    report = {
        "dataset": {"kaggle": cfg.kaggle_dataset, "file": cfg.raw_filename,
                    "sentences": len(sentences), "tokens": n_tokens,
                    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
        "vocab_size": len(vocab),
        "sequences": {name: len(x) for name, (x, _) in splits.items()},
        "target_coverage": {
            name: {"tokens": sum(map(len, part)), "scored_targets": len(splits[name][1]),
                   "excluded_unknown_targets": sum(map(len, part)) - len(splits[name][1])}
            for name, part in (("train", train_s), ("val", val_s), ("test", test_s))
        },
        "word2vec_coverage": covered,
        "word2vec_neighbours": neighbours,
        "history": history.__dict__,
        "test": test.as_dict(),
        "baseline": baseline,
        "demo_predictions": demos,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_history(history, out / "training_curves.png")
    return report


def _plot_history(history, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping training plot")
        return
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history.train_loss, label="train")
    ax1.plot(epochs, history.val_loss, label="validation")
    ax1.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    ax1.legend()
    ax2.plot(epochs, history.val_top1, label="top-1")
    ax2.plot(epochs, history.val_top5, label="top-5")
    ax2.set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the next-word prediction model.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--min-freq", type=int)
    parser.add_argument("--device")
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("gensim").setLevel(logging.WARNING)
    run(Config(**overrides))


if __name__ == "__main__":
    main()
