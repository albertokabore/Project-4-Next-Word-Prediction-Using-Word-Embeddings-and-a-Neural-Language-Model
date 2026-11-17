"""Step 6 - Build a vocabulary (word <-> integer id)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"


class Vocabulary:
    """Maps words to ids. Id 0 is padding, id 1 is the unknown-word token."""

    def __init__(self, idx2word: list[str], counts: dict[str, int] | None = None):
        if idx2word[:2] != [PAD, UNK]:
            raise ValueError("vocabulary must start with <pad>, <unk>")
        self.idx2word = list(idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.counts = dict(counts or {})

    @classmethod
    def build(cls, sentences: Iterable[list[str]], min_freq: int = 1) -> "Vocabulary":
        counts = Counter(token for sentence in sentences for token in sentence)
        # Sort by frequency (desc) then alphabetically, so the result is deterministic.
        kept = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls([PAD, UNK, *kept], {w: counts[w] for w in kept})

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.idx2word)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.word2idx.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.idx2word[i] for i in ids]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"idx2word": self.idx2word, "counts": self.counts}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["idx2word"], data.get("counts"))
