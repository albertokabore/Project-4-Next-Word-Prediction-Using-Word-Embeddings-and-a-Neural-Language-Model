"""Step 7 - Generate training sequences (context words -> next word)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .vocab import Vocabulary


def split_sentences(sentences: list[list[str]], train_frac: float, val_frac: float):
    """Split in reading order (no shuffling) so validation/test text is truly unseen."""
    n = len(sentences)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return sentences[:n_train], sentences[n_train:n_train + n_val], sentences[n_train + n_val:]


def make_sequences(sentences: list[list[str]], vocab: Vocabulary, context_size: int):
    """Slide a window over the token stream.

    Each sample is ``context_size`` word ids (input) and the id of the word that
    follows (target). Unknown targets are skipped as a modeling choice; unknown
    words can still occur in the input. Evaluation therefore covers only known
    target words, and must report how many targets were excluded. Shorter
    contexts at the start of each split are left-padded with <pad>.
    """
    stream = [vocab.pad_id] * context_size
    for sentence in sentences:
        stream.extend(vocab.encode(sentence))

    contexts, targets = [], []
    for i in range(context_size, len(stream)):
        target = stream[i]
        if target == vocab.unk_id:
            continue
        contexts.append(stream[i - context_size:i])
        targets.append(target)

    if not contexts:
        return torch.empty((0, context_size), dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class NextWordDataset(Dataset):
    def __init__(self, contexts: torch.Tensor, targets: torch.Tensor):
        if len(contexts) != len(targets):
            raise ValueError("contexts and targets must have the same length")
        self.contexts = contexts
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.contexts[idx], self.targets[idx]
