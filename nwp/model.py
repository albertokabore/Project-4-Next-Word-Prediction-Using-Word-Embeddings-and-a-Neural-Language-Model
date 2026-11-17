"""Step 9 (part 1) - The neural language model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class NextWordLSTM(nn.Module):
    """word ids -> Word2Vec embeddings -> 2-layer LSTM -> scores for every vocabulary word."""

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_id: int = 0,
    ):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def set_embeddings_trainable(self, trainable: bool) -> None:
        self.embedding.weight.requires_grad_(trainable)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (batch, context_size) ids -> logits: (batch, vocab_size)."""
        vectors = self.embedding(context)
        outputs, _ = self.lstm(vectors)
        last = outputs[:, -1, :]  # the LSTM's summary after reading the whole context
        return self.output(self.dropout(last))
