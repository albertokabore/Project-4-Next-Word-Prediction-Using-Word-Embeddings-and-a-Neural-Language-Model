"""Step 9 (part 2) - Train and evaluate the neural network."""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import NextWordLSTM

log = logging.getLogger(__name__)


@dataclass
class Metrics:
    loss: float
    perplexity: float
    top1: float
    top5: float

    def as_dict(self) -> dict:
        return {"loss": self.loss, "perplexity": self.perplexity, "top1_acc": self.top1, "top5_acc": self.top5}


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_top1: list[float] = field(default_factory=list)
    val_top5: list[float] = field(default_factory=list)
    best_epoch: int = 0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    """Average cross-entropy, perplexity (= e^loss) and top-1/top-5 accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, top1, top5, n = 0.0, 0, 0, 0
    for context, target in loader:
        context, target = context.to(device), target.to(device)
        logits = model(context)
        total_loss += criterion(logits, target).item()
        best5 = logits.topk(min(5, logits.size(1)), dim=1).indices
        top1 += (best5[:, 0] == target).sum().item()
        top5 += (best5 == target.unsqueeze(1)).any(dim=1).sum().item()
        n += target.size(0)
    if n == 0:
        raise ValueError("cannot evaluate on an empty dataset")
    loss = total_loss / n
    return Metrics(loss, math.exp(min(loss, 50)), top1 / n, top5 / n)


def train(
    model: NextWordLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    patience: int,
    freeze_embeddings_epochs: int = 0,
) -> History:
    """Mini-batch training with AdamW, gradient clipping and early stopping.

    For the first ``freeze_embeddings_epochs`` epochs the Word2Vec vectors are
    frozen to preserve their initial values; afterwards they are fine-tuned
    together with the rest of the network. The weights with the
    lowest validation loss are restored at the end.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = History()
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.set_embeddings_trainable(epoch > freeze_embeddings_epochs)
        model.train()
        start, running, seen = time.perf_counter(), 0.0, 0
        for context, target in train_loader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(context), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * target.size(0)
            seen += target.size(0)

        train_loss = running / max(seen, 1)
        val = evaluate(model, val_loader, device)
        scheduler.step(val.loss)
        history.train_loss.append(train_loss)
        history.val_loss.append(val.loss)
        history.val_top1.append(val.top1)
        history.val_top5.append(val.top5)
        log.info(
            "epoch %2d | train loss %.3f | val loss %.3f | val ppl %7.1f | top1 %.3f | top5 %.3f | %.0fs",
            epoch, train_loss, val.loss, val.perplexity, val.top1, val.top5, time.perf_counter() - start,
        )

        if val.loss < best_loss:
            best_loss, best_state, bad_epochs = val.loss, copy.deepcopy(model.state_dict()), 0
            history.best_epoch = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping: no validation improvement for %d epochs", patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.set_embeddings_trainable(True)
    return history
