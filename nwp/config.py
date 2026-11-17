"""Central configuration. Every tunable value of the pipeline lives here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    # --- Data -------------------------------------------------------------
    kaggle_dataset: str = "ronikdedhia/next-word-prediction"
    raw_filename: str = "1661-0.txt"  # The Adventures of Sherlock Holmes
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    # --- Splits (by sentence, in order, so test text is never seen) --------
    train_frac: float = 0.8
    val_frac: float = 0.1  # test gets the remainder

    # --- Vocabulary -------------------------------------------------------
    min_freq: int = 2  # words seen fewer times become <unk>

    # --- Sequences --------------------------------------------------------
    context_size: int = 5  # number of input words used to predict the next

    # --- Word2Vec ---------------------------------------------------------
    embedding_dim: int = 100
    w2v_window: int = 5
    w2v_sg: int = 1  # 1 = skip-gram, 0 = CBOW
    w2v_negative: int = 10
    w2v_epochs: int = 30
    freeze_embeddings_epochs: int = 2  # train head first, then fine-tune

    # --- Neural network ---------------------------------------------------
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3

    # --- Training ---------------------------------------------------------
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 3  # early stopping on validation loss
    seed: int = 42
    device: str = "auto"  # "auto", "cpu" or "cuda"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        if not 0 < self.train_frac < 1 or not 0 < self.val_frac < 1:
            raise ValueError("train_frac and val_frac must be in (0, 1)")
        if self.train_frac + self.val_frac >= 1:
            raise ValueError("train_frac + val_frac must leave room for a test split")
        if self.context_size < 1:
            raise ValueError("context_size must be >= 1")
        if self.min_freq < 1:
            raise ValueError("min_freq must be >= 1")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["artifacts_dir"] = str(self.artifacts_dir)
        return d
