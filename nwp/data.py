"""Step 1 - Collect the dataset from Kaggle."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def collect_dataset(dataset: str, filename: str, data_dir: Path) -> Path:
    """Return the path to the raw text file, downloading it from Kaggle if needed.

    The file is copied into ``data_dir/raw`` so later runs work offline.
    """
    target = Path(data_dir) / "raw" / filename
    if target.exists():
        log.info("Using cached dataset at %s", target)
        return target

    import kagglehub  # imported lazily so the rest of the package works without it

    log.info("Downloading Kaggle dataset %s", dataset)
    download_dir = Path(kagglehub.dataset_download(dataset))
    source = download_dir / filename
    if not source.exists():
        found = sorted(p.name for p in download_dir.iterdir())
        raise FileNotFoundError(f"{filename!r} not in Kaggle download; found {found}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    log.info("Saved dataset to %s", target)
    return target


def read_text(path: Path) -> str:
    # utf-8-sig drops the byte-order mark at the start of Gutenberg files.
    return Path(path).read_text(encoding="utf-8-sig")
