"""Check saved results against the local corpus and checkpoint, without training.

    python -m nwp.verify --output artifacts/verification.json

This checks consistency of the supplied files. It does not authenticate their
download history or independently reproduce the original training run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter
from pathlib import Path

import torch
from gensim.models import Word2Vec
from torch.utils.data import DataLoader

from .config import Config
from .data import read_text
from .pipeline import unigram_baseline
from .predict import Predictor
from .preprocess import preprocess_corpus
from .sequences import NextWordDataset, make_sequences, split_sentences
from .trainer import evaluate
from .vocab import Vocabulary


def verify(artifacts_dir: Path, data_dir: Path | None = None) -> dict:
    artifacts_dir = Path(artifacts_dir)
    cfg = Config(**json.loads((artifacts_dir / "config.json").read_text(encoding="utf-8")))
    saved = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    raw_path = (Path(data_dir) if data_dir is not None else cfg.data_dir) / "raw" / cfg.raw_filename
    raw = read_text(raw_path)
    sentences = preprocess_corpus(raw)
    parts = dict(zip(("train", "val", "test"), split_sentences(sentences, cfg.train_frac, cfg.val_frac)))
    vocab = Vocabulary.load(artifacts_dir / "vocab.json")
    rebuilt = Vocabulary.build(parts["train"], cfg.min_freq)
    splits = {name: make_sequences(part, vocab, cfg.context_size) for name, part in parts.items()}
    coverage = {}
    for name, part in parts.items():
        tokens = sum(map(len, part))
        scored = len(splits[name][1])
        coverage[name] = {"sentences": len(part), "tokens": tokens, "scored_targets": scored,
                          "excluded_unknown_targets": tokens - scored, "target_coverage": scored / tokens}

    predictor = Predictor.load(artifacts_dir)
    evaluated = {}
    for name in ("val", "test"):
        x, y = splits[name]
        loader = DataLoader(NextWordDataset(x, y), batch_size=cfg.batch_size)
        evaluated[name] = evaluate(predictor.model, loader, torch.device("cpu")).as_dict()
    baseline = unigram_baseline(splits["train"][1], splits["test"][1])
    close = lambda a, b: math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
    history = saved["history"]
    best = history["best_epoch"]
    w2v = Word2Vec.load(str(artifacts_dir / "word2vec.model"))
    train_counts = Counter(w for sentence in parts["train"] for w in sentence)
    neighbours = {w: [n for n, _ in w2v.wv.most_similar(w, topn=len(expected))]
                  for w, expected in saved["word2vec_neighbours"].items()}
    demos = {p: [list(item) for item in predictor.predict_next(p, len(expected))]
             for p, expected in saved["demo_predictions"].items()}
    checks = {
        "dataset_configuration": saved["dataset"]["kaggle"] == cfg.kaggle_dataset
            and saved["dataset"]["file"] == cfg.raw_filename,
        "dataset_counts": saved["dataset"]["sentences"] == len(sentences)
            and saved["dataset"]["tokens"] == sum(map(len, sentences)),
        "vocabulary": rebuilt.idx2word == vocab.idx2word and rebuilt.counts == vocab.counts
            and saved["vocab_size"] == len(vocab),
        "sequence_counts": saved["sequences"] == {name: len(y) for name, (_, y) in splits.items()},
        "test_metrics": all(close(value, saved["test"][key]) for key, value in evaluated["test"].items()),
        "baseline": all(close(value, saved["baseline"][key]) for key, value in baseline.items()),
        "history_lengths": len({len(history[key]) for key in ("train_loss", "val_loss", "val_top1", "val_top5")}) == 1,
        "best_epoch": best == min(range(1, len(history["val_loss"]) + 1), key=lambda e: history["val_loss"][e - 1]),
        "best_checkpoint_validation": close(evaluated["val"]["loss"], history["val_loss"][best - 1])
            and close(evaluated["val"]["top1_acc"], history["val_top1"][best - 1])
            and close(evaluated["val"]["top5_acc"], history["val_top5"][best - 1]),
        "word2vec_training_counts": set(w2v.wv.key_to_index) == set(vocab.idx2word[2:])
            and all(w2v.wv.get_vecattr(w, "count") == train_counts[w] for w in w2v.wv.key_to_index),
        "word2vec_coverage": saved["word2vec_coverage"] == len(w2v.wv) == len(vocab) - 2,
        "word2vec_neighbours": neighbours == saved["word2vec_neighbours"],
        "demo_predictions": demos == saved["demo_predictions"],
    }
    if "raw_sha256" in saved["dataset"]:
        checks["recorded_raw_sha256"] = saved["dataset"]["raw_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if "target_coverage" in saved:
        expected_coverage = {
            name: {key: coverage[name][key] for key in ("tokens", "scored_targets", "excluded_unknown_targets")}
            for name in parts
        }
        checks["recorded_target_coverage"] = saved["target_coverage"] == expected_coverage
    test_coverage = coverage["test"]["target_coverage"]
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    [raw_path, *(artifacts_dir / name for name in
                                  ("config.json", "vocab.json", "metrics.json", "model.pt", "word2vec.model"))]}
    return {
        "all_checks_passed": all(checks.values()), "checks": checks,
        "scope": "Re-evaluates local artifacts; no download authentication or full training rerun.",
        "sha256": fingerprints,
        "verification_environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("torch", "gensim", "numpy", "kagglehub", "pytest")}},
        "raw_file": {"bytes": raw_path.stat().st_size, "characters": len(raw),
                     "whitespace_words": len(raw.split()), "lines": len(raw.splitlines())},
        "coverage": coverage, "vocab_size": len(vocab),
        "model_parameters": sum(p.numel() for p in predictor.model.parameters()),
        "recomputed_validation": evaluated["val"], "recomputed_test": evaluated["test"],
        "recomputed_baseline": baseline,
        "all_test_word_accuracy_counting_unknowns_as_incorrect": {
            key: evaluated["test"][key] * test_coverage for key in ("top1_acc", "top5_acc")},
        "recomputed_demo_predictions": demos, "recomputed_word2vec_neighbours": neighbours,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = verify(args.artifacts, args.data_dir)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
