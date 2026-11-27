"""Threshold evaluation on labelled prompt pairs.

A cache hit is a prediction that two prompts can share an answer. Given pairs
labelled ``same`` (true paraphrase) or not, this computes precision (how often
a hit is safe to serve) and recall (how much reuse you get) per threshold.
For a cache, precision matters more: a wrong hit serves a wrong answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .embedders import Embedder


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_hits: int

    def as_row(self) -> str:
        return (
            f"{self.threshold:>6.2f} {self.precision:>9.3f} {self.recall:>7.3f} "
            f"{self.f1:>6.3f} {self.false_hits:>10d}"
        )


def load_pairs(path: str | Path) -> list[tuple[str, str, bool]]:
    pairs = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append((row["a"], row["b"], bool(row["same"])))
    return pairs


def pair_similarities(embedder: Embedder, pairs: Sequence[tuple[str, str, bool]]) -> np.ndarray:
    a = embedder.embed([p[0] for p in pairs])
    b = embedder.embed([p[1] for p in pairs])
    return np.sum(a * b, axis=1)


def sweep(
    embedder: Embedder,
    pairs: Sequence[tuple[str, str, bool]],
    thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95),
) -> list[ThresholdResult]:
    sims = pair_similarities(embedder, pairs)
    labels = np.array([p[2] for p in pairs])
    out = []
    for t in thresholds:
        pred = sims >= t
        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(ThresholdResult(t, precision, recall, f1, fp))
    return out


def best_threshold(results: Sequence[ThresholdResult], min_precision: float = 0.95) -> ThresholdResult | None:
    """Highest-recall threshold whose precision meets ``min_precision``."""
    ok = [r for r in results if r.precision >= min_precision and r.recall > 0]
    return max(ok, key=lambda r: (r.recall, -r.threshold)) if ok else None
