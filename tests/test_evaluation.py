from pathlib import Path

import numpy as np

from semcache import HashingEmbedder
from semcache.evaluation import ThresholdResult, best_threshold, load_pairs, sweep

DATA = Path(__file__).resolve().parent.parent / "data" / "paraphrase_eval.jsonl"


class StubEmbedder:
    """Maps known strings to fixed unit vectors so similarities are exact."""

    dim = 2

    def __init__(self, table):
        self.table = table

    def embed(self, texts):
        return np.array([self.table[t] for t in texts], dtype=np.float32)


def test_dataset_is_balanced_and_well_formed():
    pairs = load_pairs(DATA)
    assert len(pairs) >= 40
    assert sum(p[2] for p in pairs) == sum(not p[2] for p in pairs)


def test_sweep_counts_are_exact():
    v = lambda deg: [np.cos(np.radians(deg)), np.sin(np.radians(deg))]
    emb = StubEmbedder({"a": v(0), "b": v(10), "c": v(0), "d": v(60)})
    pairs = [("a", "b", True), ("c", "d", False), ("a", "d", True)]
    # cos(10deg)=0.985, cos(60deg)=0.5
    r = {x.threshold: x for x in sweep(emb, pairs, thresholds=[0.4, 0.9])}
    assert r[0.4].precision == 2 / 3 and r[0.4].recall == 1.0 and r[0.4].false_hits == 1
    assert r[0.9].precision == 1.0 and r[0.9].recall == 0.5 and r[0.9].false_hits == 0


def test_precision_rises_with_threshold_on_real_data():
    results = sweep(HashingEmbedder(), load_pairs(DATA), thresholds=[0.3, 0.95])
    assert results[1].precision >= results[0].precision
    assert results[1].recall <= results[0].recall


def test_best_threshold_prefers_recall_under_precision_floor():
    rs = [
        ThresholdResult(0.6, 0.80, 0.9, 0.85, 4),
        ThresholdResult(0.8, 0.96, 0.6, 0.74, 1),
        ThresholdResult(0.9, 1.00, 0.4, 0.57, 0),
    ]
    assert best_threshold(rs).threshold == 0.8
    assert best_threshold(rs, min_precision=0.99).threshold == 0.9
    assert best_threshold(rs[:1]) is None
