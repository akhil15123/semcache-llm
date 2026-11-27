"""Precision/recall of cache hits per threshold on a labelled paraphrase set.

Run: python examples/eval_thresholds.py [--st]
  --st  use a sentence-transformers model instead of the offline hashing embedder
"""

import sys
from pathlib import Path

from semcache import HashingEmbedder
from semcache.evaluation import best_threshold, load_pairs, sweep

DATA = Path(__file__).resolve().parent.parent / "data" / "paraphrase_eval.jsonl"

if __name__ == "__main__":
    if "--st" in sys.argv:
        from semcache.embedders import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    else:
        embedder = HashingEmbedder()
    pairs = load_pairs(DATA)
    results = sweep(embedder, pairs)
    print(f"{type(embedder).__name__} on {len(pairs)} labelled pairs\n")
    print(f"{'thr':>6} {'precision':>9} {'recall':>7} {'f1':>6} {'false_hits':>10}")
    for r in results:
        print(r.as_row())
    best = best_threshold(results)
    print("\nrecommended threshold (precision >= 0.95):", best.threshold if best else "none")
