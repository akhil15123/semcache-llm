"""Simulate a support-bot workload where users rephrase the same questions.

Run: python examples/benchmark.py
Prints hit rate, LLM calls avoided and a threshold sweep.
"""

import random

from semcache import SemanticCache

BASE = [
    "How do I reset my password?",
    "What are your opening hours?",
    "How can I cancel my subscription?",
    "Where is my order?",
    "Do you ship internationally?",
    "How do I change my email address?",
    "What payment methods do you accept?",
    "How do I get a refund?",
]


def perturb(q: str, rng: random.Random) -> str:
    ops = [
        lambda s: s.lower(),
        lambda s: s.rstrip("?") + "??",
        lambda s: "please " + s,
        lambda s: s.replace("How do I", "how can i"),
        lambda s: "hey, " + s.lower(),
        lambda s: s + " thanks",
    ]
    for op in rng.sample(ops, k=rng.randint(0, 3)):
        q = op(q)
    return q


def run(threshold: float, n: int = 400, seed: int = 7) -> dict:
    rng = random.Random(seed)
    cache = SemanticCache(threshold=threshold)
    wrong = 0
    for _ in range(n):
        intent = rng.randrange(len(BASE))
        q = perturb(BASE[intent], rng)
        hit = cache.lookup(q)
        if hit is None:
            cache.store(q, intent, tokens=120)
        elif hit.response != intent:
            wrong += 1
    s = cache.stats.as_dict()
    s["wrong_hits"] = wrong
    s["threshold"] = threshold
    return s


if __name__ == "__main__":
    print(f"{'thr':>5} {'hit_rate':>9} {'wrong':>6} {'tokens_saved':>13}")
    for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
        r = run(thr)
        print(f"{thr:>5} {r['hit_rate']:>9.3f} {r['wrong_hits']:>6} {r['tokens_saved']:>13}")
