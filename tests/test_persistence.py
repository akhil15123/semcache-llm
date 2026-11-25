import numpy as np
import pytest

from semcache import HashingEmbedder, SemanticCache


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_round_trip_preserves_hits_and_namespaces(tmp_path):
    cache = SemanticCache()
    cache.store("What is the capital of France?", "Paris", namespace="m1", tokens=3)
    cache.store("Define overfitting", {"text": "memorising noise", "score": 0.9}, namespace=("model", "x"))
    path = cache.save(tmp_path / "cache")
    assert path.suffix == ".npz"

    loaded = SemanticCache.load(path)
    assert len(loaded) == 2
    assert loaded.lookup("capital of france?", namespace="m1").response == "Paris"
    hit = loaded.lookup("define overfitting", namespace=("model", "x"))
    assert hit.response == {"text": "memorising noise", "score": 0.9}
    assert loaded.lookup("capital of france?", namespace="m2") is None


def test_decorator_namespaces_survive_reload(tmp_path):
    cache = SemanticCache()

    @cache.cached()
    def complete(prompt, model="small"):
        return f"{model}:{prompt}"

    complete("explain attention", model="small")
    path = cache.save(tmp_path / "c.npz")

    reloaded = SemanticCache.load(path)
    calls = []

    @reloaded.cached()
    def complete2(prompt, model="small"):
        calls.append(prompt)
        return "fresh"

    assert complete2("explain attention", model="small") == "small:explain attention"
    assert calls == []


def test_ttl_age_carries_over(tmp_path):
    c1 = SemanticCache(ttl_seconds=10, clock=FakeClock(100.0))
    c1.store("hello there", "hi")
    c1._clock.t = 106.0  # entry is 6s old when saved
    path = c1.save(tmp_path / "ttl.npz")

    clock = FakeClock(5000.0)
    c2 = SemanticCache.load(path, ttl_seconds=10, clock=clock)
    clock.t = 5003.0  # 9s old
    assert c2.lookup("hello there") is not None
    clock.t = 5005.0  # 11s old
    assert c2.lookup("hello there") is None


def test_load_respects_max_entries(tmp_path):
    cache = SemanticCache()
    for i in range(5):
        cache.store(f"question number {i} about topic {i}", i)
    path = cache.save(tmp_path / "many.npz")
    small = SemanticCache.load(path, max_entries=2)
    assert len(small) == 2
    assert small.stats.evictions == 3


def test_embedder_mismatch_is_rejected(tmp_path):
    cache = SemanticCache(embedder=HashingEmbedder(dim=64))
    cache.store("a prompt", "a response")
    path = cache.save(tmp_path / "dim.npz")
    with pytest.raises(ValueError, match="dim=64"):
        SemanticCache.load(path, embedder=HashingEmbedder(dim=128))


def test_empty_cache_round_trip(tmp_path):
    path = SemanticCache().save(tmp_path / "empty.npz")
    assert len(SemanticCache.load(path)) == 0
    with np.load(path) as data:
        assert data["vectors"].shape == (0, 512)
