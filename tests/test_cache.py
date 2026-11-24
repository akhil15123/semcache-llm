import pytest

from semcache import HashingEmbedder, SemanticCache


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_exact_and_near_duplicate_hit():
    cache = SemanticCache(threshold=0.8)
    cache.store("What is the capital of France?", "Paris")
    assert cache.lookup("What is the capital of France?").response == "Paris"
    hit = cache.lookup("capital of france??")
    assert hit is not None and hit.response == "Paris"
    assert 0.8 <= hit.similarity <= 1.0001


def test_unrelated_prompt_misses():
    cache = SemanticCache(threshold=0.8)
    cache.store("What is the capital of France?", "Paris")
    assert cache.lookup("Write a haiku about kubernetes") is None
    assert cache.stats.misses == 1


def test_namespaces_are_isolated():
    cache = SemanticCache()
    cache.store("summarise this", "gpt answer", namespace="model-a")
    assert cache.lookup("summarise this", namespace="model-b") is None
    assert cache.lookup("summarise this", namespace="model-a").response == "gpt answer"


def test_lru_eviction():
    cache = SemanticCache(max_entries=2, threshold=0.95)
    cache.store("alpha question one", 1)
    cache.store("beta question two", 2)
    cache.lookup("alpha question one")  # touch alpha so beta is LRU
    cache.store("gamma question three", 3)
    assert len(cache) == 2
    assert cache.lookup("beta question two") is None
    assert cache.lookup("alpha question one").response == 1
    assert cache.stats.evictions == 1


def test_ttl_expiry():
    clock = FakeClock()
    cache = SemanticCache(ttl_seconds=10, clock=clock)
    cache.store("hello world", "hi")
    clock.t = 5
    assert cache.lookup("hello world") is not None
    clock.t = 11
    assert cache.lookup("hello world") is None
    assert cache.stats.expirations == 1


def test_get_or_call_only_calls_once_and_counts_tokens():
    cache = SemanticCache()
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "answer text with five tokens"

    counter = lambda r: len(r.split())
    assert cache.get_or_call("explain RAG", llm, token_counter=counter) == "answer text with five tokens"
    assert cache.get_or_call("Explain RAG.", llm, token_counter=counter) == "answer text with five tokens"
    assert len(calls) == 1
    assert cache.stats.tokens_saved == 5
    assert cache.stats.hit_rate == 0.5


def test_decorator_partitions_by_kwargs():
    cache = SemanticCache()
    calls = []

    @cache.cached()
    def complete(prompt, model="small", temperature=0.0):
        calls.append((prompt, model))
        return f"{model}:{prompt}"

    complete("define entropy", model="small")
    complete("define entropy", model="small")
    complete("define entropy", model="large")
    assert calls == [("define entropy", "small"), ("define entropy", "large")]


@pytest.mark.parametrize("bad", [0.0, -1, 1.5])
def test_invalid_threshold(bad):
    with pytest.raises(ValueError):
        SemanticCache(threshold=bad)


def test_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed(["same text", "same text"])
    assert abs(float(a @ a) - 1.0) < 1e-5
    assert (a == b).all()
    zero = emb.embed([""])[0]
    assert float(abs(zero).sum()) == 0.0
