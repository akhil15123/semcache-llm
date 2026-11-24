# semcache-llm

A **semantic cache for LLM calls**. Instead of keying cached responses on the
exact prompt string, `semcache` keys them on the *meaning* of the prompt:
"How do I reset my password?" and "hey, how can i reset my password??" hit the
same entry, saving a round-trip to the model and the tokens that go with it.

Week 1 of a weekly series of small, focused AI engineering builds.

## Why

In production chatbots and copilots a large share of traffic is rephrased
repeats. An exact-match cache misses almost all of it. A semantic cache:

- cuts latency from seconds to microseconds on repeats,
- reduces token spend,
- gives you a knob (`threshold`) to trade hit rate against the risk of serving
  a wrong answer.

## Features

- Cosine-similarity lookup over normalised embeddings (numpy, no vector DB needed)
- **Namespaces** so answers never leak across models / temperatures / tenants
- **LRU eviction** with a hard `max_entries` cap
- **TTL** expiry for answers that go stale
- **Stats**: hit rate, tokens saved, evictions, mean similarity on hit
- Pluggable embedders: an offline `HashingEmbedder` (default, zero downloads)
  or `SentenceTransformerEmbedder` for true paraphrase matching
- Thread-safe; `get_or_call` helper and a `@cache.cached()` decorator

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
python examples/benchmark.py
```

```python
from semcache import SemanticCache

cache = SemanticCache(threshold=0.85, max_entries=5_000, ttl_seconds=3600)

@cache.cached()                      # kwargs become the namespace
def complete(prompt, model="gpt-small", temperature=0.0):
    return call_your_llm(prompt, model=model, temperature=temperature)

complete("Explain retrieval-augmented generation", model="gpt-small")
complete("explain retrieval augmented generation?", model="gpt-small")  # cache hit
print(cache.stats.as_dict())
```

Swap in a neural embedder for real paraphrases:

```python
from semcache.embedders import SentenceTransformerEmbedder
cache = SemanticCache(embedder=SentenceTransformerEmbedder(), threshold=0.9)
```

## How it works

```
prompt ──► embed ──► cosine vs. entries in same namespace ──► best ≥ threshold?
                                                               │yes        │no
                                                          return cached  call LLM,
                                                          (LRU touch)    store result
```

## Benchmark (offline hashing embedder)

`examples/benchmark.py` replays 400 rephrased support questions over 8 intents:

| threshold | hit rate | wrong hits |
|-----------|---------:|-----------:|
| 0.60      | 0.980    | 0          |
| 0.80      | 0.963    | 0          |
| 0.95      | 0.920    | 0          |

The perturbations here are surface-level; true paraphrases need a neural
embedder (see the roadmap for a labelled paraphrase eval).

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

MIT
