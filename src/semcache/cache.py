"""Core semantic cache."""

from __future__ import annotations

import functools
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable

import numpy as np

from .embedders import Embedder, HashingEmbedder
from .stats import CacheStats


@dataclass
class _Entry:
    prompt: str
    response: Any
    vector: np.ndarray
    created_at: float
    namespace: Hashable
    tokens: int = 0


@dataclass(frozen=True)
class CacheHit:
    response: Any
    similarity: float
    cached_prompt: str


class SemanticCache:
    """Cache LLM responses keyed by the *meaning* of the prompt.

    A lookup embeds the prompt and returns the stored response of the most
    similar earlier prompt if cosine similarity >= ``threshold``. Entries are
    partitioned by ``namespace`` (e.g. model name + temperature) so a cached
    answer from one model or sampling config is never served for another.

    Args:
        embedder: anything with ``.dim`` and ``.embed(list[str]) -> (n, dim)``.
        threshold: minimum cosine similarity for a hit.
        max_entries: LRU capacity; least-recently-used entries are evicted.
        ttl_seconds: entries older than this are treated as missing.
        clock: injectable time source (handy for tests).
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = 0.85,
        max_entries: int = 10_000,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._next_id = 0
        self._lock = threading.RLock()
        self.stats = CacheStats()

    # ------------------------------------------------------------------ utils
    def __len__(self) -> int:
        return len(self._entries)

    def _expired(self, entry: _Entry, now: float) -> bool:
        return self.ttl_seconds is not None and now - entry.created_at > self.ttl_seconds

    def _purge_expired(self, now: float) -> None:
        dead = [k for k, e in self._entries.items() if self._expired(e, now)]
        for k in dead:
            del self._entries[k]
        self.stats.expirations += len(dead)

    def _embed_one(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]

    # ------------------------------------------------------------------ API
    def lookup(self, prompt: str, namespace: Hashable = "default") -> CacheHit | None:
        vec = self._embed_one(prompt)
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            candidates = [(k, e) for k, e in self._entries.items() if e.namespace == namespace]
            if not candidates:
                self.stats.misses += 1
                return None
            keys, entries = zip(*candidates)
            sims = np.stack([e.vector for e in entries]) @ vec
            best = int(np.argmax(sims))
            score = float(sims[best])
            if score < self.threshold:
                self.stats.misses += 1
                return None
            key, entry = keys[best], entries[best]
            self._entries.move_to_end(key)  # LRU touch
            self.stats.hits += 1
            self.stats.tokens_saved += entry.tokens
            self.stats.similarity_on_hit.append(score)
            return CacheHit(entry.response, score, entry.prompt)

    def store(self, prompt: str, response: Any, namespace: Hashable = "default", tokens: int = 0) -> None:
        vec = self._embed_one(prompt)
        with self._lock:
            self._entries[self._next_id] = _Entry(prompt, response, vec, self._clock(), namespace, tokens)
            self._next_id += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.stats.evictions += 1

    def get_or_call(
        self,
        prompt: str,
        fn: Callable[[str], Any],
        namespace: Hashable = "default",
        token_counter: Callable[[Any], int] | None = None,
    ) -> Any:
        hit = self.lookup(prompt, namespace)
        if hit is not None:
            return hit.response
        response = fn(prompt)
        tokens = token_counter(response) if token_counter else 0
        self.store(prompt, response, namespace, tokens)
        return response

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def cached(self, namespace_fn: Callable[..., Hashable] | None = None):
        """Decorator for ``fn(prompt, **kwargs)``. Kwargs feed the namespace.

        By default the namespace is the sorted kwargs, so calls with different
        ``model=`` or ``temperature=`` never share cache entries.
        """

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(prompt: str, **kwargs):
                ns = namespace_fn(**kwargs) if namespace_fn else tuple(sorted(kwargs.items()))
                return self.get_or_call(prompt, lambda p: fn(p, **kwargs), namespace=ns)

            return wrapper

        return decorator
