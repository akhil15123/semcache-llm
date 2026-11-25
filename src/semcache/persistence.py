"""Save and load a SemanticCache to disk.

Layout: a single ``.npz`` file holding the embedding matrix plus a JSON
metadata blob (prompts, responses, namespaces, ages, tokens).

Responses and namespaces must be JSON-serialisable. Tuples (the default
namespace shape produced by ``@cache.cached()``) are round-tripped.
Entry ages are stored instead of absolute timestamps, because the cache
uses a monotonic clock that is meaningless across processes; TTLs keep
counting from where they were when the cache was saved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .cache import SemanticCache

FORMAT_VERSION = 1
_TUPLE_TAG = "__tuple__"


def _encode(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return {_TUPLE_TAG: [_encode(x) for x in obj]}
    if isinstance(obj, list):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {_TUPLE_TAG}:
            return tuple(_decode(x) for x in obj[_TUPLE_TAG])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(x) for x in obj]
    return obj


def save(cache: "SemanticCache", path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    with cache._lock:
        now = cache._clock()
        entries = list(cache._entries.values())  # LRU order, oldest first
        vectors = (
            np.stack([e.vector for e in entries])
            if entries
            else np.zeros((0, cache.embedder.dim), dtype=np.float32)
        )
        meta = {
            "version": FORMAT_VERSION,
            "dim": int(cache.embedder.dim),
            "embedder": type(cache.embedder).__name__,
            "entries": [
                {
                    "prompt": e.prompt,
                    "response": _encode(e.response),
                    "namespace": _encode(e.namespace),
                    "age": now - e.created_at,
                    "tokens": e.tokens,
                }
                for e in entries
            ],
        }
    np.savez_compressed(path, vectors=vectors, meta=np.array(json.dumps(meta)))
    return path


def load_into(cache: "SemanticCache", path: str | Path) -> int:
    """Load entries from ``path`` into ``cache``; returns how many were added.

    Raises ``ValueError`` if the embedding dimension or embedder type does not
    match, since vectors from a different embedder are not comparable.
    """
    from .cache import _Entry

    with np.load(Path(path), allow_pickle=False) as data:
        vectors = data["vectors"]
        meta = json.loads(str(data["meta"]))
    if meta.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported cache file version: {meta.get('version')}")
    if meta["dim"] != cache.embedder.dim or meta["embedder"] != type(cache.embedder).__name__:
        raise ValueError(
            f"cache file was built with {meta['embedder']}(dim={meta['dim']}), "
            f"not {type(cache.embedder).__name__}(dim={cache.embedder.dim})"
        )
    with cache._lock:
        now = cache._clock()
        for vec, m in zip(vectors, meta["entries"]):
            cache._entries[cache._next_id] = _Entry(
                prompt=m["prompt"],
                response=_decode(m["response"]),
                vector=vec.astype(np.float32),
                created_at=now - m["age"],
                namespace=_decode(m["namespace"]),
                tokens=m["tokens"],
            )
            cache._next_id += 1
        while len(cache._entries) > cache.max_entries:
            cache._entries.popitem(last=False)
            cache.stats.evictions += 1
    return len(meta["entries"])
