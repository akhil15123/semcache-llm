"""Embedding backends.

The default ``HashingEmbedder`` needs no model download: it hashes word and
character n-grams into a fixed-size vector. It is good enough to catch
near-duplicate prompts (reordered words, small typos, casing) and keeps the
test-suite fully offline. For real paraphrase detection, plug in
``SentenceTransformerEmbedder``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Feature-hashing embedder over word unigrams/bigrams and char trigrams."""

    def __init__(self, dim: int = 512, char_ngram: int = 3, stopwords: set[str] | None = None):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.char_ngram = char_ngram
        self.stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in self.stopwords]
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in tokens]
        feats += [(f"b:{a}_{b}", 0.5) for a, b in zip(tokens, tokens[1:])]
        n = self.char_ngram
        for t in tokens:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i + n]}", 0.3) for i in range(max(1, len(padded) - n + 1))]
        return feats

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                idx, sign = self._bucket(feat)
                out[row, idx] += sign * weight
        return _l2_normalise(out)


class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers model (``pip install semcache-llm[st]``)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


_DEFAULT_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "please", "can", "you", "me", "i", "my", "do", "does",
    "what", "whats", "tell", "about", "could", "would", "it",
}
