"""semcache: a semantic cache for LLM calls."""

from .cache import CacheHit, SemanticCache
from .embedders import Embedder, HashingEmbedder
from .stats import CacheStats

__all__ = ["SemanticCache", "CacheHit", "CacheStats", "Embedder", "HashingEmbedder"]
__version__ = "0.1.0"
