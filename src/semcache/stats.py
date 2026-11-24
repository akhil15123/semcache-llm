from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    tokens_saved: int = 0
    similarity_on_hit: list[float] = field(default_factory=list)

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict:
        sims = self.similarity_on_hit
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "expirations": self.expirations,
            "tokens_saved": self.tokens_saved,
            "mean_hit_similarity": round(sum(sims) / len(sims), 4) if sims else None,
        }
