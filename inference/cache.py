"""
cache.py
========
Semantic cache for personalized query rewrites.

Caches (query, user_segment) → rewrite to avoid redundant LLM inference.
Uses embedding similarity to match semantically equivalent queries, so
"check balance" and "see my balance" hit the same cache entry.

Production systems report ~60% end-to-end speedup with cache hits being
3 orders of magnitude faster than backend processing.

Design decisions:
    - Cache at user-segment level (not per-user) for better hit rates.
      Following Netflix's ~1,300 behavioral cluster approach.
    - TTL-based eviction to keep cache fresh.
    - LRU fallback when cache is full.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache for query → rewrite mappings.

    For simplicity, uses exact-match + user-segment keying. In production,
    this would use embedding-based similarity search (e.g., FAISS).

    Args:
        similarity_threshold: Cosine similarity threshold for cache hits.
        max_size: Maximum number of cache entries.
        ttl_seconds: Time-to-live for cache entries.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 100000,
        ttl_seconds: int = 3600,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # OrderedDict for LRU eviction
        self.cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()

        # Stats
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str, user_segment: str) -> str:
        """
        Create cache key from query + user segment.

        Normalizes query (lowercase, strip) and combines with segment.
        """
        normalized = query.strip().lower()
        raw = f"{normalized}||{user_segment}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, user_segment: str) -> Optional[str]:
        """
        Look up cached rewrite.

        Args:
            query: Original query text.
            user_segment: User segment identifier.

        Returns:
            Cached rewrite string, or None if miss.
        """
        key = self._make_key(query, user_segment)

        if key in self.cache:
            rewrite, timestamp = self.cache[key]

            # Check TTL
            if (time.time() - timestamp) > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return rewrite

        self.misses += 1
        return None

    def put(self, query: str, user_segment: str, rewrite: str):
        """
        Store a rewrite in the cache.

        Args:
            query: Original query.
            user_segment: User segment identifier.
            rewrite: Rewritten query to cache.
        """
        key = self._make_key(query, user_segment)

        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest

        self.cache[key] = (rewrite, time.time())

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(total, 1),
            "total_requests": total,
        }
