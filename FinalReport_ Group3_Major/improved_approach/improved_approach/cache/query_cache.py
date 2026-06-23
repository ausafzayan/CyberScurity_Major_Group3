"""
query_cache.py
==============
Redis-backed query result cache — Improvement 4 over the paper's stateless pipeline.

The paper had no caching, causing 25–50 second wait times even for repeated queries.
This module caches RAG results in Redis (an in-memory key-value store) so that
identical or near-identical queries are returned instantly from cache.

Key concepts:
  Redis      : Remote Dictionary Server — an open-source in-memory data store.
                Keys and values are stored in RAM for O(1) lookup speed.
                Data can be persisted to disk and given an expiry time (TTL).
  SHA-256    : Cryptographic hash function. Used here to create a short,
                fixed-length cache key from the (query, context) pair.
  TTL        : Time-to-live — Redis automatically deletes keys after TTL seconds.
  In-memory fallback: a plain Python dict used when Redis is unavailable.

Libraries used:
  - redis    : Python Redis client
  - hashlib  : SHA-256 hashing (stdlib)
  - json     : serialise/deserialise cached dicts
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

import redis  # Python Redis client — connects to a running Redis server

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cache key prefix — namespaces our keys inside Redis to avoid conflicts
KEY_PREFIX = "tm:"  # threat modeling namespace

# Default TTL: 1 hour (3600 seconds)
DEFAULT_TTL = 3600


class QueryCache:
    """
    Two-tier cache: Redis (preferred) with in-memory dict fallback.

    Cache key construction:
        key = "tm:" + SHA-256(query + ":" + context_hash)[:16]

    This ensures:
      - Same query on different documents gets different keys (context_hash differs)
      - Keys are compact (16 hex chars after the prefix)
      - Collisions are cryptographically negligible

    Usage:
        cache = QueryCache()
        cached = cache.get(question)
        if cached:
            return cached          # instant — no LLM call needed
        result = llm.answer(question)
        cache.set(question, result)
    """

    def __init__(
        self,
        use_redis: bool  = True,
        redis_url: str   = "redis://localhost:6379",
    ) -> None:
        """
        Connect to Redis or fall back to an in-memory dict.

        Args:
            use_redis : set False to skip Redis and always use in-memory cache
            redis_url : Redis connection URL (e.g. "redis://localhost:6379")
        """
        self._memory: Dict[str, Any] = {}  # in-memory fallback store

        if use_redis:
            try:
                # Connect to Redis using the provided URL
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()  # test that the server is reachable
                self._use_redis = True
                logger.info("Redis cache connected at %s ✓", redis_url)
            except Exception as exc:
                logger.warning(
                    "Redis not reachable (%s). Using in-memory dict cache instead.", exc
                )
                self._use_redis = False
        else:
            self._use_redis = False
            logger.info("Using in-memory cache (Redis disabled).")

    # ── Key construction ───────────────────────────────────────────────────────

    def _make_key(self, query: str, context_hash: str = "") -> str:
        """
        Build a compact, collision-resistant cache key.

        SHA-256 of "query:context_hash" → truncate to 16 hex characters.

        Args:
            query        : the question string
            context_hash : optional fingerprint of the document context
                           (e.g. hash of the PDF filename + mtime)

        Returns:
            Cache key string like "tm:a1b2c3d4e5f60718"
        """
        combined = f"{query}:{context_hash}"
        sha256   = hashlib.sha256(combined.encode()).hexdigest()
        return KEY_PREFIX + sha256[:16]   # 16 hex chars = 64 bits of uniqueness

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, query: str, context_hash: str = "") -> Optional[Dict]:
        """
        Retrieve a cached result if it exists.

        Args:
            query        : the question string
            context_hash : optional document fingerprint

        Returns:
            Cached result dict, or None if not cached / expired
        """
        key = self._make_key(query, context_hash)

        if self._use_redis:
            raw = self._redis.get(key)         # returns None if key absent or expired
            return json.loads(raw) if raw else None
        else:
            return self._memory.get(key)       # plain dict lookup

    def set(
        self,
        query:        str,
        result:       Dict,
        context_hash: str = "",
        ttl:          int = DEFAULT_TTL,
    ) -> None:
        """
        Cache a result.

        Args:
            query        : the question string
            result       : the dict to cache (must be JSON-serialisable)
            context_hash : optional document fingerprint
            ttl          : seconds until this entry expires (Redis only)
        """
        key = self._make_key(query, context_hash)

        if self._use_redis:
            # setex = set + expiry in one atomic operation
            self._redis.setex(key, ttl, json.dumps(result))
        else:
            self._memory[key] = result          # no TTL in memory cache

    def clear(self) -> None:
        """
        Remove all threat-modeling cache entries (those starting with KEY_PREFIX).
        Useful for testing or after ingesting new documents.
        """
        if self._use_redis:
            # SCAN iterates keys without blocking Redis
            for key in self._redis.scan_iter(f"{KEY_PREFIX}*"):
                self._redis.delete(key)
            logger.info("Redis cache cleared (prefix=%s).", KEY_PREFIX)
        else:
            self._memory.clear()
            logger.info("In-memory cache cleared.")
