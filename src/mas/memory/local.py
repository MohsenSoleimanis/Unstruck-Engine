"""Agent-local cache memory — fast, limited-capacity working memory per agent."""

from __future__ import annotations

import time
from typing import Any


class LocalMemory:
    """
    Per-agent working memory (Layer 2 — Cache Layer from the 3-layer hierarchy).

    Provides fast key-value storage with TTL expiration for task-specific context.
    Each agent gets its own LocalMemory to avoid cross-contamination.
    """

    def __init__(self, namespace: str, default_ttl: int = 3600) -> None:
        self.namespace = namespace
        self.default_ttl = default_ttl
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires_at = time.time() + (ttl or self.default_ttl)
        self._store[key] = (value, expires_at)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._store:
            return default
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return default
        return value

    def has(self, key: str) -> bool:
        if key not in self._store:
            return False
        _, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return False
        return True

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def get_context(self) -> dict[str, Any]:
        """Return all non-expired entries as a context dict."""
        now = time.time()
        return {k: v for k, (v, exp) in self._store.items() if now <= exp}

    @property
    def size(self) -> int:
        return len(self._store)
