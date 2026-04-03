"""Tier 2: Agent cache — per-agent, TTL-based, checked before every call.

This cache is ENFORCED, not optional. The hooks check it before every
LLM and tool call. Cache hits skip the expensive operation entirely.

Cache key = hash of (agent_id, operation_type, relevant_args).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog

from unstruck.hooks import HookAction, HookEvent, HookManager, HookResult

logger = structlog.get_logger()


class AgentCache:
    """
    Per-agent TTL cache for LLM responses and tool results.

    Reduces cost and latency by returning cached results for
    identical or near-identical requests.

    Registers as:
      - PreLLMCall hook: check if we have a cached response
      - PostLLMCall hook: store the response in cache
      - PreToolUse hook: check if we have a cached tool result
      - PostToolUse hook: store the tool result in cache
    """

    def __init__(self, default_ttl: int = 3600) -> None:
        self._default_ttl = default_ttl
        self._store: dict[str, _CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None if expired or missing."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() > entry.expires_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in cache with TTL."""
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.time() + (ttl or self._default_ttl),
        )

    def clear(self) -> None:
        self._store.clear()

    def register_hooks(self, hooks: HookManager) -> None:
        """Register cache check/store as hooks."""
        hooks.register(HookEvent.PRE_LLM_CALL, self._check_llm_cache)
        hooks.register(HookEvent.POST_LLM_CALL, self._store_llm_cache)
        hooks.register(HookEvent.PRE_TOOL_USE, self._check_tool_cache)
        hooks.register(HookEvent.POST_TOOL_USE, self._store_tool_cache)

    async def _check_llm_cache(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Check if we have a cached LLM response for this exact prompt."""
        key = self._llm_cache_key(context)
        cached = self.get(key)
        if cached is not None:
            logger.debug("cache.llm_hit", agent=context.get("agent_id"))
            # Return cached response by modifying context
            # The Context Engine checks for _cached_response and skips the LLM call
            return HookResult.modify({"_cached_response": cached})
        return HookResult.allow()

    async def _store_llm_cache(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Store LLM response in cache."""
        response = context.get("response", "")
        if response:
            key = self._llm_cache_key(context)
            self.set(key, response)
        return HookResult.allow()

    async def _check_tool_cache(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Check if we have a cached result for this tool call."""
        key = self._tool_cache_key(context)
        cached = self.get(key)
        if cached is not None:
            logger.debug("cache.tool_hit", tool=context.get("tool_name"))
            return HookResult.modify({"_cached_result": cached})
        return HookResult.allow()

    async def _store_tool_cache(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Store tool result in cache."""
        result = context.get("result")
        if result and not context.get("error"):
            key = self._tool_cache_key(context)
            # Shorter TTL for tool results — data changes more often
            self.set(key, result, ttl=600)
        return HookResult.allow()

    def _llm_cache_key(self, context: dict[str, Any]) -> str:
        """Cache key for LLM calls: hash of agent + system prompt + user prompt."""
        parts = [
            context.get("agent_id", ""),
            context.get("system_prompt", ""),
            context.get("user_prompt", ""),
        ]
        return "llm:" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _tool_cache_key(self, context: dict[str, Any]) -> str:
        """Cache key for tool calls: hash of tool name + args."""
        import json
        parts = [
            context.get("tool_name", ""),
            json.dumps(context.get("tool_args", {}), sort_keys=True, default=str),
        ]
        return "tool:" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    @property
    def stats(self) -> dict[str, int]:
        active = sum(1 for e in self._store.values() if time.time() <= e.expires_at)
        return {
            "total_entries": len(self._store),
            "active_entries": active,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1), 3),
        }


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at
