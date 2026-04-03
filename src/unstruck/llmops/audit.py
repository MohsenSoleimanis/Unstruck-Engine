"""Audit log — records every action for compliance and debugging.

Registers as multiple hooks to capture who did what, when, and why.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from unstruck.hooks import HookEvent, HookManager, HookResult


class AuditLog:
    """
    Append-only audit trail.

    Records LLM calls, tool calls, and session events.
    Each entry: who (agent), what (action), when, result, cost.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)

    def register_hooks(self, hooks: HookManager) -> None:
        hooks.register(HookEvent.POST_LLM_CALL, self._log_llm_call)
        hooks.register(HookEvent.POST_TOOL_USE, self._log_tool_call)
        hooks.register(HookEvent.SESSION_START, self._log_session_start)
        hooks.register(HookEvent.SESSION_END, self._log_session_end)

    async def _log_llm_call(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        self._entries.append({
            "type": "llm_call",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": context.get("agent_id"),
            "model": context.get("model"),
            "input_tokens": context.get("input_tokens"),
            "output_tokens": context.get("output_tokens"),
            "cost_usd": context.get("cost_usd"),
        })
        return HookResult.allow()

    async def _log_tool_call(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        self._entries.append({
            "type": "tool_call",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": context.get("agent_id"),
            "tool": context.get("tool_name"),
            "result_status": "success" if not context.get("error") else "error",
        })
        return HookResult.allow()

    async def _log_session_start(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        self._entries.append({
            "type": "session_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": context.get("session_id"),
        })
        return HookResult.allow()

    async def _log_session_end(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        self._entries.append({
            "type": "session_end",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": context.get("session_id"),
        })
        return HookResult.allow()

    def get_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._entries)[-limit:]

    @property
    def count(self) -> int:
        return len(self._entries)
