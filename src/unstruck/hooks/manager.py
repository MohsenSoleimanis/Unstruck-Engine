"""Hook system — lifecycle extension points.

Hooks are how Auth, Permissions, Guardrails, Memory, Cost Control, and Audit
connect to the core flow without being hardcoded into it.

Each hook fires at a specific moment in the pipeline. Handlers can:
  - Observe (logging, metrics, audit)
  - Modify (change prompts, args, outputs)
  - Block (reject unsafe inputs, deny tool calls)

Hook handlers are registered at startup, either by platform code or by
enterprise customization. They run in registration order.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class HookEvent(str, Enum):
    """All lifecycle events where hooks can fire."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    PRE_COMPRESS = "pre_compress"


class HookAction(str, Enum):
    """What a hook handler can decide."""

    ALLOW = "allow"      # Proceed normally
    MODIFY = "modify"    # Proceed with modified data (data in result)
    BLOCK = "block"      # Stop the operation (reason in result)


class HookResult:
    """Result from a hook handler."""

    __slots__ = ("action", "data", "reason")

    def __init__(
        self,
        action: HookAction = HookAction.ALLOW,
        data: dict[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        self.action = action
        self.data = data
        self.reason = reason

    @staticmethod
    def allow() -> HookResult:
        return HookResult(HookAction.ALLOW)

    @staticmethod
    def modify(data: dict[str, Any]) -> HookResult:
        return HookResult(HookAction.MODIFY, data=data)

    @staticmethod
    def block(reason: str) -> HookResult:
        return HookResult(HookAction.BLOCK, reason=reason)


# Type alias for hook handlers
# Handler receives (event, context_data) → returns HookResult
HookHandler = Callable[[HookEvent, dict[str, Any]], Awaitable[HookResult]]


class HookManager:
    """
    Manages hook registration and execution.

    Usage:
        hooks = HookManager()
        hooks.register(HookEvent.PRE_TOOL_USE, my_permission_check)
        hooks.register(HookEvent.POST_LLM_CALL, my_pii_detector)

        result = await hooks.fire(HookEvent.PRE_TOOL_USE, {"tool": "fs_write", "args": {...}})
        if result.action == HookAction.BLOCK:
            # Tool call denied
    """

    def __init__(self) -> None:
        self._handlers: dict[HookEvent, list[HookHandler]] = {event: [] for event in HookEvent}

    def register(self, event: HookEvent, handler: HookHandler) -> None:
        """Register a hook handler for an event. Handlers run in registration order."""
        self._handlers[event].append(handler)
        logger.debug("hook.registered", hook_event=event.value, handler=handler.__name__)

    def unregister(self, event: HookEvent, handler: HookHandler) -> None:
        """Remove a handler."""
        handlers = self._handlers[event]
        if handler in handlers:
            handlers.remove(handler)

    async def fire(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """
        Fire all handlers for an event. Returns the aggregate result.

        Handlers run sequentially in registration order.
        If any handler returns BLOCK, execution stops immediately.
        If any handler returns MODIFY, the modified data is passed to subsequent handlers.
        """
        handlers = self._handlers[event]
        if not handlers:
            return HookResult.allow()

        current_context = dict(context)

        for handler in handlers:
            try:
                result = await handler(event, current_context)

                if result.action == HookAction.BLOCK:
                    logger.info(
                        "hook.blocked",
                        hook_event=event.value,
                        handler=handler.__name__,
                        reason=result.reason,
                    )
                    return result

                if result.action == HookAction.MODIFY and result.data:
                    current_context.update(result.data)

            except Exception as e:
                # Hook errors should never crash the pipeline.
                # Log and continue — hooks are best-effort.
                logger.error(
                    "hook.handler_error",
                    hook_event=event.value,
                    handler=handler.__name__,
                    error=str(e),
                )

        # If any handler modified the context, return MODIFY with accumulated changes
        if current_context != context:
            return HookResult.modify(current_context)

        return HookResult.allow()

    def handler_count(self, event: HookEvent) -> int:
        """Number of handlers registered for an event."""
        return len(self._handlers[event])

    @property
    def total_handlers(self) -> int:
        return sum(len(h) for h in self._handlers.values())
