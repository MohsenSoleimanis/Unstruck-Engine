"""Tool registry — manages tool registration, schema validation, and dispatch.

Every external action goes through here. Tools are registered from
config/tools.yaml and can also be added at runtime (MCP servers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger()

# Tool handler type: receives kwargs, returns result dict
ToolHandler = Callable[..., Any]


class ToolRegistry:
    """
    Central registry for all tools agents can call.

    Each tool has:
      - name, description
      - permission_level (read, write, destructive)
      - sandbox rules (path restrictions, etc.)
      - handler function (sync or async)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        permission_level: str = "read",
        sandbox: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool."""
        self._tools[name] = ToolEntry(
            name=name,
            description=description,
            handler=handler,
            permission_level=permission_level,
            sandbox=sandbox or {},
        )
        logger.debug("tool_registry.registered", tool=name, permission=permission_level)

    async def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Call a tool by name. Returns result dict."""
        if name not in self._tools:
            return {"error": f"Tool '{name}' not found. Available: {list(self._tools.keys())}"}

        tool = self._tools[name]
        try:
            result = tool.handler(**kwargs)
            # Support both sync and async handlers
            if hasattr(result, "__await__"):
                result = await result
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """List all tools with their descriptions (for agent prompts)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "permission_level": t.permission_level,
            }
            for t in self._tools.values()
        ]

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def count(self) -> int:
        return len(self._tools)


class ToolEntry:
    """A registered tool."""

    __slots__ = ("name", "description", "handler", "permission_level", "sandbox")

    def __init__(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        permission_level: str,
        sandbox: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.handler = handler
        self.permission_level = permission_level
        self.sandbox = sandbox
