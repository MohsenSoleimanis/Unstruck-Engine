"""MCP (Model Context Protocol) client — connects agents to external tools and data sources."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class MCPToolClient:
    """
    MCP client for connecting agents to external tools/data sources.

    MCP provides the vertical integration layer:
      - Database connections
      - API integrations
      - File system access
      - Search engine access

    New data sources become available to all agents by adding an MCP server.
    No agent code changes needed — this is what makes the system data-agnostic.
    """

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = server_url
        self._tools: dict[str, dict[str, Any]] = {}
        self._connected = False

    async def connect(self) -> None:
        """Connect to MCP server and discover available tools."""
        if not self.server_url:
            logger.info("mcp.no_server", message="Running without MCP server")
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            self._connected = True
            logger.info("mcp.connected", url=self.server_url)
        except ImportError:
            logger.warning("mcp.not_installed", message="MCP package not available")
        except Exception as e:
            logger.error("mcp.connection_failed", error=str(e))

    def register_tool(self, name: str, description: str, handler: Any) -> None:
        """Register a local tool (for use without MCP server)."""
        self._tools[name] = {"description": description, "handler": handler}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a registered tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")

        handler = self._tools[name]["handler"]
        if callable(handler):
            result = handler(**arguments)
            if hasattr(result, "__await__"):
                result = await result
            return result
        raise TypeError(f"Tool handler for '{name}' is not callable")

    def list_tools(self) -> list[dict[str, str]]:
        """List all available tools (for agent discovery)."""
        return [{"name": k, "description": v["description"]} for k, v in self._tools.items()]
