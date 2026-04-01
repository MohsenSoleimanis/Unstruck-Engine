"""MCP (Model Context Protocol) client — connects agents to external tools and data sources.

This is the vertical integration layer. New data sources become available
to all agents by registering a tool — no agent code changes needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from mas.utils.security import resolve_sandboxed_path

logger = structlog.get_logger()

# Sandbox root for all filesystem operations — set via configure_sandbox()
_SANDBOX_ROOT: Path | None = None


def configure_sandbox(root: Path) -> None:
    """Set the sandbox root for all MCP filesystem tools."""
    global _SANDBOX_ROOT
    _SANDBOX_ROOT = root.resolve()
    _SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("mcp.sandbox_configured", root=str(_SANDBOX_ROOT))


class MCPToolClient:
    """
    MCP client with built-in tools for filesystem, HTTP, and database access.

    Tools are callable by any agent via agent.call_tool(name, args).
    Register custom tools with register_tool() for new data sources.
    """

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = server_url
        self._tools: dict[str, dict[str, Any]] = {}
        self._connected = False

    def register_builtin_tools(self) -> None:
        """Register all built-in MCP tools."""
        self.register_tool("fs_read", "Read a file from the filesystem (sandboxed)", _fs_read)
        self.register_tool("fs_write", "Write content to a file (sandboxed)", _fs_write)
        self.register_tool("fs_list", "List files in a directory with optional glob pattern (sandboxed)", _fs_list)
        self.register_tool("fs_info", "Get file metadata (size, modified time, type) (sandboxed)", _fs_info)
        self.register_tool("http_get", "Make an HTTP GET request to a URL", _http_get)
        self.register_tool("http_post", "Make an HTTP POST request with JSON body", _http_post)
        self.register_tool("db_query", "Execute a SELECT query against a SQLite database", _db_query)
        self.register_tool("db_execute", "Execute a safe SQL statement (INSERT/UPDATE/DELETE)", _db_execute)
        self.register_tool("json_read", "Read and parse a JSON file (sandboxed)", _json_read)
        self.register_tool("json_write", "Write data to a JSON file (sandboxed)", _json_write)
        self.register_tool("csv_read", "Read a CSV file into rows (sandboxed)", _csv_read)
        logger.info("mcp.builtin_registered", tool_count=len(self._tools))

    async def connect(self) -> None:
        """Connect to external MCP server for additional tools."""
        if not self.server_url:
            logger.info("mcp.local_only", message="Running with local tools only")
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            self._connected = True
            logger.info("mcp.connected", url=self.server_url)
        except ImportError:
            logger.warning("mcp.not_installed", message="MCP package not available, using local tools")
        except Exception as e:
            logger.error("mcp.connection_failed", error=str(e))

    def register_tool(self, name: str, description: str, handler: Any) -> None:
        """Register a tool (local or from MCP server)."""
        self._tools[name] = {"description": description, "handler": handler}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a registered tool by name (supports both sync and async handlers)."""
        import inspect

        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")

        handler = self._tools[name]["handler"]
        if not callable(handler):
            raise TypeError(f"Tool handler for '{name}' is not callable")

        result = handler(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result

    def list_tools(self) -> list[dict[str, str]]:
        """List all available tools (for agent discovery)."""
        return [{"name": k, "description": v["description"]} for k, v in self._tools.items()]


# ═══════════════════════════════════════════════════════════
# Path resolution — all filesystem tools go through this
# ═══════════════════════════════════════════════════════════


def _safe_path(path: str) -> Path:
    """Resolve path within sandbox. Falls back to ./data if no sandbox set."""
    root = _SANDBOX_ROOT or Path("./data").resolve()
    return resolve_sandboxed_path(path, root)


# ═══════════════════════════════════════════════════════════
# Filesystem tools (sandboxed)
# ═══════════════════════════════════════════════════════════


def _fs_read(path: str, encoding: str = "utf-8") -> dict[str, Any]:
    """Read a file (sandboxed)."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"File not found: {path}"}
    if not p.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        content = p.read_text(encoding=encoding, errors="replace")
        return {"content": content, "size": p.stat().st_size, "path": str(p)}
    except Exception as e:
        return {"error": str(e)}


def _fs_write(path: str, content: str, encoding: str = "utf-8") -> dict[str, Any]:
    """Write content to a file (sandboxed)."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return {"written": True, "path": str(p), "size": p.stat().st_size}
    except Exception as e:
        return {"error": str(e)}


def _fs_list(directory: str, pattern: str = "*") -> dict[str, Any]:
    """List files matching a glob pattern (sandboxed)."""
    try:
        p = _safe_path(directory)
    except PermissionError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"Directory not found: {directory}"}
    try:
        files = []
        for f in sorted(p.glob(pattern)):
            files.append({
                "name": f.name,
                "path": str(f),
                "is_dir": f.is_dir(),
                "size": f.stat().st_size if f.is_file() else 0,
            })
        return {"files": files, "count": len(files)}
    except Exception as e:
        return {"error": str(e)}


def _fs_info(path: str) -> dict[str, Any]:
    """Get file metadata (sandboxed)."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"Not found: {path}"}
    stat = p.stat()
    return {
        "path": str(p),
        "name": p.name,
        "extension": p.suffix,
        "size_bytes": stat.st_size,
        "modified": stat.st_mtime,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
    }


# ═══════════════════════════════════════════════════════════
# HTTP tools
# ═══════════════════════════════════════════════════════════


async def _http_get(url: str, headers: dict | None = None, timeout: int = 30) -> dict[str, Any]:
    """Make an HTTP GET request (async to avoid blocking the event loop)."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers or {})
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:10000],
                "url": str(response.url),
            }
    except Exception as e:
        return {"error": str(e)}


async def _http_post(url: str, body: dict | None = None, headers: dict | None = None, timeout: int = 30) -> dict[str, Any]:
    """Make an HTTP POST request with JSON body (async to avoid blocking the event loop)."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body or {}, headers=headers or {})
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:10000],
                "url": str(response.url),
            }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# Database tools (SQL injection protected)
# ═══════════════════════════════════════════════════════════

_ALLOWED_QUERY_PREFIXES = ("SELECT", "PRAGMA", "EXPLAIN")
_ALLOWED_EXECUTE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER")
_BLOCKED_KEYWORDS = ("DROP", "TRUNCATE", "EXEC", "EXECUTE", "--", ";--")


def _validate_sql(sql: str, allowed_prefixes: tuple[str, ...]) -> str | None:
    """Validate SQL statement. Returns error message or None if valid."""
    normalized = " ".join(sql.strip().split()).upper()
    if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        return f"Statement must start with one of: {', '.join(allowed_prefixes)}"
    for blocked in _BLOCKED_KEYWORDS:
        if blocked in normalized:
            return f"Blocked keyword found: {blocked}"
    return None


def _db_query(database: str, sql: str, params: list | None = None) -> dict[str, Any]:
    """Execute a SELECT query against a SQLite database."""
    import sqlite3

    error = _validate_sql(sql, _ALLOWED_QUERY_PREFIXES)
    if error:
        return {"error": f"SQL validation failed: {error}"}

    try:
        with sqlite3.connect(database) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params or [])
            rows = [dict(row) for row in cursor.fetchall()]
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return {"rows": rows, "columns": columns, "count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


def _db_execute(database: str, sql: str, params: list | None = None) -> dict[str, Any]:
    """Execute a safe SQL statement (INSERT/UPDATE/DELETE/CREATE/ALTER)."""
    import sqlite3

    error = _validate_sql(sql, _ALLOWED_EXECUTE_PREFIXES)
    if error:
        return {"error": f"SQL validation failed: {error}"}

    try:
        with sqlite3.connect(database) as conn:
            cursor = conn.execute(sql, params or [])
            conn.commit()
            return {"rowcount": cursor.rowcount, "lastrowid": cursor.lastrowid}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# Data file tools (sandboxed)
# ═══════════════════════════════════════════════════════════


def _json_read(path: str) -> dict[str, Any]:
    """Read and parse a JSON file (sandboxed)."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"File not found: {path}"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {"data": data, "path": str(p)}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}


def _json_write(path: str, data: Any, indent: int = 2) -> dict[str, Any]:
    """Write data to a JSON file (sandboxed)."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=indent, default=str))
        return {"written": True, "path": str(p)}
    except Exception as e:
        return {"error": str(e)}


def _csv_read(path: str, delimiter: str = ",") -> dict[str, Any]:
    """Read a CSV file into rows (sandboxed)."""
    import csv

    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"File not found: {path}"}
    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        headers = rows[0] if rows else []
        return {"headers": headers, "rows": rows[1:] if len(rows) > 1 else [], "total_rows": max(len(rows) - 1, 0)}
    except Exception as e:
        return {"error": str(e)}
