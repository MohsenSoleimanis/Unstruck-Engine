"""Built-in tools — filesystem, HTTP, database, JSON/CSV, RAG ingest.

Each tool is a plain function. Registered with the ToolRegistry at startup.
All filesystem tools go through the sandbox. All DB tools validate SQL.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from unstruck.tools.sandbox import SandboxError, resolve_path, validate_sql


# ── Filesystem (sandboxed) ──────────────────────────────────────

def fs_read(path: str, sandbox_root: str = "./data", encoding: str = "utf-8") -> dict[str, Any]:
    """Read a file from the sandboxed directory."""
    try:
        resolved = resolve_path(path, Path(sandbox_root))
    except SandboxError as e:
        return {"error": str(e)}
    if not resolved.exists():
        return {"error": f"File not found: {path}"}
    if not resolved.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        return {"content": resolved.read_text(encoding=encoding, errors="replace"), "path": str(resolved)}
    except Exception as e:
        return {"error": str(e)}


def fs_write(path: str, content: str, sandbox_root: str = "./data", encoding: str = "utf-8") -> dict[str, Any]:
    """Write content to a file in the sandboxed directory."""
    try:
        resolved = resolve_path(path, Path(sandbox_root))
    except SandboxError as e:
        return {"error": str(e)}
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
        return {"written": True, "path": str(resolved), "size": resolved.stat().st_size}
    except Exception as e:
        return {"error": str(e)}


def fs_list(directory: str, sandbox_root: str = "./data", pattern: str = "*") -> dict[str, Any]:
    """List files in a sandboxed directory."""
    try:
        resolved = resolve_path(directory, Path(sandbox_root))
    except SandboxError as e:
        return {"error": str(e)}
    if not resolved.exists():
        return {"error": f"Directory not found: {directory}"}
    try:
        files = [
            {"name": f.name, "is_dir": f.is_dir(), "size": f.stat().st_size if f.is_file() else 0}
            for f in sorted(resolved.glob(pattern))
        ]
        return {"files": files, "count": len(files)}
    except Exception as e:
        return {"error": str(e)}


# ── HTTP ────────────────────────────────────────────────────────

async def http_get(url: str, headers: dict | None = None, timeout: int = 30) -> dict[str, Any]:
    """Make an HTTP GET request."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers or {})
            return {
                "status_code": response.status_code,
                "body": response.text[:10000],
                "url": str(response.url),
            }
    except Exception as e:
        return {"error": str(e)}


async def http_post(url: str, body: dict | None = None, headers: dict | None = None, timeout: int = 30) -> dict[str, Any]:
    """Make an HTTP POST request with JSON body."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body or {}, headers=headers or {})
            return {
                "status_code": response.status_code,
                "body": response.text[:10000],
                "url": str(response.url),
            }
    except Exception as e:
        return {"error": str(e)}


# ── Database (SQL validated) ────────────────────────────────────

def db_query(database: str, sql: str, params: list | None = None) -> dict[str, Any]:
    """Execute a SELECT query against a SQLite database."""
    error = validate_sql(sql, ["SELECT", "PRAGMA", "EXPLAIN"], ["DROP", "TRUNCATE", "DELETE", "INSERT", "UPDATE", "ALTER", "EXEC"])
    if error:
        return {"error": f"SQL validation: {error}"}
    try:
        with sqlite3.connect(database) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params or [])
            rows = [dict(row) for row in cursor.fetchall()]
            columns = [d[0] for d in cursor.description] if cursor.description else []
            return {"rows": rows, "columns": columns, "count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


def db_execute(database: str, sql: str, params: list | None = None) -> dict[str, Any]:
    """Execute a SQL write statement."""
    error = validate_sql(sql, ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER"], ["DROP", "TRUNCATE", "EXEC"])
    if error:
        return {"error": f"SQL validation: {error}"}
    try:
        with sqlite3.connect(database) as conn:
            cursor = conn.execute(sql, params or [])
            conn.commit()
            return {"rowcount": cursor.rowcount, "lastrowid": cursor.lastrowid}
    except Exception as e:
        return {"error": str(e)}


# ── JSON/CSV ────────────────────────────────────────────────────

def json_read(path: str, sandbox_root: str = "./data") -> dict[str, Any]:
    """Read and parse a JSON file."""
    try:
        resolved = resolve_path(path, Path(sandbox_root))
    except SandboxError as e:
        return {"error": str(e)}
    if not resolved.exists():
        return {"error": f"File not found: {path}"}
    try:
        return {"data": json.loads(resolved.read_text(encoding="utf-8")), "path": str(resolved)}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}


def csv_read(path: str, sandbox_root: str = "./data", delimiter: str = ",") -> dict[str, Any]:
    """Read a CSV file."""
    import csv
    try:
        resolved = resolve_path(path, Path(sandbox_root))
    except SandboxError as e:
        return {"error": str(e)}
    if not resolved.exists():
        return {"error": f"File not found: {path}"}
    try:
        with open(resolved, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f, delimiter=delimiter))
        headers = rows[0] if rows else []
        return {"headers": headers, "rows": rows[1:] if len(rows) > 1 else [], "total_rows": max(len(rows) - 1, 0)}
    except Exception as e:
        return {"error": str(e)}


# ── Registration helper ─────────────────────────────────────────

def register_builtin_tools(registry, tools_config: dict[str, dict[str, Any]], data_dir: str = "./data") -> None:
    """Register all built-in tools from config/tools.yaml."""
    from unstruck.tools.registry import ToolRegistry

    handlers = {
        "fs_read": lambda **kw: fs_read(sandbox_root=data_dir, **kw),
        "fs_write": lambda **kw: fs_write(sandbox_root=data_dir, **kw),
        "fs_list": lambda **kw: fs_list(sandbox_root=data_dir, **kw),
        "http_get": http_get,
        "http_post": http_post,
        "db_query": db_query,
        "db_execute": db_execute,
        "json_read": lambda **kw: json_read(sandbox_root=data_dir, **kw),
        "csv_read": lambda **kw: csv_read(sandbox_root=data_dir, **kw),
    }

    for name, cfg in tools_config.items():
        handler = handlers.get(name)
        if handler:
            registry.register(
                name=name,
                description=cfg.get("description", name),
                handler=handler,
                permission_level=cfg.get("permission_level", "read"),
                sandbox=cfg.get("sandbox", {}),
            )
