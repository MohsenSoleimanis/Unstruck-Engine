"""Sandbox — path resolution and security for filesystem tools.

All filesystem operations go through resolve_path() which ensures
the path stays within the sandbox root. Prevents path traversal attacks.
"""

from __future__ import annotations

from pathlib import Path


class SandboxError(Exception):
    """Raised when a path escapes the sandbox."""


def resolve_path(path: str, sandbox_root: Path) -> Path:
    """
    Resolve a path safely within a sandbox directory.

    Raises SandboxError if the resolved path escapes the sandbox.
    """
    resolved = (sandbox_root / path).resolve()
    root_resolved = sandbox_root.resolve()

    if not str(resolved).startswith(str(root_resolved)):
        raise SandboxError(f"Path '{path}' escapes sandbox root '{sandbox_root}'")

    return resolved


def validate_sql(sql: str, allowed_prefixes: list[str], blocked_keywords: list[str]) -> str | None:
    """
    Validate a SQL statement.

    Returns error message if invalid, None if valid.
    """
    normalized = " ".join(sql.strip().split()).upper()

    if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        return f"Statement must start with: {', '.join(allowed_prefixes)}"

    for keyword in blocked_keywords:
        if keyword.upper() in normalized:
            return f"Blocked keyword: {keyword}"

    return None
