"""Security utilities — path sandboxing, label sanitization, input validation."""

from __future__ import annotations

import re
from pathlib import Path


def resolve_sandboxed_path(path: str, sandbox_root: Path) -> Path:
    """
    Resolve a path safely within a sandbox directory.

    Prevents path traversal attacks (../../etc/passwd).
    Raises PermissionError if the resolved path escapes the sandbox.
    """
    resolved = (sandbox_root / path).resolve()
    sandbox_resolved = sandbox_root.resolve()
    if not str(resolved).startswith(str(sandbox_resolved)):
        raise PermissionError(f"Path escapes sandbox: {path}")
    return resolved


def safe_label(name: str) -> str:
    """
    Sanitize a string for use as a Neo4j label or relationship type.

    Prevents Cypher injection by restricting to alphanumeric + underscore.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "N_" + sanitized
    return sanitized


def sanitize_filename(filename: str) -> str:
    """
    Extract just the filename, stripping any directory components.

    Prevents path traversal via crafted filenames in uploads.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    # Strip both Unix and Windows path components
    name = PurePosixPath(filename).name
    name = PureWindowsPath(name).name
    if not name or name in (".", ".."):
        raise ValueError(f"Invalid filename: {filename}")
    return name
