"""Search tools — web search, semantic search, and file search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class SearchTool:
    """Aggregated search capabilities for agents."""

    @staticmethod
    def search_files(directory: str | Path, pattern: str = "**/*", extensions: list[str] | None = None) -> list[str]:
        """Search for files matching a pattern."""
        path = Path(directory)
        results = []
        for f in path.glob(pattern):
            if f.is_file():
                if extensions and f.suffix.lower() not in extensions:
                    continue
                results.append(str(f))
        return results

    @staticmethod
    def search_content(file_path: str | Path, query: str) -> list[dict[str, Any]]:
        """Search within a file for content matching a query."""
        path = Path(file_path)
        if not path.exists():
            return []

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        query_lower = query.lower()
        results = []
        for i, line in enumerate(text.splitlines(), 1):
            if query_lower in line.lower():
                results.append({"line_number": i, "content": line.strip(), "file": str(path)})
        return results
