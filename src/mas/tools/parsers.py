"""Document parser utilities — deterministic parsing tools for agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DocumentParser:
    """
    Pluggable document parser (factory pattern from RAG-Anything).

    Supports registration of custom parsers for new file types.
    """

    _parsers: dict[str, Any] = {}

    @classmethod
    def register_parser(cls, extension: str, parser_fn: Any) -> None:
        cls._parsers[extension.lower()] = parser_fn

    @classmethod
    def parse(cls, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in cls._parsers:
            return cls._parsers[ext](path)

        # Default: read as text
        if ext in (".txt", ".md", ".csv", ".json", ".html", ".xml"):
            return {
                "type": "text",
                "content": path.read_text(encoding="utf-8", errors="replace"),
                "file_type": ext,
            }

        return {"type": "binary", "file_type": ext, "size": path.stat().st_size}

    @classmethod
    def supported_types(cls) -> list[str]:
        return list(cls._parsers.keys()) + [".txt", ".md", ".csv", ".json", ".html", ".xml"]
