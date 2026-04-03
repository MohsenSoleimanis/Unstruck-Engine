"""RAG tools — registers rag_ingest and rag_query with the tool registry.

These are the tools that agents call to interact with RAG-Anything.
The tools delegate to the RAGService singleton.
"""

from __future__ import annotations

from typing import Any

from unstruck.rag.service import RAGService
from unstruck.tools.registry import ToolRegistry


def register_rag_tools(tool_registry: ToolRegistry, rag_service: RAGService) -> None:
    """Register RAG ingest and query tools."""

    async def rag_ingest(file_path: str = "", doc_id: str = "") -> dict[str, Any]:
        """Ingest a document via RAG-Anything."""
        if not file_path:
            return {"error": "file_path is required"}
        return await rag_service.ingest(file_path, doc_id=doc_id or None)

    async def rag_query(query: str = "", mode: str = "mix") -> dict[str, Any]:
        """Query the RAG knowledge graph."""
        if not query:
            return {"error": "query is required"}
        return await rag_service.query(query, mode=mode)

    tool_registry.register(
        name="rag_ingest",
        description="Ingest a document via RAG-Anything (parse, build KG, index)",
        handler=rag_ingest,
        permission_level="write",
    )

    tool_registry.register(
        name="rag_query",
        description="Query RAG-Anything knowledge graph for relevant context",
        handler=rag_query,
        permission_level="read",
    )
