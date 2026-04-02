"""RAG-Anything agent — wraps the RAG engine for ingestion and retrieval.

Replaces 7 separate agents (ingestion, separator, text/image/table processors,
chunker, embedder) with a single agent that delegates to RAG-Anything.

Falls back to legacy ingestion if RAG-Anything is not installed.
"""

from __future__ import annotations

from typing import Any

import structlog

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

logger = structlog.get_logger()

# Will be set by pipeline.py during initialization
_rag_engine = None


def set_rag_engine(engine: Any) -> None:
    """Set the global RAG engine reference (called by MASPipeline)."""
    global _rag_engine
    _rag_engine = engine


@registry.register
class RAGAnythingAgent(BaseAgent):
    """
    Unified ingestion + retrieval agent via RAG-Anything.

    Modes:
      - ingest: parse document, extract entities, build KG, index chunks
      - query: retrieve relevant context using hybrid search

    Falls back to legacy agents if RAG-Anything is not installed.
    """

    agent_type = "raganything"
    description = "Ingests documents and retrieves context via RAG-Anything (multimodal RAG with knowledge graph)"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        ctx = self.get_pipeline_context(task)
        mode = task.context.get("mode", "ingest")

        if mode == "ingest":
            return await self._ingest(task, ctx)
        elif mode == "query":
            return await self._query(task, ctx)
        else:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[f"Unknown mode: {mode}. Use 'ingest' or 'query'."],
            )

    async def _ingest(self, task: Task, ctx: Any) -> AgentResult:
        file_path = task.context.get("file_path", "")
        if not file_path:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=["No file_path in task context"],
            )

        # Try RAG engine (LightRAG + RAG-Anything) — lazy-initializes on first call
        if _rag_engine:
            result = await _rag_engine.ingest_document(file_path)
            if not result.get("fallback"):
                # Also build text_aggregate for the analyst
                text_parts = []
                for item in result.get("content_items", []):
                    if item.get("content"):
                        page = item.get("page_idx", "")
                        prefix = f"[Page {page}] " if page else ""
                        text_parts.append(f"{prefix}{item['content']}")

                return AgentResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    status=ResultStatus.SUCCESS,
                    output={
                        "doc_id": result.get("doc_id", ""),
                        "indexed": result.get("indexed", False),
                        "content_items": result.get("content_items", [])[:50],
                        "text_aggregate": "\n\n".join(text_parts)[:50000],
                        "document": {
                            "file_path": file_path,
                            "file_type": file_path.rsplit(".", 1)[-1] if "." in file_path else "",
                        },
                    },
                )

        # Fallback: use legacy ingestion (PyMuPDF only)
        logger.info("raganything.fallback_to_legacy", file_path=file_path)
        return await self._legacy_ingest(task, file_path)

    async def _query(self, task: Task, ctx: Any) -> AgentResult:
        query = task.instruction
        pipeline_ctx = self.get_pipeline_context(task)

        # Try RAG engine (LightRAG KG-based retrieval)
        if _rag_engine and _rag_engine.is_available:
            result = await _rag_engine.query(query, mode="mix")
            response = result.get("response", "")
            if response and not result.get("error"):
                logger.info("raganything.kg_query_success", response_len=len(response))
                return AgentResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    status=ResultStatus.SUCCESS,
                    output={
                        "response": response,
                        "retrieved": [{"text": response, "source": "lightrag_kg", "score": 1.0}],
                    },
                )

        # Fallback: use text_aggregate from ingest step
        text = pipeline_ctx.text_aggregate
        if text:
            logger.info("raganything.fallback_text_retrieval", text_len=len(text))
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.SUCCESS,
                output={
                    "response": text[:15000],
                    "retrieved": [{"text": text[:15000], "source": "fallback_text", "score": 0.8}],
                },
            )

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.PARTIAL,
            output={"response": "", "retrieved": [], "message": "No content available"},
        )

    async def _legacy_ingest(self, task: Task, file_path: str) -> AgentResult:
        """Fallback ingestion using PyMuPDF when RAG-Anything is not installed."""
        from pathlib import Path
        import hashlib

        path = Path(file_path)
        if not path.exists():
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[f"File not found: {file_path}"],
            )

        items: list[dict[str, Any]] = []
        text_parts: list[str] = []

        if path.suffix.lower() == ".pdf":
            import fitz
            with fitz.open(str(path)) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text.strip():
                        items.append({"type": "text", "content": text, "page_idx": page_num + 1, "source": str(path)})
                        text_parts.append(f"[Page {page_num + 1}]\n{text}")
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                items.append({"type": "text", "content": text, "source": str(path)})
                text_parts.append(text)
            except Exception:
                pass

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "indexed": False,
                "content_items": items[:100],
                "text_aggregate": "\n\n".join(text_parts)[:50000],
                "document": {
                    "file_path": str(path),
                    "file_type": path.suffix.lower(),
                    "content_hash": content_hash,
                    "file_size_bytes": path.stat().st_size,
                },
            },
        )
