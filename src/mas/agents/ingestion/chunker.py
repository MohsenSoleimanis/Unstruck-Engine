"""Chunker agent — splits text into retrieval-optimized chunks.

Handles semantic chunking with overlap, respecting section boundaries.
"""

from __future__ import annotations

import hashlib
from typing import Any

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task


@registry.register
class ChunkerAgent(BaseAgent):
    """
    Splits text into overlapping chunks for vector storage and retrieval.

    Supports multiple strategies:
      - fixed: fixed-size chunks with overlap
      - semantic: split on paragraph/section boundaries
    """

    agent_type = "chunker"
    description = "Splits text into retrieval-optimized chunks with configurable strategy"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        text = task.context.get("text", "")
        strategy = task.context.get("strategy", "semantic")
        chunk_size = task.context.get("chunk_size", 1000)
        overlap = task.context.get("overlap", 200)
        source = task.context.get("source", "unknown")

        if not text:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"chunks": [], "message": "No text provided"},
            )

        if strategy == "semantic":
            chunks = self._semantic_chunk(text, chunk_size, overlap)
        else:
            chunks = self._fixed_chunk(text, chunk_size, overlap)

        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = hashlib.sha256(f"{source}_{i}_{chunk['text'][:50]}".encode()).hexdigest()[:12]
            chunk["chunk_index"] = i
            chunk["source"] = source

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "chunks": chunks,
                "total_chunks": len(chunks),
                "strategy": strategy,
                "avg_chunk_size": sum(len(c["text"]) for c in chunks) // max(len(chunks), 1),
            },
        )

    def _semantic_chunk(self, text: str, max_size: int, overlap: int) -> list[dict[str, Any]]:
        """Split on paragraph boundaries, merging small paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > max_size and current:
                chunks.append({"text": current.strip()})
                # Keep overlap from end of current chunk
                current = current[-overlap:] + "\n\n" + para if overlap else para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append({"text": current.strip()})

        return chunks

    def _fixed_chunk(self, text: str, size: int, overlap: int) -> list[dict[str, Any]]:
        """Fixed-size chunks with overlap."""
        if overlap >= size:
            overlap = size // 2  # Prevent infinite loop
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append({"text": text[start:end]})
            start = end - overlap
        return chunks
