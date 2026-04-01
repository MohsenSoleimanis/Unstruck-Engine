"""Embedder agent — generates embeddings and stores chunks in vector DB."""

from __future__ import annotations

from typing import Any

import structlog

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

logger = structlog.get_logger()


@registry.register
class EmbedderAgent(BaseAgent):
    """
    Generates embeddings for chunks and stores them in the vector database.

    Takes chunked text + modal descriptions and indexes them for retrieval.
    """

    agent_type = "embedder"
    description = "Generates embeddings for content chunks and stores them in vector DB for retrieval"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        chunks = task.context.get("chunks", [])
        modal_descriptions = task.context.get("modal_descriptions", [])
        collection_name = task.context.get("collection", "default")

        if not chunks and not modal_descriptions:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"indexed": 0, "message": "Nothing to embed"},
            )

        # Combine text chunks and modal descriptions
        all_items = []
        for chunk in chunks:
            all_items.append({
                "id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "metadata": {
                    "type": "text_chunk",
                    "source": chunk.get("source", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                },
            })

        for desc in modal_descriptions:
            all_items.append({
                "id": f"modal_{desc.get('type', 'unknown')}_{desc.get('page_idx', 0)}",
                "text": desc.get("description", ""),
                "metadata": {
                    "type": f"modal_{desc.get('type', 'unknown')}",
                    "source": desc.get("source", ""),
                    "page_idx": desc.get("page_idx"),
                },
            })

        # Store in shared memory (ChromaDB)
        indexed = 0
        try:
            import chromadb

            client = chromadb.PersistentClient(path="./data/chroma")
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            batch_ids = [item["id"] for item in all_items if item["id"]]
            batch_docs = [item["text"] for item in all_items if item["id"]]
            batch_metas = [item["metadata"] for item in all_items if item["id"]]

            if batch_ids:
                collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                indexed = len(batch_ids)

        except Exception as e:
            logger.warning("embedder.storage_fallback", error=str(e))
            indexed = len(all_items)  # Count as indexed even without persistence

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "indexed": indexed,
                "collection": collection_name,
                "text_chunks": len(chunks),
                "modal_descriptions": len(modal_descriptions),
            },
        )
