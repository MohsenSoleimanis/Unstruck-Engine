"""Shared memory — cross-agent vector store for knowledge persistence."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class SharedMemory:
    """
    Shared memory layer (Layer 3 — Memory Layer from the 3-layer hierarchy).

    Wraps a ChromaDB collection for cross-agent knowledge sharing.
    All agents read/write to the same collection, enabling knowledge transfer.

    Supports:
      - Semantic search over agent outputs
      - Metadata filtering by agent_type, task_id, content_type
      - Task board: shared state visible to all agents
    """

    def __init__(self, collection_name: str = "mas_shared", persist_dir: str = "./data/chroma"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._task_board: dict[str, Any] = {}

    def _ensure_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def store(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Store a document in shared memory."""
        self._ensure_client()
        kwargs: dict[str, Any] = {
            "ids": [doc_id],
            "documents": [text],
        }
        if metadata:
            kwargs["metadatas"] = [metadata]
        if embedding:
            kwargs["embeddings"] = [embedding]
        self._collection.upsert(**kwargs)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across shared memory."""
        self._ensure_client()
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            }
            for i in range(len(results["ids"][0]))
        ]

    def store_result(self, task_id: str, agent_type: str, content: str, extra: dict | None = None) -> None:
        """Store an agent result for cross-agent access."""
        metadata = {"agent_type": agent_type, "task_id": task_id, "type": "agent_result"}
        if extra:
            metadata.update(extra)
        self.store(doc_id=f"result_{task_id}", text=content, metadata=metadata)

    # --- Task Board (shared state visible to all agents) ---

    def post_to_board(self, key: str, value: Any) -> None:
        self._task_board[key] = value

    def read_board(self, key: str, default: Any = None) -> Any:
        return self._task_board.get(key, default)

    def get_board(self) -> dict[str, Any]:
        return dict(self._task_board)
