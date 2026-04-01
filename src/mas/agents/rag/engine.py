"""RAG-Anything engine — lifecycle management for the RAG engine singleton.

Initializes RAG-Anything once at startup, provides it to agents.
Handles the LLM adapter (LangChain → RAG-Anything callable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# Sentinel for when RAG-Anything is not installed
_RAG_AVAILABLE = False
try:
    from raganything import RAGAnything, RAGAnythingConfig
    _RAG_AVAILABLE = True
except ImportError:
    RAGAnything = None  # type: ignore
    RAGAnythingConfig = None  # type: ignore


class RAGEngine:
    """
    Manages RAG-Anything lifecycle.

    - Initializes once with working_dir, LLM config, embedding config
    - Provides process_document + query methods
    - Adapts LangChain LLMs to RAG-Anything's callable interface
    """

    def __init__(
        self,
        working_dir: str = "./data/raganything",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._rag: Any = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize RAG-Anything. Returns True if successful."""
        if self._initialized:
            return True

        if not _RAG_AVAILABLE:
            logger.warning("rag_engine.not_installed", message="raganything package not available — using fallback agents")
            return False

        try:
            llm_func = self._create_llm_func()
            embedding_func = self._create_embedding_func()

            config = RAGAnythingConfig(
                working_dir=str(self.working_dir),
                llm_model_func=llm_func,
                embedding_func=embedding_func,
                embedding_dim=self.embedding_dim,
            )

            self._rag = RAGAnything(config=config)
            self._initialized = True
            logger.info("rag_engine.initialized", working_dir=str(self.working_dir))
            return True

        except Exception as e:
            logger.error("rag_engine.init_failed", error=str(e))
            return False

    def _create_llm_func(self):
        """Create an async callable that RAG-Anything expects for LLM calls."""
        import openai
        client = openai.AsyncOpenAI()
        model = self.llm_model

        async def llm_func(prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = await client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content or ""

        return llm_func

    def _create_embedding_func(self):
        """Create an async callable for embeddings."""
        import openai
        client = openai.AsyncOpenAI()
        model = self.embedding_model

        async def embedding_func(texts: list[str]) -> list[list[float]]:
            response = await client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]

        return embedding_func

    async def ingest_document(self, file_path: str, doc_id: str | None = None) -> dict[str, Any]:
        """Process and index a document via RAG-Anything."""
        if not self._initialized or not self._rag:
            return {"error": "RAG engine not initialized", "fallback": True}

        try:
            result = await self._rag.process_document_complete(
                file_path=file_path,
                doc_id=doc_id,
            )
            return {
                "doc_id": result.get("doc_id", doc_id or file_path),
                "indexed": True,
                "content_items": result.get("content_items", []),
                "chunks": result.get("chunks", []),
                "entities": result.get("entities", []),
            }
        except Exception as e:
            logger.error("rag_engine.ingest_failed", error=str(e), file_path=file_path)
            return {"error": str(e), "fallback": True}

    async def query(self, prompt: str, mode: str = "hybrid") -> dict[str, Any]:
        """Query the RAG engine."""
        if not self._initialized or not self._rag:
            return {"error": "RAG engine not initialized", "response": ""}

        try:
            response = await self._rag.aquery(prompt, mode=mode)
            return {
                "response": response if isinstance(response, str) else str(response),
                "mode": mode,
            }
        except Exception as e:
            logger.error("rag_engine.query_failed", error=str(e))
            return {"error": str(e), "response": ""}

    @property
    def is_available(self) -> bool:
        return self._initialized and self._rag is not None
