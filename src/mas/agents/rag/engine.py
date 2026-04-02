"""RAG-Anything engine — lifecycle management for the RAG engine singleton.

Properly initializes LightRAG + RAG-Anything with:
  - Pre-initialized LightRAG instance (required when MinerU parser not installed)
  - Correct LLM function wrapper (model name bound into closure)
  - Proper embedding function with numpy return
  - Storage initialization before any insert/query
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_RAG_AVAILABLE = False
_LIGHTRAG_AVAILABLE = False

try:
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    _LIGHTRAG_AVAILABLE = True
except ImportError:
    LightRAG = None  # type: ignore
    EmbeddingFunc = None  # type: ignore

try:
    from raganything import RAGAnything, RAGAnythingConfig
    _RAG_AVAILABLE = True
except ImportError:
    RAGAnything = None  # type: ignore
    RAGAnythingConfig = None  # type: ignore


class RAGEngine:
    """
    Manages LightRAG + RAG-Anything lifecycle.

    Key insight from testing: RAG-Anything needs a pre-initialized LightRAG
    when MinerU parser is not installed. LightRAG needs initialize_storages()
    called before any insert/query. The LLM function must accept
    (prompt, system_prompt=..., **kwargs) — NOT (model, prompt, ...).
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
        self._lightrag: Any = None
        self._rag: Any = None
        self._initialized = False
        self._storages_ready = False

    async def initialize(self) -> bool:
        """Initialize LightRAG + RAG-Anything. Returns True if successful."""
        if self._initialized:
            return True

        if not _LIGHTRAG_AVAILABLE:
            logger.warning("rag_engine.lightrag_not_installed")
            return False

        try:
            # Create LLM and embedding functions
            llm_func = self._create_llm_func()
            embed_func = self._create_embedding_func()

            # Create LightRAG first (required by RAG-Anything)
            self._lightrag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_func,
                llm_model_name=self.llm_model,
                embedding_func=embed_func,
            )

            # Initialize storages (REQUIRED before insert/query)
            await self._lightrag.initialize_storages()
            self._storages_ready = True

            # Create RAG-Anything with pre-initialized LightRAG
            if _RAG_AVAILABLE:
                self._rag = RAGAnything(
                    lightrag=self._lightrag,
                    llm_model_func=llm_func,
                    embedding_func=embed_func,
                    config=RAGAnythingConfig(
                        working_dir=str(self.working_dir),
                        enable_image_processing=False,
                        enable_table_processing=False,
                        enable_equation_processing=False,
                    ),
                )

            self._initialized = True
            logger.info("rag_engine.initialized",
                        working_dir=str(self.working_dir),
                        has_raganything=_RAG_AVAILABLE,
                        lightrag=True)
            return True

        except Exception as e:
            logger.error("rag_engine.init_failed", error=str(e))
            return False

    def _create_llm_func(self):
        """
        Create an async LLM callable matching LightRAG's expected signature.

        LightRAG calls: llm_func(prompt, system_prompt=None, history_messages=None, **kwargs)
        NOT: llm_func(model, prompt, ...) — the model is bound in the closure.
        """
        from lightrag.llm.openai import openai_complete_if_cache
        model = self.llm_model

        async def llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list | None = None,
            **kwargs: Any,
        ) -> str:
            # openai_complete_if_cache expects (model, prompt, system_prompt, ...)
            return await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                **kwargs,
            )

        return llm_func

    def _create_embedding_func(self):
        """Create embedding function matching LightRAG's EmbeddingFunc spec."""
        from lightrag.llm.openai import openai_embed
        model = self.embedding_model

        return EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(texts, model=model),
        )

    async def ingest_document(self, file_path: str, doc_id: str | None = None) -> dict[str, Any]:
        """Ingest a document — parse with PyMuPDF, index via LightRAG."""
        if not await self._ensure_ready():
            return {"error": "RAG engine not initialized", "fallback": True}

        try:
            # Parse with PyMuPDF (works without MinerU)
            content_list = self._parse_document(file_path)

            if not content_list:
                return {"error": f"No content extracted from {file_path}", "fallback": True}

            # Insert directly into LightRAG (builds KG + vector index)
            # Using ainsert() instead of RAG-Anything's insert_content_list()
            # because insert_content_list doesn't forward text to LightRAG's
            # entity extraction pipeline when parser isn't installed.
            texts = [item["content"] for item in content_list if item.get("content")]
            full_text = "\n\n".join(texts)
            await self._lightrag.ainsert(full_text)

            logger.info("rag_engine.ingested",
                        file_path=file_path,
                        items=len(content_list),
                        via="raganything" if self._rag else "lightrag")

            return {
                "doc_id": doc_id or Path(file_path).stem,
                "indexed": True,
                "content_items": content_list[:100],  # cap for serialization
            }

        except Exception as e:
            logger.error("rag_engine.ingest_failed", error=str(e), file_path=file_path)
            return {"error": str(e), "fallback": True}

    async def query(self, prompt: str, mode: str = "mix") -> dict[str, Any]:
        """Query the indexed documents."""
        if not await self._ensure_ready():
            return {"error": "RAG engine not initialized", "response": ""}

        try:
            # Use LightRAG directly — more reliable than RAG-Anything wrapper
            response = await self._lightrag.aquery(prompt)

            return {
                "response": response if isinstance(response, str) else str(response),
                "mode": mode,
            }

        except Exception as e:
            logger.error("rag_engine.query_failed", error=str(e))
            return {"error": str(e), "response": ""}

    async def _ensure_ready(self) -> bool:
        """Lazy initialization — initialize on first use."""
        if not self._initialized:
            return await self.initialize()
        return self._initialized

    def _parse_document(self, file_path: str) -> list[dict[str, Any]]:
        """Parse document with PyMuPDF. Returns content list for RAG-Anything."""
        import fitz

        path = Path(file_path)
        if not path.exists():
            logger.error("rag_engine.file_not_found", file_path=file_path)
            return []

        content_list: list[dict[str, Any]] = []

        if path.suffix.lower() == ".pdf":
            with fitz.open(str(path)) as doc:
                for pn in range(len(doc)):
                    page = doc[pn]

                    text = page.get_text("text")
                    if text.strip():
                        content_list.append({
                            "type": "text",
                            "content": text,
                            "page_idx": pn + 1,
                        })

                    # Extract tables and add as text (LightRAG processes all text)
                    tables = page.find_tables()
                    if tables and tables.tables:
                        for t in tables.tables:
                            try:
                                data = t.extract()
                                table_str = "\n".join(
                                    " | ".join(str(c or "") for c in row) for row in data
                                )
                                content_list.append({
                                    "type": "text",
                                    "content": f"[TABLE on page {pn + 1}]\n{table_str}",
                                    "page_idx": pn + 1,
                                })
                            except Exception:
                                pass
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                content_list.append({"type": "text", "content": text})
            except Exception:
                pass

        return content_list

    @property
    def is_available(self) -> bool:
        return self._initialized and (self._lightrag is not None or self._rag is not None)
