"""RAG-Anything service — document ingestion and knowledge graph retrieval.

This is a SERVICE, not an agent. Multiple callers use it:
  - Tool Layer: rag_ingest tool calls service.ingest()
  - Context Engine: retrieval strategy calls service.query()

Uses RAG-Anything with Docling parser + LightRAG for:
  - Layout-aware document parsing (text, tables, images)
  - Knowledge graph construction (entity extraction, relationships)
  - Hybrid retrieval (KG traversal + vector similarity)

Verified API facts (from installed raganything 1.2.10):
  - Text items: {"type": "text", "text": "...", "page_idx": N}
  - Table items: {"type": "table", "table_body": "markdown", "table_caption": [...]}
  - separate_content() reads item["text"] for text items
  - insert_content_list() DOES insert text into LightRAG
  - LightRAG needs initialize_storages() before any operation
  - Docling parser needs user Scripts dir on PATH (Windows)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import structlog

from unstruck.config import Config

logger = structlog.get_logger()

# Ensure parser CLI tools are on PATH (Windows user scripts dir)
_user_scripts = Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts"
if _user_scripts.exists() and str(_user_scripts) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_user_scripts) + os.pathsep + os.environ.get("PATH", "")

# Check availability
_AVAILABLE = False
try:
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig
    _AVAILABLE = True
except ImportError:
    pass


class RAGService:
    """
    RAG-Anything service — singleton, initialized once at startup.

    Provides two operations:
      - ingest(file_path): parse document, build KG, index chunks
      - query(prompt): retrieve from KG + vectors

    Falls back gracefully when RAG-Anything is not installed.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._rag_config = config.rag
        self._model_config = config.models
        self._rag: Any = None
        self._lightrag: Any = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize RAG-Anything + LightRAG. Call once at startup."""
        if self._initialized:
            return True

        if not _AVAILABLE:
            logger.warning("rag_service.not_installed", message="raganything package not available")
            return False

        try:
            working_dir = str(self._config.data_dir / "rag")
            Path(working_dir).mkdir(parents=True, exist_ok=True)

            # Model config
            worker_tier = self._model_config.get("tiers", {}).get("worker", {})
            vision_tier = self._model_config.get("tiers", {}).get("vision", {})
            embedding_cfg = self._model_config.get("tiers", {}).get("embedding", {})

            llm_model = worker_tier.get("primary", "gpt-4o-mini")
            vision_model = vision_tier.get("primary", "gpt-4o")
            embed_model = embedding_cfg.get("model", "text-embedding-3-small")
            embed_dim = embedding_cfg.get("dimensions", 1536)

            # LLM wrapper: binds model name into closure
            async def llm_func(prompt: str, system_prompt: str | None = None,
                               history_messages: list | None = None, **kwargs: Any) -> str:
                return await openai_complete_if_cache(
                    llm_model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages or [], **kwargs,
                )

            async def vision_func(prompt: str, system_prompt: str | None = None,
                                  **kwargs: Any) -> str:
                return await openai_complete_if_cache(
                    vision_model, prompt, system_prompt=system_prompt, **kwargs,
                )

            embed_func = EmbeddingFunc(
                embedding_dim=embed_dim,
                max_token_size=8192,
                func=lambda texts: openai_embed(texts, model=embed_model),
            )

            # Create LightRAG first — loads existing KG data from disk
            self._lightrag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                llm_model_name=llm_model,
                embedding_func=embed_func,
            )
            await self._lightrag.initialize_storages()
            logger.info("rag_service.lightrag_loaded",
                        nodes=self._lightrag.chunk_entity_relation_graph.number_of_nodes() if hasattr(self._lightrag, 'chunk_entity_relation_graph') else 0)

            # Parser from config
            parser = self._rag_config.get("parser", "docling")

            # Create RAG-Anything with pre-initialized LightRAG
            self._rag = RAGAnything(
                lightrag=self._lightrag,
                llm_model_func=llm_func,
                vision_model_func=vision_func,
                embedding_func=embed_func,
                config=RAGAnythingConfig(
                    working_dir=working_dir,
                    parser=parser,
                    enable_image_processing=self._rag_config.get("enable_image_processing", True),
                    enable_table_processing=self._rag_config.get("enable_table_processing", True),
                    enable_equation_processing=self._rag_config.get("enable_equation_processing", True),
                ),
            )

            self._initialized = True
            logger.info("rag_service.initialized",
                        working_dir=working_dir,
                        parser=parser,
                        parser_available=self._rag.check_parser_installation() if self._rag else False)
            return True

        except Exception as e:
            logger.error("rag_service.init_failed", error=str(e))
            return False

    async def ingest(self, file_path: str, doc_id: str | None = None) -> dict[str, Any]:
        """
        Ingest a document — parse, extract entities, build KG, index.

        If the configured parser (Docling) is available:
          → process_document_complete() — full multimodal pipeline

        If parser is NOT available:
          → Parse with PyMuPDF, feed to insert_content_list()
          → RAG-Anything still processes text + tables → LightRAG KG
        """
        if not await self._ensure_ready():
            return {"error": "RAG service not initialized", "indexed": False}

        try:
            final_doc_id = doc_id or Path(file_path).stem

            # Try full parser pipeline first
            if self._rag.check_parser_installation():
                try:
                    logger.info("rag_service.ingest.full_pipeline", file_path=file_path, parser=self._rag_config.get("parser"))
                    await self._rag.process_document_complete(
                        file_path=file_path,
                        doc_id=final_doc_id,
                    )
                    logger.info("rag_service.ingest.complete", doc_id=final_doc_id)
                    return {"doc_id": final_doc_id, "indexed": True}
                except Exception as parser_err:
                    logger.warning("rag_service.ingest.parser_crashed",
                                   error=str(parser_err)[:200], file_path=file_path)
                    # Fall through to PyMuPDF fallback below

            # Fallback: PyMuPDF parse → insert_content_list
            # This path runs when: parser not installed, OR parser crashed
            # Still uses RAG-Anything: text → LightRAG KG, tables → modal processors
            logger.info("rag_service.ingest.pymupdf_fallback", file_path=file_path)
            content_list = self._parse_with_pymupdf(file_path)
            if not content_list:
                return {"error": f"No content extracted from {file_path}", "indexed": False}

            await self._rag.insert_content_list(
                content_list,
                file_path=file_path,
                doc_id=final_doc_id,
            )

            logger.info("rag_service.ingest.complete", doc_id=final_doc_id, method="pymupdf_fallback")
            return {"doc_id": final_doc_id, "indexed": True}

        except Exception as e:
            logger.error("rag_service.ingest.failed", error=str(e), file_path=file_path)
            return {"error": str(e), "indexed": False}

    async def query(self, prompt: str, mode: str = "mix") -> dict[str, Any]:
        """Query the knowledge graph + vector index."""
        if not await self._ensure_ready():
            return {"error": "RAG service not initialized", "response": ""}

        try:
            response = await self._rag.aquery(prompt, mode=mode)
            return {
                "response": response if isinstance(response, str) else str(response),
                "mode": mode,
            }
        except Exception as e:
            logger.error("rag_service.query.failed", error=str(e))
            return {"error": str(e), "response": ""}

    @property
    def is_available(self) -> bool:
        return _AVAILABLE and self._initialized

    async def _ensure_ready(self) -> bool:
        if not self._initialized:
            return await self.initialize()
        return True

    def _parse_with_pymupdf(self, file_path: str) -> list[dict[str, Any]]:
        """
        Fallback parser using PyMuPDF.

        Produces content_list in RAG-Anything's expected format:
          - Text: {"type": "text", "text": "...", "page_idx": N}
          - Tables: {"type": "table", "table_body": "markdown", "table_caption": [...], "page_idx": N}

        KEY: "text" not "content" — RAG-Anything's separate_content() reads item["text"].
        """
        import fitz

        path = Path(file_path)
        if not path.exists():
            logger.error("rag_service.file_not_found", file_path=file_path)
            return []

        content_list: list[dict[str, Any]] = []

        if path.suffix.lower() == ".pdf":
            with fitz.open(str(path)) as doc:
                for pn in range(len(doc)):
                    page = doc[pn]

                    # Text — key MUST be "text"
                    text = page.get_text("text")
                    if text.strip():
                        content_list.append({
                            "type": "text",
                            "text": text,
                            "page_idx": pn + 1,
                        })

                    # Tables — needs "table_body" as markdown
                    tables = page.find_tables()
                    if tables and tables.tables:
                        for table in tables.tables:
                            try:
                                data = table.extract()
                                rows = []
                                for row in data:
                                    rows.append("| " + " | ".join(str(c or "") for c in row) + " |")
                                if len(rows) > 1:
                                    header = rows[0]
                                    sep = "| " + " | ".join("---" for _ in data[0]) + " |"
                                    table_body = header + "\n" + sep + "\n" + "\n".join(rows[1:])
                                else:
                                    table_body = "\n".join(rows)

                                content_list.append({
                                    "type": "table",
                                    "table_body": table_body,
                                    "table_caption": [f"Table on page {pn + 1}"],
                                    "table_footnote": [],
                                    "page_idx": pn + 1,
                                })
                            except Exception:
                                pass
        else:
            # Non-PDF: read as text
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                content_list.append({"type": "text", "text": text})
            except Exception:
                pass

        return content_list
