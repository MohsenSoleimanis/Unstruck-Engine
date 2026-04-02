"""RAG-Anything engine — proper integration based on actual source code reading.

RAGAnything expects:
  - Text items: {"type": "text", "text": "...", "page_idx": N}
  - Table items: {"type": "table", "table_body": "...", "table_caption": [...], "page_idx": N}
  - Image items: {"type": "image", "img_path": "/abs/path.jpg", "image_caption": [...], "page_idx": N}

separate_content() reads item["text"] NOT item["content"].
insert_content_list() DOES insert text into LightRAG via ainsert().
Modal processors need llm_model_func + optional vision_model_func.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import structlog

# Ensure MinerU scripts are on PATH (Windows user scripts dir)
_user_scripts = Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts"
if _user_scripts.exists() and str(_user_scripts) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_user_scripts) + os.pathsep + os.environ.get("PATH", "")

logger = structlog.get_logger()

_LIGHTRAG_AVAILABLE = False
_RAG_AVAILABLE = False

try:
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
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
    Manages RAG-Anything lifecycle.

    Uses RAG-Anything's actual API — not bypassed, not mimicked:
      - process_document_complete() when MinerU is installed
      - insert_content_list() with correctly formatted content when MinerU is not
      - aquery() for retrieval (with optional VLM enhancement)
    """

    def __init__(
        self,
        working_dir: str = "./data/raganything",
        llm_model: str = "gpt-4o-mini",
        vision_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._rag: Any = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize RAG-Anything with LightRAG, LLM, embedding, and vision functions."""
        if self._initialized:
            return True

        if not _RAG_AVAILABLE:
            logger.warning("rag_engine.raganything_not_installed")
            if not _LIGHTRAG_AVAILABLE:
                logger.warning("rag_engine.lightrag_not_installed")
                return False

        try:
            # Create the LLM wrapper: binds model name so LightRAG can call
            # llm_func(prompt, system_prompt=...) without passing model
            model = self.llm_model

            async def llm_func(prompt: str, system_prompt: str | None = None,
                               history_messages: list | None = None, **kwargs: Any) -> str:
                return await openai_complete_if_cache(
                    model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages or [], **kwargs,
                )

            # Vision model function for image processing
            vision_model = self.vision_model

            async def vision_func(prompt: str, system_prompt: str | None = None,
                                  image_data: str | None = None, **kwargs: Any) -> str:
                return await openai_complete_if_cache(
                    vision_model, prompt, system_prompt=system_prompt, **kwargs,
                )

            embed_func = EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=8192,
                func=lambda texts: openai_embed(texts, model=self.embedding_model),
            )

            if _RAG_AVAILABLE:
                self._rag = RAGAnything(
                    llm_model_func=llm_func,
                    vision_model_func=vision_func,
                    embedding_func=embed_func,
                    config=RAGAnythingConfig(
                        working_dir=str(self.working_dir),
                        enable_image_processing=True,
                        enable_table_processing=True,
                        enable_equation_processing=True,
                    ),
                    lightrag_kwargs={
                        "llm_model_name": self.llm_model,
                    },
                )
            else:
                # Fallback: LightRAG only (no multimodal)
                lightrag = LightRAG(
                    working_dir=str(self.working_dir),
                    llm_model_func=llm_func,
                    llm_model_name=self.llm_model,
                    embedding_func=embed_func,
                )
                await lightrag.initialize_storages()
                self._rag = lightrag

            self._initialized = True
            logger.info("rag_engine.initialized",
                        working_dir=str(self.working_dir),
                        backend="raganything" if _RAG_AVAILABLE else "lightrag",
                        parser_available=self._check_parser())
            return True

        except Exception as e:
            logger.error("rag_engine.init_failed", error=str(e))
            return False

    def _check_parser(self) -> bool:
        """Check if the configured parser (MinerU) is installed."""
        if _RAG_AVAILABLE and hasattr(self._rag, "check_parser_installation"):
            return self._rag.check_parser_installation()
        return False

    async def ingest_document(self, file_path: str, doc_id: str | None = None) -> dict[str, Any]:
        """
        Ingest a document into RAG-Anything.

        If MinerU is installed: uses process_document_complete() — full pipeline
        with layout-aware parsing, multimodal processing, KG construction.

        If MinerU is NOT installed: parses with PyMuPDF, builds correctly
        formatted content list, uses insert_content_list() which:
          1. Separates text vs multimodal items
          2. Inserts text into LightRAG (KG + vector index)
          3. Processes tables/images via modal processors
        """
        if not await self._ensure_ready():
            return {"error": "RAG engine not initialized", "fallback": True}

        try:
            if _RAG_AVAILABLE and isinstance(self._rag, RAGAnything):
                if self._check_parser():
                    try:
                        # Full RAG-Anything pipeline with MinerU parser
                        logger.info("rag_engine.full_pipeline", file_path=file_path)
                        await self._rag.process_document_complete(
                            file_path=file_path,
                            doc_id=doc_id or Path(file_path).stem,
                        )
                    except Exception as parse_err:
                        # MinerU failed (timeout, OOM, etc) — fallback to content list
                        logger.warning("rag_engine.mineru_failed_fallback", error=str(parse_err))
                        content_list = self._parse_to_rag_format(file_path)
                        await self._rag.insert_content_list(
                            content_list,
                            file_path=file_path,
                            doc_id=doc_id or Path(file_path).stem,
                        )
                else:
                    # No MinerU — parse with PyMuPDF, use insert_content_list
                    # This still goes through RAG-Anything's full pipeline:
                    # separate text/multimodal → LightRAG KG for text →
                    # modal processors for tables/images → merge into KG
                    logger.info("rag_engine.content_list_pipeline", file_path=file_path)
                    content_list = self._parse_to_rag_format(file_path)
                    await self._rag.insert_content_list(
                        content_list,
                        file_path=file_path,
                        doc_id=doc_id or Path(file_path).stem,
                    )
            else:
                # LightRAG only — text insert
                logger.info("rag_engine.lightrag_only", file_path=file_path)
                content_list = self._parse_to_rag_format(file_path)
                texts = [item["text"] for item in content_list if item.get("type") == "text" and item.get("text")]
                await self._rag.ainsert("\n\n".join(texts))

            return {
                "doc_id": doc_id or Path(file_path).stem,
                "indexed": True,
            }

        except Exception as e:
            logger.error("rag_engine.ingest_failed", error=str(e), file_path=file_path)
            return {"error": str(e), "fallback": True}

    async def query(self, prompt: str, mode: str = "mix") -> dict[str, Any]:
        """Query using RAG-Anything (KG + vector hybrid retrieval)."""
        if not await self._ensure_ready():
            return {"error": "RAG engine not initialized", "response": ""}

        try:
            if _RAG_AVAILABLE and isinstance(self._rag, RAGAnything):
                response = await self._rag.aquery(prompt, mode=mode)
            else:
                response = await self._rag.aquery(prompt)

            return {
                "response": response if isinstance(response, str) else str(response),
                "mode": mode,
            }
        except Exception as e:
            logger.error("rag_engine.query_failed", error=str(e))
            return {"error": str(e), "response": ""}

    async def _ensure_ready(self) -> bool:
        if not self._initialized:
            return await self.initialize()
        return self._initialized

    def _parse_to_rag_format(self, file_path: str) -> list[dict[str, Any]]:
        """
        Parse document with PyMuPDF into RAG-Anything's expected format.

        CRITICAL: RAG-Anything's separate_content() reads item["text"] NOT item["content"].
        Tables need "table_body", "table_caption", "table_footnote" keys.
        Images need "img_path" (absolute), "image_caption", "image_footnote" keys.
        """
        import fitz

        path = Path(file_path)
        if not path.exists():
            return []

        content_list: list[dict[str, Any]] = []

        if path.suffix.lower() == ".pdf":
            with fitz.open(str(path)) as doc:
                for pn in range(len(doc)):
                    page = doc[pn]

                    # Text — key MUST be "text" not "content"
                    text = page.get_text("text")
                    if text.strip():
                        content_list.append({
                            "type": "text",
                            "text": text,
                            "page_idx": pn + 1,
                        })

                    # Tables — needs "table_body", "table_caption", "table_footnote"
                    tables = page.find_tables()
                    if tables and tables.tables:
                        for t_idx, table in enumerate(tables.tables):
                            try:
                                data = table.extract()
                                # Format as markdown table
                                rows = []
                                for row in data:
                                    rows.append("| " + " | ".join(str(c or "") for c in row) + " |")
                                if len(rows) > 1:
                                    # Add header separator
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

                    # Images — needs "img_path" (absolute), "image_caption"
                    for img_idx, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        try:
                            pix = fitz.Pixmap(doc, xref)
                            img_dir = Path(self.working_dir) / "images"
                            img_dir.mkdir(parents=True, exist_ok=True)
                            img_path = img_dir / f"page{pn + 1}_img{img_idx}.png"
                            pix.save(str(img_path))
                            content_list.append({
                                "type": "image",
                                "img_path": str(img_path.resolve()),
                                "image_caption": [],
                                "image_footnote": [],
                                "page_idx": pn + 1,
                            })
                        except Exception:
                            pass
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                content_list.append({"type": "text", "text": text})
            except Exception:
                pass

        return content_list

    @property
    def is_available(self) -> bool:
        return self._initialized and self._rag is not None
