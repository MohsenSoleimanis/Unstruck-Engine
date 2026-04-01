"""PipelineContext — typed shared blackboard for cross-agent data flow.

Replaces the untyped dict[str, Any] context passing. Every agent reads
from and writes to specific sections of this structure. No more guessing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    """Metadata about an ingested document."""

    file_path: str = ""
    file_type: str = ""
    content_hash: str = ""
    file_size_bytes: int = 0
    doc_id: str = ""  # RAG-Anything doc_id after indexing


class ContentItem(BaseModel):
    """A single parsed content element."""

    type: str  # "text" | "table" | "image" | "equation" | "structured"
    content: Any = None  # str for text, list[list] for tables, None for binary
    page_idx: int | None = None
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkItem(BaseModel):
    """A retrieval-ready chunk."""

    chunk_id: str = ""
    text: str = ""
    chunk_index: int = 0
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityItem(BaseModel):
    """An extracted entity for the knowledge graph."""

    name: str
    type: str = "Other"
    properties: dict[str, Any] = Field(default_factory=dict)
    source_modality: str = "unknown"
    page_idx: int | None = None


class RelationshipItem(BaseModel):
    """A relationship between entities."""

    source: str
    target: str
    relation: str = "RELATED_TO"
    properties: dict[str, Any] = Field(default_factory=dict)


class RetrievedItem(BaseModel):
    """A retrieval result."""

    id: str = ""
    text: str = ""
    score: float = 0.0
    source: str = ""  # "vector" | "graph" | "raganything"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Output from reasoning/analysis."""

    answer: str = ""
    key_points: list[str] = Field(default_factory=list)
    citations: list[dict[str, str]] = Field(default_factory=list)
    confidence: str = "medium"
    limitations: list[str] = Field(default_factory=list)


class PipelineContext(BaseModel):
    """
    Typed shared blackboard for the multi-agent pipeline.

    Each section is "owned" by a stage:
      - document + content_items: written by ingestion / RAG-Anything
      - streams + text_aggregate: written by separator
      - chunks: written by chunker / RAG-Anything
      - entities + relationships: written by processors / RAG-Anything
      - graph_nodes + graph_edges: written by KG builder
      - retrieved: written by retriever / RAG-Anything
      - analysis + synthesis: written by analyst / synthesizer

    Agents read from earlier sections, write to their own.
    """

    # --- Input ---
    query: str = ""
    document: DocumentMeta | None = None

    # --- Ingestion output ---
    content_items: list[ContentItem] = Field(default_factory=list)

    # --- Separation output ---
    streams: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    text_aggregate: str = ""

    # --- Chunking output ---
    chunks: list[ChunkItem] = Field(default_factory=list)

    # --- Entity extraction output ---
    entities: list[EntityItem] = Field(default_factory=list)
    relationships: list[RelationshipItem] = Field(default_factory=list)

    # --- Graph output ---
    graph_nodes: list[dict[str, Any]] = Field(default_factory=list)
    graph_edges: list[dict[str, Any]] = Field(default_factory=list)

    # --- Retrieval output ---
    retrieved: list[RetrievedItem] = Field(default_factory=list)

    # --- Analysis/reasoning output ---
    analysis: AnalysisResult | None = None
    synthesis: dict[str, Any] = Field(default_factory=dict)

    # --- RAG-Anything state ---
    rag_doc_id: str = ""
    rag_indexed: bool = False
    rag_response: str = ""  # raw RAG-Anything query response

    # --- Escape hatch ---
    extra: dict[str, Any] = Field(default_factory=dict)
