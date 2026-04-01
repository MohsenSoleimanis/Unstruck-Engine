"""Context merger — merges agent outputs into PipelineContext by agent type.

Each agent type has a known output shape. The merger knows where to write
each agent's output in the PipelineContext. This replaces the ad-hoc
key-flattening that was in the router.
"""

from __future__ import annotations

from typing import Any

import structlog

from mas.schemas.context import (
    AnalysisResult,
    ChunkItem,
    ContentItem,
    DocumentMeta,
    EntityItem,
    PipelineContext,
    RelationshipItem,
    RetrievedItem,
)

logger = structlog.get_logger()


def merge_agent_output(
    ctx: PipelineContext,
    agent_type: str,
    output: dict[str, Any],
) -> PipelineContext:
    """
    Merge an agent's output into the correct PipelineContext slots.

    Each agent type has a known output shape — the merger maps agent
    output keys to PipelineContext fields. No more ad-hoc key guessing.
    """
    merger = _MERGERS.get(agent_type)
    if merger:
        try:
            ctx = merger(ctx, output)
        except Exception as e:
            logger.warning("context_merger.failed", agent_type=agent_type, error=str(e))
    else:
        # Unknown agent — store in extra
        ctx.extra[agent_type] = output

    return ctx


# --- Per-agent-type mergers ---


def _merge_raganything(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    """RAG-Anything agent output — covers ingestion, processing, and retrieval."""
    if "doc_id" in output:
        ctx.rag_doc_id = output["doc_id"]
    if output.get("indexed"):
        ctx.rag_indexed = True
    if "content_items" in output:
        ctx.content_items = [ContentItem(**i) if isinstance(i, dict) else i for i in output["content_items"]]
    if "chunks" in output:
        ctx.chunks = [ChunkItem(**c) if isinstance(c, dict) else c for c in output["chunks"]]
    if "entities" in output:
        ctx.entities = [EntityItem(**e) if isinstance(e, dict) else e for e in output["entities"]]
    if "relationships" in output:
        ctx.relationships = [RelationshipItem(**r) if isinstance(r, dict) else r for r in output["relationships"]]
    if "retrieved" in output:
        ctx.retrieved = [RetrievedItem(**r) if isinstance(r, dict) else r for r in output["retrieved"]]
    if "response" in output:
        ctx.rag_response = output["response"]
    if "document" in output and isinstance(output["document"], dict):
        ctx.document = DocumentMeta(**output["document"])
    return ctx


def _merge_ingestion(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    """Legacy ingestion agent output."""
    if "items" in output:
        ctx.content_items = [ContentItem(**i) if isinstance(i, dict) else i for i in output["items"]]
    if "file_path" in output:
        ctx.document = DocumentMeta(
            file_path=output["file_path"],
            file_type=output.get("file_type", ""),
            content_hash=output.get("content_hash", ""),
            file_size_bytes=output.get("stats", {}).get("file_size_bytes", 0),
        )
    return ctx


def _merge_separator(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    if "streams" in output:
        ctx.streams = output["streams"]
    if "text_aggregate" in output:
        ctx.text_aggregate = output["text_aggregate"]
    return ctx


def _merge_chunker(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    if "chunks" in output:
        ctx.chunks = [ChunkItem(**c) if isinstance(c, dict) else c for c in output["chunks"]]
    return ctx


def _merge_modal_processor(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    """Shared merger for text_processor, image_processor, table_processor."""
    if "entities" in output:
        new_entities = [EntityItem(**e) if isinstance(e, dict) else e for e in output["entities"]]
        ctx.entities.extend(new_entities)
    if "relationships" in output:
        new_rels = [RelationshipItem(**r) if isinstance(r, dict) else r for r in output["relationships"]]
        ctx.relationships.extend(new_rels)
    return ctx


def _merge_kg_builder(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    if "nodes" in output:
        ctx.graph_nodes = output["nodes"]
    if "edges" in output:
        ctx.graph_edges = output["edges"]
    return ctx


def _merge_retriever(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    if "retrieved" in output:
        ctx.retrieved = [RetrievedItem(**r) if isinstance(r, dict) else r for r in output["retrieved"]]
    return ctx


def _merge_analyst(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    ctx.analysis = AnalysisResult(
        answer=output.get("answer", ""),
        key_points=output.get("key_points", []),
        citations=output.get("citations", []),
        confidence=output.get("confidence", "medium"),
        limitations=output.get("limitations", []),
    )
    return ctx


def _merge_synthesizer(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    ctx.synthesis = output
    return ctx


def _merge_kg_query(ctx: PipelineContext, output: dict[str, Any]) -> PipelineContext:
    if "answer" in output:
        ctx.analysis = AnalysisResult(
            answer=output.get("answer", ""),
            confidence=str(output.get("confidence", "medium")),
        )
    return ctx


# Registry: agent_type → merger function
_MERGERS = {
    "raganything": _merge_raganything,
    "ingestion": _merge_ingestion,
    "separator": _merge_separator,
    "chunker": _merge_chunker,
    "text_processor": _merge_modal_processor,
    "image_processor": _merge_modal_processor,
    "table_processor": _merge_modal_processor,
    "kg_builder": _merge_kg_builder,
    "hybrid_retriever": _merge_retriever,
    "embedder": lambda ctx, _: ctx,  # embedder writes to storage, not context
    "analyst": _merge_analyst,
    "synthesizer": _merge_synthesizer,
    "kg_query": _merge_kg_query,
}
