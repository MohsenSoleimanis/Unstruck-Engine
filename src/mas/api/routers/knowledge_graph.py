"""Knowledge graph API router."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/kg", tags=["knowledge_graph"])


def _get_kg():
    from mas.api.server import app
    return app.state.pipeline.knowledge_graph


@router.get("/graph")
async def get_graph(limit: int = 500):
    """Return full graph as nodes + edges (capped)."""
    kg = _get_kg()
    nodes = []
    for node_id, data in list(kg._graph.nodes(data=True))[:limit]:
        nodes.append({"id": node_id, **data})
    edges = []
    for src, tgt, data in list(kg._graph.edges(data=True))[:limit * 2]:
        edges.append({"source": src, "target": tgt, **data})
    return {"nodes": nodes, "edges": edges, "stats": kg.stats}


@router.get("/subgraph/{entity_id}")
async def get_subgraph(entity_id: str, depth: int = 2):
    """Get ego graph around an entity."""
    return _get_kg().get_subgraph(entity_id, depth=min(depth, 5))


@router.get("/search")
async def search_entities(entity_type: str | None = None, q: str | None = None):
    """Search entities by type or properties."""
    kg = _get_kg()
    results = kg.search_entities(entity_type=entity_type)
    if q:
        q_lower = q.lower()
        results = [e for e in results if q_lower in str(e).lower()]
    return results[:100]


@router.get("/stats")
async def get_stats():
    return _get_kg().stats
