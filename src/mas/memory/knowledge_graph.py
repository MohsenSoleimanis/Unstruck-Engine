"""Knowledge graph memory — MAGMA-inspired multi-relation graph store."""

from __future__ import annotations

from typing import Any

import networkx as nx
import structlog

from mas.utils.security import safe_label

logger = structlog.get_logger()


class KnowledgeGraph:
    """
    Graph-based memory for capturing semantic, temporal, causal, and entity relations.

    Inspired by MAGMA architecture and RAG-Anything's shared KG pattern.
    Uses NetworkX locally; can be backed by Neo4j for production scale.

    Agents write entities and relationships; the graph enables cross-modal
    and cross-agent knowledge discovery.
    """

    def __init__(self, neo4j_uri: str | None = None, neo4j_user: str = "", neo4j_password: str = ""):
        self._graph = nx.DiGraph()
        self._neo4j_uri = neo4j_uri
        self._neo4j_driver = None
        if neo4j_uri:
            self._init_neo4j(neo4j_uri, neo4j_user, neo4j_password)

    def _init_neo4j(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase

            self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("knowledge_graph.neo4j_connected", uri=uri)
        except Exception as e:
            logger.warning("knowledge_graph.neo4j_fallback", error=str(e))

    def add_entity(self, entity_id: str, entity_type: str, properties: dict[str, Any] | None = None) -> None:
        props = properties or {}
        props["entity_type"] = entity_type
        self._graph.add_node(entity_id, **props)

        if self._neo4j_driver:
            self._neo4j_add_entity(entity_id, entity_type, props)

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        props = properties or {}
        props["relation_type"] = relation_type
        self._graph.add_edge(source, target, **props)

        if self._neo4j_driver:
            self._neo4j_add_relationship(source, target, relation_type, props)

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        if entity_id not in self._graph:
            return None
        return dict(self._graph.nodes[entity_id])

    def get_neighbors(self, entity_id: str, relation_type: str | None = None) -> list[dict[str, Any]]:
        if entity_id not in self._graph:
            return []
        results = []
        for _, target, data in self._graph.edges(entity_id, data=True):
            if relation_type and data.get("relation_type") != relation_type:
                continue
            results.append({"entity_id": target, "relation": data, **self._graph.nodes[target]})
        return results

    def search_entities(self, entity_type: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        results = []
        for node, data in self._graph.nodes(data=True):
            if entity_type and data.get("entity_type") != entity_type:
                continue
            if all(data.get(k) == v for k, v in filters.items()):
                results.append({"entity_id": node, **data})
        return results

    def get_subgraph(self, entity_id: str, depth: int = 2) -> dict[str, Any]:
        """Get ego graph around an entity up to given depth."""
        if entity_id not in self._graph:
            return {"nodes": [], "edges": []}
        subgraph = nx.ego_graph(self._graph, entity_id, radius=depth)
        return {
            "nodes": [{"id": n, **subgraph.nodes[n]} for n in subgraph.nodes],
            "edges": [
                {"source": u, "target": v, **d} for u, v, d in subgraph.edges(data=True)
            ],
        }

    @property
    def stats(self) -> dict[str, int]:
        return {"nodes": self._graph.number_of_nodes(), "edges": self._graph.number_of_edges()}

    # --- Neo4j sync helpers ---

    def _neo4j_add_entity(self, entity_id: str, entity_type: str, props: dict) -> None:
        label = safe_label(entity_type)
        with self._neo4j_driver.session() as session:
            safe_props = {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool))}
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=entity_id,
                props=safe_props,
            )

    def _neo4j_add_relationship(self, source: str, target: str, rel_type: str, props: dict) -> None:
        rel_label = safe_label(rel_type)
        with self._neo4j_driver.session() as session:
            safe_props = {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool))}
            session.run(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                f"MERGE (a)-[r:{rel_label}]->(b) SET r += $props",
                src=source,
                tgt=target,
                props=safe_props,
            )
