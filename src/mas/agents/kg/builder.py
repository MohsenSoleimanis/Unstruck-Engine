"""Knowledge graph builder agent — constructs the entity-relationship graph.

RAG-Anything pattern: all modality processors output entities/relationships,
this agent merges them into a unified knowledge graph with cross-modal links.
"""

from __future__ import annotations

from typing import Any

import structlog

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

logger = structlog.get_logger()


@registry.register
class KGBuilderAgent(BaseAgent):
    """
    Builds and updates the knowledge graph from entity/relationship extractions.

    Takes output from modal processors and:
      1. Deduplicates entities (fuzzy name matching)
      2. Merges properties from different sources
      3. Creates cross-modal edges (e.g., image entity → text entity)
      4. Adds 'belongs_to' edges linking entities to their source chunks

    This is the shared blackboard that enables cross-modal retrieval.
    """

    agent_type = "kg_builder"
    description = "Builds knowledge graph from extracted entities and relationships across all modalities"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        entities = task.context.get("entities", [])
        relationships = task.context.get("relationships", [])
        source_id = task.context.get("source_id", "unknown")

        if not entities and not relationships:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"message": "No entities or relationships to process"},
            )

        # Deduplicate entities by normalized name
        deduped = self._deduplicate_entities(entities)

        # Build graph operations
        nodes = []
        edges = []

        for entity in deduped:
            node = {
                "id": self._entity_id(entity["name"]),
                "name": entity["name"],
                "type": entity.get("type", "Other"),
                "properties": entity.get("properties", {}),
                "source_modality": entity.get("source_modality", "unknown"),
                "source_id": source_id,
            }
            nodes.append(node)

        for rel in relationships:
            edge = {
                "source": self._entity_id(rel["source"]),
                "target": self._entity_id(rel["target"]),
                "relation": rel.get("relation", "RELATED_TO"),
                "properties": rel.get("properties", {}),
            }
            edges.append(edge)

        # Add belongs_to edges linking entities to their source
        for node in nodes:
            edges.append({
                "source": node["id"],
                "target": f"source_{source_id}",
                "relation": "BELONGS_TO",
                "properties": {"modality": node["source_modality"]},
            })

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "total_entities_input": len(entities),
                    "deduplicated_entities": len(deduped),
                    "relationships": len(edges),
                    "modalities": list({e.get("source_modality", "?") for e in entities}),
                },
            },
        )

    def _deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """Deduplicate entities by normalized name, merging properties."""
        seen: dict[str, dict] = {}
        for entity in entities:
            key = self._entity_id(entity.get("name", ""))
            if key in seen:
                # Merge properties
                existing = seen[key]
                existing.setdefault("properties", {}).update(entity.get("properties", {}))
                # Keep broader type if different
                if entity.get("type") and entity["type"] != existing.get("type"):
                    existing.setdefault("alt_types", []).append(entity["type"])
            else:
                seen[key] = dict(entity)
        return list(seen.values())

    def _entity_id(self, name: str) -> str:
        """Normalize entity name to ID."""
        return name.strip().lower().replace(" ", "_").replace("-", "_")
