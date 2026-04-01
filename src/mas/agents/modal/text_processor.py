"""Text modal processor — entity and relationship extraction from text content."""

from __future__ import annotations

import json
from typing import Any

import structlog

from mas.agents.modal.base_modal import BaseModalProcessor
from mas.agents.registry import registry

logger = structlog.get_logger()

TEXT_ENTITY_PROMPT = """Extract all entities and relationships from this text.

Text:
{text}

Output JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "relation": "...", "properties": {{}}}}
  ]
}}

Entity types: Person, Organization, Location, Concept, Metric, Process, Technology, Date, Product, Event, Other
Be exhaustive. Extract every identifiable entity and every relationship between them."""


@registry.register
class TextProcessor(BaseModalProcessor):
    """Processes text content — extracts entities and relationships for the knowledge graph."""

    agent_type = "text_processor"
    description = "Extracts entities and relationships from text for knowledge graph construction"
    version = "0.1.0"

    async def generate_description(self, item: dict[str, Any], context: str) -> str:
        text = item.get("content", "")
        if isinstance(text, str) and len(text) > 200:
            summary = await self._llm_call(
                "You are a summarization expert.",
                f"Summarize this text in 2-3 sentences:\n\n{text[:3000]}",
            )
            return summary
        return text if isinstance(text, str) else str(text)

    async def extract_entities(self, item: dict[str, Any], description: str) -> list[dict[str, Any]]:
        text = item.get("content", "")
        if not isinstance(text, str) or len(text) < 20:
            return []

        raw = await self._llm_call(
            "You are an entity extraction assistant. Output valid JSON only.",
            TEXT_ENTITY_PROMPT.format(text=text[:5000]),
        )
        try:
            data = self._parse_json(raw)
            entities = data.get("entities", [])
            for e in entities:
                e["source_modality"] = "text"
                e["page_idx"] = item.get("page_idx")
            return entities
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("text_processor.entity_parse_failed", error=str(exc))
            return []
