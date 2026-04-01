"""Table modal processor — structured data analysis for any tabular content.

RAG-Anything pattern: analyze table structure, extract semantics, build entities.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from mas.agents.modal.base_modal import BaseModalProcessor
from mas.agents.registry import registry

logger = structlog.get_logger()

TABLE_DESCRIPTION_PROMPT = """Analyze this table data. Provide:
1. What this table represents (subject, purpose)
2. Column descriptions and data types
3. Key patterns, trends, or insights
4. Notable values, outliers, or relationships between rows/columns
5. Statistical summary if applicable

Table:
{table_text}

Surrounding document context:
{context}

Be detailed. Your description will be used for knowledge graph construction and retrieval."""

TABLE_ENTITY_PROMPT = """Extract entities and relationships from this table description.

Description: {description}

Table data:
{table_text}

Output JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "relation": "...", "properties": {{}}}}
  ]
}}

Extract entities from both column headers and cell values. Capture relationships between row entities."""


@registry.register
class TableProcessor(BaseModalProcessor):
    """Processes tabular data — extracts semantics, patterns, and entities."""

    agent_type = "table_processor"
    description = "Analyzes tables — extracts structure, patterns, entities, and relationships"
    version = "0.1.0"

    async def generate_description(self, item: dict[str, Any], context: str) -> str:
        table_data = item.get("content", [])
        table_text = self._format_table(table_data)

        return await self._llm_call(
            "You are a data analysis expert.",
            TABLE_DESCRIPTION_PROMPT.format(table_text=table_text[:4000], context=context[:1000]),
        )

    async def extract_entities(self, item: dict[str, Any], description: str) -> list[dict[str, Any]]:
        table_data = item.get("content", [])
        table_text = self._format_table(table_data)

        raw = await self._llm_call(
            "You are an entity extraction assistant. Output valid JSON only.",
            TABLE_ENTITY_PROMPT.format(description=description, table_text=table_text[:3000]),
        )
        try:
            data = self._parse_json(raw)
            entities = data.get("entities", [])
            for e in entities:
                e["source_modality"] = "table"
                e["page_idx"] = item.get("page_idx")
            return entities
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("table_processor.entity_parse_failed", error=str(exc))
            return []

    def _format_table(self, data: Any) -> str:
        if isinstance(data, list) and data:
            return "\n".join(
                " | ".join(str(cell) if cell else "" for cell in row)
                for row in data
                if isinstance(row, (list, tuple))
            )
        return str(data)
