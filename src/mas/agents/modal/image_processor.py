"""Image modal processor — vision-based analysis for any image content.

RAG-Anything pattern: encode to base64, send to VLM, extract entities.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.modal.base_modal import BaseModalProcessor
from mas.agents.registry import registry

logger = structlog.get_logger()

IMAGE_DESCRIPTION_PROMPT = """Analyze this image comprehensively. Describe:
1. What is shown (diagram, chart, photo, figure, screenshot, etc.)
2. All text, labels, and annotations visible
3. Key entities and their relationships
4. Any data, statistics, or measurements presented
5. Spatial layout and visual structure

Surrounding document context:
{context}

Be detailed and precise. Your description will be used for knowledge graph construction and retrieval."""

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this image description.

Description: {description}

Output JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "relation": "...", "properties": {{}}}}
  ]
}}

Entity types: Person, Organization, Location, Concept, Metric, Object, Process, Technology, Date, Other
Be exhaustive — extract every identifiable entity."""


@registry.register
class ImageProcessor(BaseModalProcessor):
    """Processes images using vision models. Extracts descriptions and entities."""

    agent_type = "image_processor"
    description = "Analyzes images via vision models — extracts descriptions, entities, and relationships"
    version = "0.1.0"

    async def generate_description(self, item: dict[str, Any], context: str) -> str:
        content_data = item.get("content")

        if isinstance(content_data, bytes):
            img_b64 = base64.b64encode(content_data).decode()
            messages = [
                SystemMessage(content="You are a visual analysis expert."),
                HumanMessage(content=[
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT.format(context=context[:1500])},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]),
            ]
            response = await self.llm.ainvoke(messages)
            return response.content
        else:
            # No image bytes — generate description from metadata
            page = item.get("page_idx", "?")
            return await self._llm_call(
                "You are an analyst.",
                f"An image was found on page {page}. Context: {context[:500]}. "
                "Based on context, what might this image show?",
            )

    async def extract_entities(self, item: dict[str, Any], description: str) -> list[dict[str, Any]]:
        raw = await self._llm_call(
            "You are an entity extraction assistant. Output valid JSON only.",
            ENTITY_EXTRACTION_PROMPT.format(description=description),
        )
        try:
            data = self._parse_json(raw)
            entities = data.get("entities", [])
            for e in entities:
                e["source_modality"] = "image"
                e["page_idx"] = item.get("page_idx")
            return entities
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("image_processor.entity_parse_failed", error=str(exc))
            return []
