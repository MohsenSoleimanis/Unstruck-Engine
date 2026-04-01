"""Multimodal processing agent — handles images, tables, equations via vision models."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

IMAGE_ANALYSIS_PROMPT = """Analyze this image in detail. Provide:
1. A comprehensive description of what is shown
2. Key entities, labels, or text visible
3. Relationships between elements
4. Any data or statistics presented

Context from the surrounding document:
{context}

Output JSON:
{{
  "description": "detailed description",
  "entities": ["entity1", "entity2"],
  "relationships": ["entity1 -> entity2: relationship"],
  "data_points": {{}},
  "content_type": "diagram|chart|photo|table|figure"
}}
"""

TABLE_ANALYSIS_PROMPT = """Analyze this table data. Provide:
1. A semantic summary of what the table represents
2. Key patterns or insights in the data
3. Column descriptions and data types
4. Notable values or outliers

Table data:
{table_data}

Context:
{context}

Output JSON:
{{
  "summary": "what this table represents",
  "columns": [{{"name": "...", "type": "...", "description": "..."}}],
  "patterns": ["pattern1"],
  "key_values": {{}},
  "row_count": 0
}}
"""


@registry.register
class MultimodalAgent(BaseAgent):
    """
    Processes multimodal content (images, tables, equations).

    Inspired by RAG-Anything's modality-specific processors:
      - ImageModalProcessor: vision-based analysis
      - TableModalProcessor: structured data analysis
      - ContextExtractor: surrounding document context injection

    Each modality gets a specialized prompt and processing path,
    but all output entities/relationships for the knowledge graph.
    """

    agent_type = "multimodal"
    description = "Processes images, tables, and other non-text content using vision and specialized analysis"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        items = task.context.get("multimodal_items", [])
        context = task.context.get("document_context", "")

        if not items:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"processed": [], "message": "No multimodal items provided"},
            )

        processed = []
        total_tokens: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        for item in items:
            item_type = item.get("type", "unknown")

            if item_type == "image":
                result = await self._process_image(item, context)
            elif item_type == "table":
                result = await self._process_table(item, context)
            else:
                result = {"type": item_type, "description": f"Unhandled modality: {item_type}"}

            processed.append(result)

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "processed_items": processed,
                "total_processed": len(processed),
                "modalities": list({i.get("type") for i in items}),
            },
            token_usage=total_tokens,
        )

    async def _process_image(self, item: dict, context: str) -> dict:
        """Process an image item using vision model."""
        image_path = item.get("path", "")

        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            messages = [
                SystemMessage(content="You are a visual analysis assistant. Output valid JSON only."),
                HumanMessage(content=[
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT.format(context=context[:1000])},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]),
            ]
        else:
            # No image file — describe what we know
            messages = [
                SystemMessage(content="You are an analysis assistant."),
                HumanMessage(
                    content=f"Image reference at page {item.get('page_idx', '?')}. "
                    f"Context: {context[:500]}. Describe what this image likely contains."
                ),
            ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
        except Exception:
            result = {"description": "Image processing failed", "error": True}

        result["type"] = "image"
        result["page_idx"] = item.get("page_idx")
        return result

    async def _process_table(self, item: dict, context: str) -> dict:
        """Process a table item."""
        table_data = item.get("data", [])
        table_str = "\n".join(
            " | ".join(str(cell) for cell in row) for row in table_data
        ) if isinstance(table_data, list) else str(table_data)

        messages = [
            SystemMessage(content="You are a data analysis assistant. Output valid JSON only."),
            HumanMessage(content=TABLE_ANALYSIS_PROMPT.format(
                table_data=table_str[:3000],
                context=context[:500],
            )),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
        except Exception:
            result = {"summary": "Table processing failed", "error": True}

        result["type"] = "table"
        result["page_idx"] = item.get("page_idx")
        return result
