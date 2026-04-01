"""Content separator agent — routes content items by modality.

RAG-Anything's core pattern: separate_content() splits parsed output into
text content vs multimodal items, then each goes to the right processor.
"""

from __future__ import annotations

from typing import Any

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task


@registry.register
class ContentSeparatorAgent(BaseAgent):
    """
    Separates ingested content by modality and prepares routing.

    Input: unified content items from IngestionAgent
    Output: separated streams — text, tables, images, structured, equations
    Each stream gets routed to the appropriate ModalProcessor.
    """

    agent_type = "separator"
    description = "Separates content items by modality (text, images, tables, structured) for specialized processing"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        items = task.context.get("items", [])
        if not items:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"streams": {}, "message": "No items to separate"},
            )

        streams: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            modality = item.get("type", "unknown")
            streams.setdefault(modality, []).append(item)

        # Build text aggregate (all text items joined)
        text_aggregate = ""
        for item in streams.get("text", []):
            content = item.get("content", "")
            if isinstance(content, str):
                page = item.get("page_idx", "")
                prefix = f"[Page {page}] " if page else ""
                text_aggregate += f"{prefix}{content}\n\n"

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "streams": dict(streams),
                "text_aggregate": text_aggregate[:50000],
                "modalities_found": list(streams.keys()),
                "counts": {k: len(v) for k, v in streams.items()},
            },
        )
