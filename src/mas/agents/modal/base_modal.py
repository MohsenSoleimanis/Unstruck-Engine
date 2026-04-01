"""Base modal processor — the pattern every modality-specific agent follows.

Direct from RAG-Anything: BaseModalProcessor defines the contract.
Each modality implements generate_description() and extract_entities().
Output always feeds into the knowledge graph.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task
from mas.utils.parsing import extract_json


class BaseModalProcessor(BaseAgent):
    """
    Base class for modality-specific processors.

    Every modal processor must:
      1. Generate a semantic description of the content
      2. Extract entities and relationships for the knowledge graph
      3. Return structured output that can be indexed and retrieved

    This is the RAG-Anything pattern: each modality gets specialized
    processing, but all output the same entity/relationship format.
    """

    @abstractmethod
    async def generate_description(self, item: dict[str, Any], context: str) -> str:
        """Generate a natural language description of this content item."""

    @abstractmethod
    async def extract_entities(self, item: dict[str, Any], description: str) -> list[dict[str, Any]]:
        """Extract entities and relationships from this content item."""

    async def execute(self, task: Task) -> AgentResult:
        items = task.context.get("items", [])
        context = task.context.get("document_context", "")

        if not items:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"processed": []},
            )

        processed = []
        all_entities: list[dict[str, Any]] = []

        for item in items:
            try:
                description = await self.generate_description(item, context)
                entities = await self.extract_entities(item, description)

                processed.append({
                    "type": item.get("type"),
                    "page_idx": item.get("page_idx"),
                    "source": item.get("source"),
                    "description": description,
                    "entity_count": len(entities),
                })
                all_entities.extend(entities)
            except Exception as e:
                processed.append({
                    "type": item.get("type"),
                    "page_idx": item.get("page_idx"),
                    "error": str(e),
                })

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "processed": processed,
                "entities": all_entities,
                "total_processed": len(processed),
                "total_entities": len(all_entities),
            },
        )

    async def _llm_call(self, system: str, user: str) -> str:
        """Helper for LLM calls."""
        response = await self.llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return response.content

    def _parse_json(self, raw: str) -> Any:
        """Parse JSON from LLM output. Delegates to shared utility."""
        return extract_json(raw)
