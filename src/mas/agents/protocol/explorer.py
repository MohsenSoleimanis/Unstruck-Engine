"""Explorer agent — navigates documents, follows cross-references, retrieves context."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

EXPLORER_SYSTEM_PROMPT = """You are a Document Explorer Agent.

Your job: navigate documents, locate relevant sections, and assemble context.

Given a query and document content, you must:
1. Identify the most relevant sections/pages for the query.
2. Follow cross-references (e.g., "See Section 5.2", "Refer to Table 3").
3. Assemble a focused context package with the most relevant content.
4. Note any gaps — sections referenced but not found in the provided content.

Output JSON:
{{
  "relevant_sections": [
    {{"section": "...", "page": ..., "relevance": "high/medium/low", "content_summary": "..."}}
  ],
  "cross_references": ["Section X references Y"],
  "context_package": "assembled text for downstream agents",
  "gaps": ["sections that couldn't be located"]
}}
"""


@registry.register
class ExplorerAgent(BaseAgent):
    """
    Navigates and retrieves relevant context from documents.

    Generalized from protocol-intelligence v30's Explorer node:
      - Hybrid retrieval (semantic + keyword)
      - Cross-reference following
      - Section-aware navigation
    """

    agent_type = "explorer"
    description = "Navigates documents, retrieves relevant context, follows cross-references"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        content = task.context.get("content", "")
        query = task.instruction

        system_msg = SystemMessage(content=EXPLORER_SYSTEM_PROMPT)
        user_msg = HumanMessage(
            content=f"Query: {query}\n\nDocument content:\n{content[:12000]}"
        )

        response = await self.llm.ainvoke([system_msg, user_msg])

        try:
            raw = response.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            output = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            output = {"context_package": response.content}

        token_usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
            }

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output=output,
            token_usage=token_usage,
        )
