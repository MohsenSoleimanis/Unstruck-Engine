"""Research agent — deep analysis, question answering, and reasoning."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

RESEARCHER_SYSTEM_PROMPT = """You are a Research Agent specialized in deep analysis and reasoning.

Given context and a research question, you must:
1. Analyze the provided content thoroughly.
2. Identify key findings, patterns, and insights.
3. Draw connections between different pieces of information.
4. Provide evidence-based answers with citations to source material.
5. Flag uncertainties and areas needing further investigation.

Output JSON:
{{
  "findings": [
    {{"finding": "...", "evidence": "...", "confidence": 0.0-1.0, "source": "..."}}
  ],
  "analysis": "detailed analysis text",
  "connections": ["connection between finding A and B"],
  "uncertainties": ["areas needing more investigation"],
  "recommendations": ["next steps"]
}}
"""


@registry.register
class ResearcherAgent(BaseAgent):
    """
    Deep analysis and research agent.

    Handles analytical tasks: question answering, pattern recognition,
    insight extraction, and evidence-based reasoning over retrieved context.
    """

    agent_type = "researcher"
    description = "Performs deep analysis, answers research questions, identifies patterns and insights"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        context = task.context.get("content", "")
        retrieved = task.context.get("retrieved", [])

        # Assemble context from retrieved items if provided
        if retrieved and not context:
            context = "\n\n".join(
                str(r.get("item", {}).get("content", r)) for r in retrieved
            )

        system_msg = SystemMessage(content=RESEARCHER_SYSTEM_PROMPT)
        user_msg = HumanMessage(
            content=f"Research question: {task.instruction}\n\nContext:\n{context[:10000]}"
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
            output = {"analysis": response.content}

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
