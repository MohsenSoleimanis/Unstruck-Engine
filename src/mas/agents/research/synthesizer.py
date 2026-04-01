"""Synthesizer agent — combines outputs from multiple agents into a coherent final result."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

SYNTHESIZER_PROMPT = """You are a Synthesis Agent. Your job is to combine outputs from multiple
specialized agents into a coherent, comprehensive final result.

Given the outputs from various agents, you must:
1. Identify the key information from each agent's output.
2. Resolve any contradictions or inconsistencies between agents.
3. Organize the information into a logical, unified structure.
4. Highlight areas of strong agreement (high confidence) vs. disagreement.
5. Produce a final synthesized output.

Agent outputs:
{agent_outputs}

Original query: {query}

Output a well-structured JSON with the synthesized result.
"""


@registry.register
class SynthesizerAgent(BaseAgent):
    """
    Combines outputs from multiple agents into a unified result.

    Used as the final step in multi-agent pipelines to produce
    coherent output from parallel agent work.
    """

    agent_type = "synthesizer"
    description = "Combines and synthesizes outputs from multiple agents into a coherent final result"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        agent_outputs = task.context.get("agent_outputs", {})
        query = task.context.get("original_query", task.instruction)

        outputs_text = json.dumps(agent_outputs, indent=2, default=str)[:8000]

        system_msg = SystemMessage(content="You are a synthesis assistant. Output valid JSON only.")
        user_msg = HumanMessage(
            content=SYNTHESIZER_PROMPT.format(agent_outputs=outputs_text, query=query)
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
            output = {"synthesis": response.content}

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
