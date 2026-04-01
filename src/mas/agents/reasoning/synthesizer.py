"""Synthesizer agent — combines outputs from multiple agents into a final answer."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

SYNTHESIS_PROMPT = """Combine these agent outputs into a single coherent answer.

Original question: {question}

Agent outputs:
{outputs}

Rules:
1. Resolve contradictions — prefer higher-confidence sources.
2. Merge complementary information from different modalities.
3. Cite which agent/modality each piece of information came from.
4. Produce a structured, comprehensive final answer.

Output JSON:
{{
  "answer": "final synthesized answer",
  "key_findings": ["finding 1", "finding 2"],
  "sources_used": ["agent_type: what it contributed"],
  "modalities_covered": ["text", "table", "image"],
  "confidence": "high|medium|low"
}}
"""


@registry.register
class SynthesizerAgent(BaseAgent):
    """
    Final synthesis — combines multi-agent, multi-modal outputs into one answer.

    This is the last step in the pipeline: takes outputs from analyst,
    graph query, and modal processors and produces the user-facing result.
    """

    agent_type = "synthesizer"
    description = "Combines and synthesizes outputs from multiple agents and modalities into a final answer"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        agent_outputs = task.context.get("agent_outputs", {})
        question = task.context.get("original_query", task.instruction)

        outputs_text = json.dumps(agent_outputs, indent=2, default=str)[:10000]

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a synthesis expert. Output valid JSON only."),
            HumanMessage(content=SYNTHESIS_PROMPT.format(question=question, outputs=outputs_text)),
        ])

        try:
            raw = response.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            output = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            output = {"answer": response.content}

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
