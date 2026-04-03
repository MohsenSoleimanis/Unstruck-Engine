"""Built-in agents — analyst, synthesizer, kg_reasoner.

Each agent:
  - Loads its prompt from prompts/ (via config)
  - Calls LLM through Context Engine (never directly)
  - Returns structured output as AgentResult
  - Handles errors-as-feedback (returns FAILED, not raises)
"""

from __future__ import annotations

import json
import re
from typing import Any

from unstruck.agents.base import BaseAgent
from unstruck.config import get_config
from unstruck.schemas import AgentResult, ResultStatus, Task


class AnalystAgent(BaseAgent):
    """Reasons over retrieved context to produce grounded answers with citations."""

    agent_type = "analyst"

    async def execute(self, task: Task) -> AgentResult:
        config = get_config()
        prompt_template = config.load_prompt(
            config.get_agent_config("analyst")["prompt"]
        )

        # Build context from what's available in the task
        context = task.context.get("retrieved_context", "")
        if not context:
            context = task.context.get("rag_response", "")
        if not context:
            context = task.context.get("text_content", "")

        if not context:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"answer": "No context available to answer the question.", "confidence": "low"},
            )

        prompt = prompt_template.format(
            context=context,
            question=task.instruction,
        )

        result = await self.llm_call(
            system_prompt="You are an Analyst agent. Output valid JSON only.",
            user_prompt=prompt,
        )

        if result.blocked:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[f"LLM call blocked: {result.block_reason}"],
            )

        output = _parse_json(result.text, {"answer": result.text, "confidence": "medium"})

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output=output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )


class SynthesizerAgent(BaseAgent):
    """Fuses outputs from multiple agents into a coherent final answer."""

    agent_type = "synthesizer"

    async def execute(self, task: Task) -> AgentResult:
        config = get_config()
        prompt_template = config.load_prompt(
            config.get_agent_config("synthesizer")["prompt"]
        )

        agent_outputs = task.context.get("agent_outputs", {})
        question = task.context.get("original_query", task.instruction)

        prompt = prompt_template.format(
            question=question,
            agent_outputs=json.dumps(agent_outputs, indent=2, default=str)[:10000],
        )

        result = await self.llm_call(
            system_prompt="You are a Synthesizer agent. Output valid JSON only.",
            user_prompt=prompt,
        )

        if result.blocked:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[f"LLM call blocked: {result.block_reason}"],
            )

        output = _parse_json(result.text, {"answer": result.text})

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output=output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )


class KGReasonerAgent(BaseAgent):
    """Graph traversal + LLM reasoning for multi-hop questions."""

    agent_type = "kg_reasoner"

    async def execute(self, task: Task) -> AgentResult:
        config = get_config()
        prompt_template = config.load_prompt(
            config.get_agent_config("kg_reasoner")["prompt"]
        )

        kg_context = task.context.get("kg_context", "")
        question = task.instruction

        if not kg_context:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"answer": "No knowledge graph context available.", "confidence": 0.0},
            )

        prompt = prompt_template.format(
            kg_context=kg_context,
            question=question,
        )

        result = await self.llm_call(
            system_prompt="You are a Knowledge Graph Reasoner. Output valid JSON only.",
            user_prompt=prompt,
        )

        if result.blocked:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[f"LLM call blocked: {result.block_reason}"],
            )

        output = _parse_json(result.text, {"answer": result.text, "confidence": 0.5})

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output=output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )


def _parse_json(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON from LLM output. Returns fallback on failure."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return fallback
