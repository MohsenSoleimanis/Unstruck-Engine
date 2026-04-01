"""Schema-driven extraction agent — generalized from protocol-intelligence extractor."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

EXTRACTOR_SYSTEM_PROMPT = """You are a Schema-Driven Extraction Agent.

Your job: extract structured data from provided content according to a schema.

Rules:
1. Extract ONLY what is explicitly present in the source content.
2. For every extracted field, provide:
   - The value
   - source_text: exact quote from the content (10-80 chars)
   - confidence: 0.9 (verbatim), 0.7 (paraphrased), 0.5 (inferred)
   - page/section reference if available
3. If a field cannot be found, set it to null with confidence 0.0.
4. Output valid JSON matching the requested schema.
5. Never hallucinate or infer beyond what the content states.

{schema_description}
"""


@registry.register
class ExtractorAgent(BaseAgent):
    """
    Generalized schema-driven extractor.

    Handles any extraction task by combining:
      - A Pydantic schema (or JSON schema) defining what to extract
      - Source content (text, tables, or structured data)
      - Domain-specific instructions

    Inspired by protocol-intelligence v30's single-prompt + schema pattern.
    """

    agent_type = "extractor"
    description = "Extracts structured data from content using a provided schema"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        content = task.context.get("content", "")
        schema = task.context.get("schema", {})
        domain = task.context.get("domain", "general")

        schema_desc = ""
        if schema:
            schema_desc = f"Target schema:\n```json\n{json.dumps(schema, indent=2)}\n```"

        system_msg = SystemMessage(
            content=EXTRACTOR_SYSTEM_PROMPT.format(schema_description=schema_desc)
        )
        user_msg = HumanMessage(
            content=f"Domain: {domain}\n\nInstruction: {task.instruction}\n\nContent to extract from:\n{content[:8000]}"
        )

        response = await self.llm.ainvoke([system_msg, user_msg])

        # Parse structured output
        try:
            raw = response.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            extracted = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            extracted = {"raw_response": response.content}

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
            output={"extraction": extracted, "domain": domain},
            token_usage=token_usage,
        )
