"""Reviewer agent — cross-checks, validates, and ensures consistency."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

REVIEWER_SYSTEM_PROMPT = """You are a Cross-Check & Validation Agent.

Your job: review extracted data for consistency, completeness, and accuracy.

Given extraction results and source content, you must:
1. Verify each extracted value against the source (grounding check).
2. Check internal consistency (e.g., numbers match across sections).
3. Identify missing fields or incomplete extractions.
4. Flag ambiguities or contradictions.
5. Assign a verification status to each field.

Output JSON:
{{
  "verification_results": [
    {{
      "field": "...",
      "status": "CONSISTENT | INCONSISTENT | AMBIGUOUS | CANNOT_VERIFY",
      "source_evidence": "exact quote supporting verification",
      "notes": "explanation if inconsistent or ambiguous"
    }}
  ],
  "overall_quality": {{
    "completeness": 0.0-1.0,
    "consistency": 0.0-1.0,
    "confidence": 0.0-1.0
  }},
  "issues": ["list of problems found"],
  "recommendations": ["suggested fixes or additional extractions needed"]
}}
"""


@registry.register
class ReviewerAgent(BaseAgent):
    """
    Validates and cross-checks agent outputs.

    Generalized from protocol-engine's Phase 2-3:
      - Flag-based cross-checking between agents
      - 4-level deterministic validation
      - Consistency checks across domains
    """

    agent_type = "reviewer"
    description = "Cross-checks, validates, and ensures consistency of extracted data"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        extraction = task.context.get("extraction", {})
        source_content = task.context.get("source_content", "")
        domain = task.context.get("domain", "general")

        system_msg = SystemMessage(content=REVIEWER_SYSTEM_PROMPT)
        user_msg = HumanMessage(
            content=(
                f"Domain: {domain}\n\n"
                f"Instruction: {task.instruction}\n\n"
                f"Extracted data to review:\n{json.dumps(extraction, indent=2, default=str)[:4000]}\n\n"
                f"Source content for verification:\n{source_content[:8000]}"
            )
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
            output = {"raw_review": response.content}

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
