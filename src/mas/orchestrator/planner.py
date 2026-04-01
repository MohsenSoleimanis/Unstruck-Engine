"""Task planner — decomposes user queries into executable task plans."""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.registry import AgentRegistry
from mas.schemas.tasks import Task, TaskPriority
from mas.utils.parsing import extract_json

logger = structlog.get_logger()

PLANNER_SYSTEM_PROMPT = """You are the Planner in a multi-agent orchestration system.

Your job: decompose the user's request into a set of concrete tasks that specialized agents can execute.

Available agents:
{agent_list}

Rules:
1. Each task must target exactly one agent_type from the available agents.
2. Tasks can have dependencies (task IDs they depend on).
3. Independent tasks should have NO dependencies so they can run in parallel.
4. Output valid JSON: a list of task objects.
5. Keep tasks focused — one clear instruction per task.
6. For complex requests, use a multi-phase approach:
   - Phase 1: Information gathering (parsing, retrieval, exploration)
   - Phase 2: Processing (extraction, analysis, transformation)
   - Phase 3: Review & validation (cross-checking, consistency)

Output format (JSON array):
[
  {{
    "agent_type": "...",
    "instruction": "Clear, specific instruction for the agent",
    "context": {{}},
    "dependencies": [],
    "priority": "medium"
  }}
]
"""


class Planner:
    """
    Decomposes user queries into executable task graphs.

    Inspired by:
      - Magentic-One's Task Ledger
      - TEA Protocol's dynamic task planning
      - Your protocol-engine's parallel extraction + sequential review pattern
    """

    def __init__(self, llm: BaseChatModel, registry: AgentRegistry) -> None:
        self.llm = llm
        self.registry = registry

    async def plan(self, user_query: str, context: dict[str, Any] | None = None) -> list[Task]:
        """Generate a task plan from a user query."""
        agent_list = "\n".join(
            f"  - {a['agent_type']}: {a['description']}" for a in self.registry.list_agents()
        )

        system_msg = SystemMessage(content=PLANNER_SYSTEM_PROMPT.format(agent_list=agent_list))
        user_content = f"User request: {user_query}"
        if context:
            user_content += f"\n\nAdditional context: {json.dumps(context, default=str)}"

        response = await self.llm.ainvoke([system_msg, HumanMessage(content=user_content)])
        return self._parse_plan(response.content)

    async def replan(
        self,
        original_query: str,
        completed_results: list[dict[str, Any]],
        failed_tasks: list[str],
        progress_summary: str,
    ) -> list[Task]:
        """Re-plan based on progress so far (Progress Ledger pattern)."""
        agent_list = "\n".join(
            f"  - {a['agent_type']}: {a['description']}" for a in self.registry.list_agents()
        )

        system_msg = SystemMessage(content=PLANNER_SYSTEM_PROMPT.format(agent_list=agent_list))
        user_content = (
            f"Original request: {original_query}\n\n"
            f"Progress so far:\n{progress_summary}\n\n"
            f"Failed tasks: {failed_tasks}\n\n"
            f"Completed results summary: {json.dumps(completed_results[:5], default=str)}\n\n"
            "Generate ONLY the remaining tasks needed to complete the request. "
            "Do not repeat already-completed work."
        )

        response = await self.llm.ainvoke([system_msg, HumanMessage(content=user_content)])
        return self._parse_plan(response.content)

    def _parse_plan(self, raw: str) -> list[Task]:
        """Parse LLM output into Task objects."""
        try:
            items = extract_json(raw)
            tasks = []
            for i, item in enumerate(items):
                task = Task(
                    agent_type=item["agent_type"],
                    instruction=item["instruction"],
                    context=item.get("context", {}),
                    dependencies=item.get("dependencies", []),
                    priority=TaskPriority(item.get("priority", "medium")),
                )
                tasks.append(task)

            # Resolve dependency references (index-based to ID-based)
            for task in tasks:
                resolved = []
                for dep in task.dependencies:
                    if isinstance(dep, int) and 0 <= dep < len(tasks):
                        resolved.append(tasks[dep].id)
                    elif isinstance(dep, str):
                        resolved.append(dep)
                task.dependencies = resolved

            logger.info("planner.plan_created", task_count=len(tasks))
            return tasks

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error("planner.parse_error", error=str(e), raw=raw[:500])
            return []
