"""Task router — dispatches tasks to agents, manages parallel execution."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel

from mas.agents.base import BaseAgent
from mas.agents.registry import AgentRegistry
from mas.llmops.cost_tracker import CostTracker
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task, TaskStatus

logger = structlog.get_logger()


class Router:
    """
    Routes tasks to agents and manages execution.

    Handles:
      - Agent instantiation from registry
      - Parallel execution of independent tasks
      - Sequential execution of dependent tasks
      - Result collection and error handling
    """

    def __init__(
        self,
        registry: AgentRegistry,
        worker_llm: BaseChatModel,
        cost_tracker: CostTracker | None = None,
        max_concurrent: int = 5,
    ) -> None:
        self.registry = registry
        self.worker_llm = worker_llm
        self.cost_tracker = cost_tracker
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._agents: dict[str, BaseAgent] = {}

    def _get_agent(self, agent_type: str) -> BaseAgent:
        """Get or create an agent instance."""
        if agent_type not in self._agents:
            self._agents[agent_type] = self.registry.create(
                agent_type, self.worker_llm, cost_tracker=self.cost_tracker
            )
        return self._agents[agent_type]

    async def execute_task(self, task: Task) -> AgentResult:
        """Execute a single task via the appropriate agent."""
        async with self._semaphore:
            try:
                agent = self._get_agent(task.agent_type)
                task.status = TaskStatus.IN_PROGRESS
                result = await agent.run(task)
                task.status = TaskStatus.COMPLETED if result.status == ResultStatus.SUCCESS else TaskStatus.FAILED
                return result
            except Exception as e:
                task.status = TaskStatus.FAILED
                logger.error("router.task_failed", task_id=task.id, error=str(e))
                return AgentResult(
                    task_id=task.id,
                    agent_id="router",
                    agent_type=task.agent_type,
                    status=ResultStatus.FAILED,
                    errors=[str(e)],
                )

    async def execute_batch(self, tasks: list[Task]) -> list[AgentResult]:
        """
        Execute a batch of tasks respecting dependencies.

        Tasks with no unmet dependencies run in parallel.
        """
        results: list[AgentResult] = []
        completed_ids: set[str] = set()
        remaining = list(tasks)

        while remaining:
            # Find all tasks whose dependencies are met
            ready = [t for t in remaining if t.is_ready(completed_ids)]
            if not ready:
                # Deadlock — remaining tasks have unmet dependencies
                logger.error("router.deadlock", remaining=[t.id for t in remaining])
                for t in remaining:
                    t.status = TaskStatus.FAILED
                    results.append(AgentResult(
                        task_id=t.id,
                        agent_id="router",
                        agent_type=t.agent_type,
                        status=ResultStatus.FAILED,
                        errors=["Deadlock: unresolvable dependencies"],
                    ))
                break

            # Execute ready tasks in parallel
            batch_results = await asyncio.gather(
                *[self.execute_task(t) for t in ready],
                return_exceptions=True,
            )

            for task, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    result = AgentResult(
                        task_id=task.id,
                        agent_id="router",
                        agent_type=task.agent_type,
                        status=ResultStatus.FAILED,
                        errors=[str(result)],
                    )
                results.append(result)
                completed_ids.add(task.id)
                remaining.remove(task)

        return results

    async def execute_plan(self, plan: list[Task], results_so_far: list[AgentResult] | None = None) -> list[AgentResult]:
        """Execute a full plan, carrying forward any existing results."""
        completed_ids = set()
        if results_so_far:
            completed_ids = {r.task_id for r in results_so_far if r.status == ResultStatus.SUCCESS}

        # Filter out already-completed tasks
        pending = [t for t in plan if t.id not in completed_ids]
        return await self.execute_batch(pending)
