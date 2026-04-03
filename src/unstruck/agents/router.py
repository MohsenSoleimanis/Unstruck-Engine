"""Agent router — dispatches tasks to agents respecting dependencies.

The orchestrator's delegate step calls router.execute_plan().
The router:
  1. Resolves dependencies (which tasks can run now?)
  2. Runs ready tasks in parallel (asyncio.gather)
  3. Passes results back
  4. Repeats until all tasks are done or deadlocked

No ad-hoc context flattening. The Context Engine handles all context
management. Each agent reads what it needs from task.context.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from unstruck.agents.base import BaseAgent
from unstruck.agents.registry import AgentRegistry
from unstruck.context import ContextEngine
from unstruck.schemas import AgentResult, ResultStatus, Task, TaskStatus

logger = structlog.get_logger()


class Router:
    """Dispatches tasks to agents, manages parallel execution."""

    def __init__(
        self,
        registry: AgentRegistry,
        context_engine: ContextEngine,
        max_concurrent: int = 5,
    ) -> None:
        self._registry = registry
        self._context_engine = context_engine
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._agents: dict[str, BaseAgent] = {}

    def _get_agent(self, agent_type: str) -> BaseAgent:
        """Get or create an agent instance."""
        if agent_type not in self._agents:
            self._agents[agent_type] = self._registry.create(
                agent_type, self._context_engine
            )
        return self._agents[agent_type]

    async def execute_task(self, task: Task) -> AgentResult:
        """Execute a single task via the appropriate agent."""
        async with self._semaphore:
            try:
                agent = self._get_agent(task.agent_type)
                task.status = TaskStatus.RUNNING
                result = await agent.run(task)

                if result.status == ResultStatus.SUCCESS:
                    task.status = TaskStatus.COMPLETED
                else:
                    task.status = TaskStatus.FAILED

                return result

            except Exception as e:
                task.status = TaskStatus.FAILED
                logger.error("router.task_error", task_id=task.id, error=str(e))

                # Errors-as-feedback: structured error, not crash
                return AgentResult(
                    task_id=task.id,
                    agent_id="router",
                    agent_type=task.agent_type,
                    status=ResultStatus.FAILED,
                    errors=[str(e)],
                )

    async def execute_plan(
        self,
        plan: list[Task],
        existing_results: list[AgentResult] | None = None,
    ) -> list[AgentResult]:
        """
        Execute a full plan respecting dependencies.

        Tasks with satisfied dependencies run in parallel.
        Results accumulate. Deadlocks are detected and reported.
        """
        results: list[AgentResult] = []
        completed_ids: set[str] = set()

        # Carry forward existing completed tasks (from replan)
        if existing_results:
            for r in existing_results:
                if r.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL):
                    completed_ids.add(r.task_id)

        # Only execute tasks that haven't been completed yet
        remaining = [t for t in plan if t.id not in completed_ids]

        while remaining:
            # Find tasks whose dependencies are all satisfied
            ready = [t for t in remaining if t.is_ready(completed_ids)]

            if not ready:
                # Deadlock — remaining tasks have unsatisfiable dependencies
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

            # Process results
            completed_in_batch: set[str] = set()
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
                completed_in_batch.add(task.id)

            # Remove completed tasks from remaining (O(n) not O(n²))
            remaining = [t for t in remaining if t.id not in completed_in_batch]

        return results
