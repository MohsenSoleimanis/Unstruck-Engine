"""Task router — dispatches tasks to agents, manages parallel execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from langchain_core.language_models import BaseChatModel

from mas.agents.base import BaseAgent
from mas.agents.registry import AgentRegistry
from mas.llmops.cost_tracker import CostTracker
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task, TaskStatus

if TYPE_CHECKING:
    from mas.a2a.bus import MessageBus
    from mas.memory.knowledge_graph import KnowledgeGraph
    from mas.memory.shared import SharedMemory
    from mas.tools.mcp_client import MCPToolClient

logger = structlog.get_logger()


class Router:
    """
    Routes tasks to agents and manages execution.

    Passes shared infrastructure (memory, KG, bus, MCP) to every agent
    so they can communicate and share state during execution.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        worker_llm: BaseChatModel,
        *,
        cost_tracker: CostTracker | None = None,
        shared_memory: SharedMemory | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        message_bus: MessageBus | None = None,
        mcp_client: MCPToolClient | None = None,
        max_concurrent: int = 5,
    ) -> None:
        self.registry = registry
        self.worker_llm = worker_llm
        self.cost_tracker = cost_tracker
        self.shared_memory = shared_memory
        self.knowledge_graph = knowledge_graph
        self.message_bus = message_bus
        self.mcp_client = mcp_client
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._agents: dict[str, BaseAgent] = {}

    def _get_agent(self, agent_type: str) -> BaseAgent:
        """Get or create an agent instance with full infrastructure access."""
        if agent_type not in self._agents:
            self._agents[agent_type] = self.registry.create(
                agent_type,
                self.worker_llm,
                cost_tracker=self.cost_tracker,
                shared_memory=self.shared_memory,
                knowledge_graph=self.knowledge_graph,
                message_bus=self.message_bus,
                mcp_client=self.mcp_client,
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
            ready = [t for t in remaining if t.is_ready(completed_ids)]
            if not ready:
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

            batch_results = await asyncio.gather(
                *[self.execute_task(t) for t in ready],
                return_exceptions=True,
            )

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

            remaining = [t for t in remaining if t.id not in completed_in_batch]

        return results

    async def execute_plan(self, plan: list[Task], results_so_far: list[AgentResult] | None = None) -> list[AgentResult]:
        """Execute a full plan, carrying forward any existing results."""
        completed_ids = set()
        if results_so_far:
            completed_ids = {r.task_id for r in results_so_far if r.status == ResultStatus.SUCCESS}
        pending = [t for t in plan if t.id not in completed_ids]
        return await self.execute_batch(pending)
