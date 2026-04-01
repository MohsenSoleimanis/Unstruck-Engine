"""Base agent interface — all agents inherit from this."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel

from mas.llmops.cost_tracker import CostTracker
from mas.memory.local import LocalMemory
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Contract every agent must fulfill.

    Generalizes your protocol-engine pattern:
      - Receives a Task with instruction + context
      - Uses an LLM + tools to produce structured output
      - Returns an AgentResult with token/cost tracking
    """

    agent_type: str = "base"
    description: str = "Base agent"
    version: str = "0.1.0"

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        agent_id: str | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.agent_id = agent_id or f"{self.agent_type}_{uuid.uuid4().hex[:6]}"
        self.llm = llm
        self.cost_tracker = cost_tracker
        self.local_memory = LocalMemory(namespace=self.agent_id)
        self.logger = logger.bind(agent_id=self.agent_id, agent_type=self.agent_type)

    @abstractmethod
    async def execute(self, task: Task) -> AgentResult:
        """Run the agent's core logic on a task. Must be implemented by subclasses."""

    async def run(self, task: Task) -> AgentResult:
        """Entry point — wraps execute() with logging, timing, and cost tracking."""
        self.logger.info("agent.start", task_id=task.id, instruction=task.instruction[:100])
        start = time.perf_counter()

        try:
            result = await self.execute(task)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result.duration_ms = elapsed_ms

            if self.cost_tracker:
                self.cost_tracker.record(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task_id=task.id,
                    token_usage=result.token_usage,
                    cost_usd=result.cost_usd,
                    model=self.llm.model_name if hasattr(self.llm, "model_name") else "unknown",
                )

            self.logger.info(
                "agent.done",
                task_id=task.id,
                status=result.status,
                duration_ms=elapsed_ms,
                tokens=result.token_usage,
            )
            return result

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self.logger.error("agent.error", task_id=task.id, error=str(e))
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[str(e)],
                duration_ms=elapsed_ms,
            )

    def get_capabilities(self) -> dict[str, Any]:
        """Return agent card (A2A-inspired) for dynamic discovery."""
        return {
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "description": self.description,
            "version": self.version,
        }
