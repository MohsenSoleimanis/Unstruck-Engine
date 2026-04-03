"""Base agent contract — what every agent must implement.

Every agent:
  - Receives a Task with instruction + context
  - Goes through Context Engine for LLM calls (never calls LLM directly)
  - Returns an AgentResult with output, token usage, and cost
  - Errors are returned as results, not raised as exceptions
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from unstruck.context import ContextEngine, ContextEngineResult
from unstruck.schemas import AgentResult, ResultStatus, Task

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Contract every agent must fulfill.

    Agents are stateless — they receive a Task, process it, and return a result.
    All state is external (Context Engine, Memory Layer, session).
    All LLM calls go through the Context Engine.
    """

    agent_type: str = "base"

    def __init__(self, agent_id: str, context_engine: ContextEngine) -> None:
        self.agent_id = agent_id
        self.context_engine = context_engine
        self._logger = logger.bind(agent_id=agent_id, agent_type=self.agent_type)

    @abstractmethod
    async def execute(self, task: Task) -> AgentResult:
        """
        Core logic — implemented by each agent.

        The agent should:
          1. Read what it needs from task.instruction and task.context
          2. Use self.llm_call() for LLM interactions (goes through Context Engine)
          3. Return an AgentResult with structured output

        Errors should be caught and returned as AgentResult(status=FAILED),
        NOT raised as exceptions.
        """

    async def run(self, task: Task) -> AgentResult:
        """
        Entry point — wraps execute() with timing, logging, and error handling.

        The orchestrator's delegate step calls this, not execute() directly.
        """
        self._logger.info("agent.start", task_id=task.id, instruction=task.instruction[:80])
        start = time.perf_counter()

        try:
            result = await self.execute(task)
            result.duration_ms = int((time.perf_counter() - start) * 1000)
            self._logger.info(
                "agent.done",
                task_id=task.id,
                status=result.status.value,
                duration_ms=result.duration_ms,
                tokens=result.total_tokens,
            )
            return result

        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self._logger.error("agent.error", task_id=task.id, error=str(e))

            # Errors-as-feedback: return structured error, don't crash
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[str(e)],
                duration_ms=duration_ms,
            )

    async def llm_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
        model_name: str = "",
    ) -> ContextEngineResult:
        """
        Make an LLM call through the Context Engine.

        This is how agents interact with LLMs. Never call llm.ainvoke() directly.
        The Context Engine handles budget, truncation, hooks, cost tracking.
        """
        from unstruck.config import get_config
        config = get_config()

        if not model_name:
            agent_config = config.agents.get(self.agent_type, {})
            tier = agent_config.get("model_tier", "worker")
            model_name = config.get_model_tier(tier)["primary"]

        # Get the LLM instance for this model
        llm = self._get_llm(model_name)

        return await self.context_engine.call(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            agent_id=self.agent_id,
            model_name=model_name,
        )

    def _get_llm(self, model_name: str):
        """Get an LLM instance by model name. Uses LangChain."""
        if "claude" in model_name or "anthropic" in model_name:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, temperature=0)
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=0)
