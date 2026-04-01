"""Agent registry — factory + discovery for dynamically adding agents."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel

from mas.agents.base import BaseAgent
from mas.llmops.cost_tracker import CostTracker

logger = structlog.get_logger()


class AgentRegistry:
    """
    Central registry for agent types (factory + registry pattern from RAG-Anything).

    Register agent classes by type name. The orchestrator uses this to
    instantiate the right agent for each task.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseAgent]] = {}

    def register(self, agent_cls: type[BaseAgent]) -> type[BaseAgent]:
        """Register an agent class. Can be used as a decorator."""
        name = agent_cls.agent_type
        if name in self._registry:
            logger.warning("agent_registry.overwrite", agent_type=name)
        self._registry[name] = agent_cls
        logger.info("agent_registry.registered", agent_type=name)
        return agent_cls

    def create(
        self,
        agent_type: str,
        llm: BaseChatModel,
        *,
        cost_tracker: CostTracker | None = None,
        **kwargs: Any,
    ) -> BaseAgent:
        """Instantiate an agent by type name."""
        if agent_type not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown agent type '{agent_type}'. Available: {available}")
        cls = self._registry[agent_type]
        return cls(llm=llm, cost_tracker=cost_tracker, **kwargs)

    def list_agents(self) -> list[dict[str, str]]:
        """Return all registered agent capabilities (for orchestrator routing)."""
        return [
            {"agent_type": cls.agent_type, "description": cls.description, "version": cls.version}
            for cls in self._registry.values()
        ]

    def has(self, agent_type: str) -> bool:
        return agent_type in self._registry

    def __contains__(self, agent_type: str) -> bool:
        return self.has(agent_type)


# Global singleton
registry = AgentRegistry()
