"""Agent registry — discovers, registers, and instantiates agents.

Agents come from two sources:
  1. Built-in agents (registered via @registry.register decorator)
  2. Plugin agents (discovered from plugins/ directory via config)

The orchestrator queries the registry to find agents for tasks.
"""

from __future__ import annotations

from typing import Any

import structlog

from unstruck.agents.base import BaseAgent
from unstruck.context import ContextEngine

logger = structlog.get_logger()


class AgentRegistry:
    """
    Central registry for agent types.

    Provides:
      - Registration (decorator or manual)
      - Discovery (list all with capabilities)
      - Instantiation (create agent by type name)
    """

    def __init__(self) -> None:
        self._classes: dict[str, type[BaseAgent]] = {}
        self._configs: dict[str, dict[str, Any]] = {}

    def register(self, agent_cls: type[BaseAgent], config: dict[str, Any] | None = None) -> type[BaseAgent]:
        """
        Register an agent class. Can be used as a decorator:

            @registry.register
            class MyAgent(BaseAgent):
                agent_type = "my_agent"
        """
        name = agent_cls.agent_type
        self._classes[name] = agent_cls
        if config:
            self._configs[name] = config
        logger.debug("registry.registered", agent_type=name)
        return agent_cls

    def load_from_config(self, agents_config: dict[str, dict[str, Any]]) -> None:
        """Load agent configs from config/agents.yaml. Does NOT register classes — just stores configs."""
        for name, cfg in agents_config.items():
            self._configs[name] = cfg

    def create(
        self,
        agent_type: str,
        context_engine: ContextEngine,
        agent_id: str | None = None,
    ) -> BaseAgent:
        """Instantiate an agent by type name."""
        if agent_type not in self._classes:
            available = list(self._classes.keys())
            raise ValueError(f"Unknown agent type '{agent_type}'. Registered: {available}")

        cls = self._classes[agent_type]
        aid = agent_id or f"{agent_type}_{id(cls):x}"[:20]
        return cls(agent_id=aid, context_engine=context_engine)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents with their config (for the planner)."""
        agents = []
        for name, cls in self._classes.items():
            cfg = self._configs.get(name, {})
            agents.append({
                "agent_type": name,
                "description": cfg.get("description", getattr(cls, "description", "")),
                "version": cfg.get("version", getattr(cls, "version", "1.0.0")),
                "model_tier": cfg.get("model_tier", "worker"),
                "allowed_tools": cfg.get("allowed_tools", []),
                "trust_level": cfg.get("trust_level", "auto"),
            })
        return agents

    def has(self, agent_type: str) -> bool:
        return agent_type in self._classes

    def __contains__(self, agent_type: str) -> bool:
        return self.has(agent_type)

    @property
    def agent_count(self) -> int:
        return len(self._classes)
