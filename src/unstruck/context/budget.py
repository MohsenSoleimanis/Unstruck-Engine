"""Token budget — enforces limits, not just tracks.

When the budget is exhausted, the Context Engine signals the Orchestrator
to trigger early synthesis. This prevents runaway cost.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class TokenBudget:
    """
    Tracks and enforces token budgets for a pipeline run.

    The budget is set from config/budgets.yaml. The Context Engine checks
    can_continue() before every LLM call and allocate() to get the max
    tokens an agent is allowed to use.
    """

    def __init__(
        self,
        total: int = 50000,
        per_agent: int = 8000,
        context_limit: int = 12000,
        synthesis_threshold: float = 0.85,
    ) -> None:
        self.total = total
        self.per_agent = per_agent
        self.context_limit = context_limit
        self.synthesis_threshold = synthesis_threshold
        self._consumed: dict[str, int] = {}  # agent_id → tokens used
        self._total_consumed: int = 0

    def record(self, agent_id: str, tokens: int) -> None:
        """Record tokens consumed by an agent."""
        self._consumed[agent_id] = self._consumed.get(agent_id, 0) + tokens
        self._total_consumed += tokens

    @property
    def consumed(self) -> int:
        return self._total_consumed

    @property
    def remaining(self) -> int:
        return max(0, self.total - self._total_consumed)

    @property
    def utilization(self) -> float:
        if self.total == 0:
            return 0.0
        return self._total_consumed / self.total

    def can_continue(self) -> bool:
        """False when budget utilization exceeds threshold → trigger early synthesis."""
        return self.utilization < self.synthesis_threshold

    def allocate(self, agent_id: str) -> int:
        """Return the max tokens this agent call should use."""
        return min(self.per_agent, self.remaining)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "consumed": self._total_consumed,
            "remaining": self.remaining,
            "utilization": round(self.utilization, 3),
            "by_agent": dict(self._consumed),
        }

    @classmethod
    def from_config(cls, config_budgets: dict[str, Any]) -> TokenBudget:
        """Create from config/budgets.yaml tokens section."""
        return cls(
            total=config_budgets.get("total_budget", 50000),
            per_agent=config_budgets.get("per_agent_budget", 8000),
            context_limit=config_budgets.get("context_budget", 12000),
            synthesis_threshold=config_budgets.get("synthesis_threshold", 0.85),
        )
