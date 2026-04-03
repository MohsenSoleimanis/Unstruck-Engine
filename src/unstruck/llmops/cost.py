"""Cost tracking with ceilings — halts execution when budget exceeded.

Registers as PostLLMCall hook to record every call. Registers as
PreLLMCall hook to BLOCK calls when the ceiling is hit.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog

from unstruck.hooks import HookAction, HookEvent, HookManager, HookResult

logger = structlog.get_logger()


class CostTracker:
    """
    Tracks costs and enforces ceilings.

    Reads pricing from config. Registered as hooks:
      - PreLLMCall: checks if cost ceiling would be exceeded
      - PostLLMCall: records cost
    """

    def __init__(self, pricing: dict[str, list[float]], ceiling_usd: float = 1.0) -> None:
        self._pricing = pricing  # model → [input_per_1M, output_per_1M]
        self._ceiling = ceiling_usd
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._calls: int = 0
        self._by_agent: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
        self._records: list[dict[str, Any]] = []

    def register_hooks(self, hooks: HookManager) -> None:
        hooks.register(HookEvent.PRE_LLM_CALL, self._check_ceiling)
        hooks.register(HookEvent.POST_LLM_CALL, self._record_cost)

    async def _check_ceiling(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Block LLM call if cost ceiling would be exceeded."""
        if self._total_cost >= self._ceiling:
            return HookResult.block(
                f"Cost ceiling reached: ${self._total_cost:.4f} >= ${self._ceiling:.2f}"
            )
        return HookResult.allow()

    async def _record_cost(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Record cost from an LLM call."""
        agent_id = context.get("agent_id", "unknown")
        model = context.get("model", "unknown")
        input_tokens = context.get("input_tokens", 0)
        output_tokens = context.get("output_tokens", 0)
        cost = context.get("cost_usd", 0.0)

        # Calculate cost from pricing if not provided
        if cost == 0.0 and model in self._pricing:
            inp_price, out_price = self._pricing[model]
            cost = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000

        total_tokens = input_tokens + output_tokens
        self._total_cost += cost
        self._total_tokens += total_tokens
        self._calls += 1
        self._by_agent[agent_id]["tokens"] += total_tokens
        self._by_agent[agent_id]["cost"] += cost

        self._records.append({
            "agent_id": agent_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

        return HookResult.allow()

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self._total_cost, 4),
            "total_tokens": self._total_tokens,
            "total_calls": self._calls,
            "ceiling_usd": self._ceiling,
            "by_agent": {k: dict(v) for k, v in self._by_agent.items()},
        }

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def ceiling_reached(self) -> bool:
        return self._total_cost >= self._ceiling
