"""Token and cost tracking for LLMOps."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# Approximate pricing per 1M tokens (input/output) as of 2026-04
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


class CostTracker:
    """Tracks token usage and estimated costs per agent, task, and session."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._records: list[dict[str, Any]] = []
        self._by_agent: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
        self._by_session: dict[str, float] = {"total_tokens": 0, "total_cost": 0.0}
        self._output_dir = output_dir

    def record(
        self,
        agent_id: str,
        agent_type: str,
        task_id: str,
        token_usage: dict[str, int],
        cost_usd: float = 0.0,
        model: str = "unknown",
    ) -> None:
        input_tokens = token_usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
        output_tokens = token_usage.get("output_tokens", token_usage.get("completion_tokens", 0))
        total = input_tokens + output_tokens

        if cost_usd == 0.0 and model in MODEL_PRICING:
            inp_price, out_price = MODEL_PRICING[model]
            cost_usd = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "agent_type": agent_type,
            "task_id": task_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "cost_usd": round(cost_usd, 6),
        }
        self._records.append(entry)
        self._by_agent[agent_type]["tokens"] += total
        self._by_agent[agent_type]["cost"] += cost_usd
        self._by_session["total_tokens"] += total
        self._by_session["total_cost"] += cost_usd

        logger.debug(
            "cost.recorded",
            agent=agent_type,
            model=model,
            tokens=total,
            cost=f"${cost_usd:.4f}",
        )

    def get_summary(self) -> dict[str, Any]:
        return {
            "session": {
                "total_tokens": int(self._by_session["total_tokens"]),
                "total_cost_usd": round(self._by_session["total_cost"], 4),
                "num_calls": len(self._records),
            },
            "by_agent": {k: dict(v) for k, v in self._by_agent.items()},
        }

    def export(self, path: Path | None = None) -> Path:
        out = path or (self._output_dir or Path("./output")) / "cost_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": self.get_summary(), "records": self._records}, indent=2))
        return out
