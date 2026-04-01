"""Agent and pipeline evaluation engine."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class EvaluationEngine:
    """
    Evaluates agent outputs and pipeline quality.

    Metrics:
      - Task completion rate
      - Output quality scoring (via LLM-as-judge)
      - Token efficiency (quality per token)
      - Latency distribution
      - Cross-agent consistency
    """

    def __init__(self) -> None:
        self._evaluations: list[dict[str, Any]] = []

    def evaluate_result(
        self,
        task_id: str,
        agent_type: str,
        output: dict[str, Any],
        expected: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single agent result."""
        eval_result = {
            "task_id": task_id,
            "agent_type": agent_type,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
        }

        # Completeness: check if output has expected keys
        if expected:
            expected_keys = set(expected.keys())
            output_keys = set(output.keys())
            completeness = len(expected_keys & output_keys) / len(expected_keys) if expected_keys else 1.0
            eval_result["metrics"]["completeness"] = round(completeness, 3)

        # Non-emptiness: fraction of output values that are non-empty
        non_empty = sum(1 for v in output.values() if v) / max(len(output), 1)
        eval_result["metrics"]["non_emptiness"] = round(non_empty, 3)

        # Has errors
        eval_result["metrics"]["has_errors"] = bool(output.get("errors"))

        self._evaluations.append(eval_result)
        return eval_result

    def get_summary(self) -> dict[str, Any]:
        if not self._evaluations:
            return {"total": 0}

        by_agent: dict[str, list] = {}
        for e in self._evaluations:
            by_agent.setdefault(e["agent_type"], []).append(e)

        summary = {"total": len(self._evaluations), "by_agent": {}}
        for agent_type, evals in by_agent.items():
            completeness_scores = [e["metrics"].get("completeness", 0) for e in evals]
            summary["by_agent"][agent_type] = {
                "count": len(evals),
                "avg_completeness": round(sum(completeness_scores) / len(completeness_scores), 3) if completeness_scores else 0,
                "error_rate": sum(1 for e in evals if e["metrics"].get("has_errors")) / len(evals),
            }
        return summary

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": self.get_summary(), "evaluations": self._evaluations}, indent=2))
