"""Evaluation — measures system quality and accuracy.

Two types of evaluation:

1. ONLINE (per-run): Runs automatically after every pipeline execution.
   Checks completeness, grounding, confidence. Registered as hooks.

2. OFFLINE (batch): Runs test cases against the system, compares outputs
   to expected results. Used for development, regression testing, and
   accuracy measurement before deployment.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from unstruck.hooks import HookEvent, HookManager, HookResult

logger = structlog.get_logger()


class OnlineEvaluator:
    """
    Evaluates every pipeline run automatically.

    Registers as PostLLMCall hook to score individual LLM outputs.
    Also provides evaluate_run() for end-of-pipeline assessment.

    Metrics:
      - Completeness: did the output address the question?
      - Grounding: does the output cite sources?
      - Confidence: how confident is the system?
      - Token efficiency: quality per token spent
    """

    def __init__(self) -> None:
        self._scores: deque[dict[str, Any]] = deque(maxlen=1000)

    def register_hooks(self, hooks: HookManager) -> None:
        hooks.register(HookEvent.POST_LLM_CALL, self._score_output)

    async def _score_output(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Quick heuristic scoring on every LLM output."""
        response = context.get("response", "")

        score = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": context.get("agent_id", ""),
            "has_content": bool(response.strip()),
            "length": len(response),
            "has_json": "{" in response,
            "tokens": context.get("input_tokens", 0) + context.get("output_tokens", 0),
        }
        self._scores.append(score)
        return HookResult.allow()

    def evaluate_run(self, output: dict[str, Any], query: str) -> dict[str, Any]:
        """
        Evaluate a complete pipeline run.

        Returns quality metrics that the orchestrator's Evaluate step uses.
        """
        results = output.get("results", {})

        # Completeness: did we get results?
        has_results = len(results) > 0
        has_answer = any(
            "answer" in r.get("output", {}) or "response" in r.get("output", {})
            for r in results.values()
            if isinstance(r, dict)
        )

        # Grounding: do results cite sources?
        has_citations = any(
            "citations" in r.get("output", {})
            for r in results.values()
            if isinstance(r, dict)
        )

        # Task completion
        total_tasks = output.get("total_tasks", 0)
        completed = output.get("completed", 0)
        completion_rate = completed / max(total_tasks, 1)

        return {
            "completeness": 1.0 if has_answer else (0.5 if has_results else 0.0),
            "grounding": 1.0 if has_citations else 0.5,
            "task_completion_rate": round(completion_rate, 2),
            "total_tasks": total_tasks,
            "completed": completed,
            "failed": output.get("failed", 0),
        }

    def get_recent_scores(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self._scores)[-limit:]


class OfflineEvaluator:
    """
    Batch evaluation — run test cases and measure accuracy.

    Test cases are JSON files:
    [
      {
        "query": "What is the study design?",
        "file": "protocol.pdf",
        "expected": {
          "must_contain": ["Phase III", "randomized", "double-blind"],
          "must_not_contain": ["Phase I"],
          "expected_confidence": "high"
        }
      }
    ]
    """

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []

    def load_test_cases(self, path: Path) -> list[dict[str, Any]]:
        """Load test cases from a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Test case file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def evaluate_case(self, test_case: dict[str, Any], actual_output: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a single test case against actual output.

        Returns pass/fail with details.
        """
        expected = test_case.get("expected", {})
        query = test_case.get("query", "")

        # Extract the actual answer text
        answer = self._extract_answer(actual_output)

        result = {
            "query": query,
            "passed": True,
            "checks": [],
        }

        # Check must_contain
        for term in expected.get("must_contain", []):
            found = term.lower() in answer.lower()
            result["checks"].append({
                "type": "must_contain",
                "term": term,
                "found": found,
            })
            if not found:
                result["passed"] = False

        # Check must_not_contain
        for term in expected.get("must_not_contain", []):
            found = term.lower() in answer.lower()
            result["checks"].append({
                "type": "must_not_contain",
                "term": term,
                "found": found,
            })
            if found:
                result["passed"] = False

        self._results.append(result)
        return result

    def get_summary(self) -> dict[str, Any]:
        """Get overall accuracy metrics."""
        if not self._results:
            return {"total": 0, "passed": 0, "failed": 0, "accuracy": 0.0}

        passed = sum(1 for r in self._results if r["passed"])
        return {
            "total": len(self._results),
            "passed": passed,
            "failed": len(self._results) - passed,
            "accuracy": round(passed / len(self._results), 3),
        }

    def _extract_answer(self, output: dict[str, Any]) -> str:
        """Extract the answer text from pipeline output."""
        results = output.get("results", {})
        for task_data in results.values():
            if isinstance(task_data, dict):
                out = task_data.get("output", {})
                if isinstance(out, dict):
                    for key in ("answer", "response", "summary"):
                        if key in out:
                            val = out[key]
                            return val if isinstance(val, str) else json.dumps(val)
        return ""
