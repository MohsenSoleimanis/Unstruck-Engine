"""Structured Task and Progress Ledgers — not decorative strings.

Magentic-One pattern: the orchestrator reasons over these structures
at each iteration to decide what to do next, not just appends text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LedgerTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LedgerTask(BaseModel):
    """A single task entry in the Task Ledger."""

    task_id: str
    agent_type: str
    instruction: str
    status: LedgerTaskStatus = LedgerTaskStatus.PENDING
    result_summary: str = ""
    errors: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0

    def mark_completed(self, summary: str, tokens: int = 0, cost: float = 0.0, duration: int = 0) -> None:
        self.status = LedgerTaskStatus.COMPLETED
        self.result_summary = summary
        self.tokens_used = tokens
        self.cost_usd = cost
        self.duration_ms = duration

    def mark_failed(self, errors: list[str]) -> None:
        self.status = LedgerTaskStatus.FAILED
        self.errors = errors


class TaskLedger(BaseModel):
    """
    Structured task tracking — the orchestrator queries this to
    understand what has been done and what remains.
    """

    tasks: list[LedgerTask] = Field(default_factory=list)

    def add(self, task_id: str, agent_type: str, instruction: str) -> None:
        self.tasks.append(LedgerTask(task_id=task_id, agent_type=agent_type, instruction=instruction))

    @property
    def pending(self) -> list[LedgerTask]:
        return [t for t in self.tasks if t.status == LedgerTaskStatus.PENDING]

    @property
    def completed(self) -> list[LedgerTask]:
        return [t for t in self.tasks if t.status == LedgerTaskStatus.COMPLETED]

    @property
    def failed(self) -> list[LedgerTask]:
        return [t for t in self.tasks if t.status == LedgerTaskStatus.FAILED]

    @property
    def completion_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return len(self.completed) / len(self.tasks)

    def get_task(self, task_id: str) -> LedgerTask | None:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def summary(self) -> str:
        """Human-readable summary for LLM prompts."""
        lines = []
        for t in self.tasks:
            icon = {"pending": "⬜", "running": "🔄", "completed": "✅", "failed": "❌", "skipped": "⏭️"}
            lines.append(f"{icon.get(t.status, '?')} [{t.agent_type}] {t.instruction[:60]} → {t.status}")
            if t.result_summary:
                lines.append(f"    Result: {t.result_summary[:100]}")
            if t.errors:
                lines.append(f"    Errors: {', '.join(t.errors[:2])}")
        return "\n".join(lines)


class IterationReflection(BaseModel):
    """Structured self-reflection at each review step."""

    iteration: int
    completed_count: int = 0
    failed_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    insights: list[str] = Field(default_factory=list)
    should_change_strategy: bool = False
    strategy_change: str = ""
    next_actions: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProgressLedger(BaseModel):
    """
    Structured progress tracking — the orchestrator reasons over this
    to answer: "Am I making progress? Should I change strategy?"
    """

    iterations: list[IterationReflection] = Field(default_factory=list)

    def add_reflection(self, reflection: IterationReflection) -> None:
        self.iterations.append(reflection)

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    @property
    def is_making_progress(self) -> bool:
        """Check if the last iteration made progress vs the one before."""
        if len(self.iterations) < 2:
            return True
        curr = self.iterations[-1]
        prev = self.iterations[-2]
        return curr.completed_count > prev.completed_count or curr.failed_count < prev.failed_count

    def summary(self) -> str:
        """Human-readable summary for LLM prompts."""
        if not self.iterations:
            return "No iterations completed yet."
        lines = []
        for r in self.iterations:
            lines.append(
                f"Iteration {r.iteration}: {r.completed_count} completed, "
                f"{r.failed_count} failed, {r.total_tokens} tokens, ${r.total_cost:.4f}"
            )
            for insight in r.insights:
                lines.append(f"  → {insight}")
            if r.should_change_strategy:
                lines.append(f"  ⚠️ Strategy change: {r.strategy_change}")
        return "\n".join(lines)
