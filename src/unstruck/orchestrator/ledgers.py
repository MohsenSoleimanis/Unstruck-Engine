"""Structured ledgers — the orchestrator's working memory.

TaskLedger: what needs to be done, what's done, what failed.
ProgressLedger: structured reflection at each review step.

These are queryable data structures the orchestrator reasons over —
not decorative strings. The LLM reads them to make decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from unstruck.schemas.result import AgentResult, ResultStatus
from unstruck.schemas.task import Task, TaskStatus


class LedgerEntry(BaseModel):
    """One task's record in the ledger."""

    task_id: str
    agent_type: str
    instruction: str
    status: TaskStatus = TaskStatus.PENDING
    result_summary: str = ""
    errors: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0


class TaskLedger(BaseModel):
    """
    Tracks all tasks — the orchestrator queries this to understand state.

    The strategize step creates entries. The delegate step updates them.
    The evaluate step reads them to assess progress.
    """

    entries: list[LedgerEntry] = Field(default_factory=list)

    def add_task(self, task: Task) -> None:
        self.entries.append(LedgerEntry(
            task_id=task.id,
            agent_type=task.agent_type,
            instruction=task.instruction,
        ))

    def record_result(self, result: AgentResult) -> None:
        """Update a task's entry with the result."""
        for entry in self.entries:
            if entry.task_id == result.task_id:
                if result.status == ResultStatus.SUCCESS:
                    entry.status = TaskStatus.COMPLETED
                    entry.result_summary = _summarize_output(result.output)
                elif result.status == ResultStatus.PARTIAL:
                    entry.status = TaskStatus.COMPLETED
                    entry.result_summary = _summarize_output(result.output) + " [partial]"
                else:
                    entry.status = TaskStatus.FAILED
                    entry.errors = result.errors
                entry.tokens_used = result.total_tokens
                entry.cost_usd = result.cost_usd
                entry.duration_ms = result.duration_ms
                return

    @property
    def pending(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.status == TaskStatus.PENDING]

    @property
    def completed(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.status == TaskStatus.COMPLETED]

    @property
    def failed(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.status == TaskStatus.FAILED]

    @property
    def completion_rate(self) -> float:
        if not self.entries:
            return 0.0
        return len(self.completed) / len(self.entries)

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.tokens_used for e in self.entries)

    def for_prompt(self) -> str:
        """Format for inclusion in LLM prompts."""
        lines = []
        icons = {
            TaskStatus.PENDING: "⬜",
            TaskStatus.RUNNING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.SKIPPED: "⏭️",
        }
        for e in self.entries:
            icon = icons.get(e.status, "?")
            line = f"{icon} [{e.agent_type}] {e.instruction[:60]}"
            if e.result_summary:
                line += f" → {e.result_summary[:80]}"
            if e.errors:
                line += f" → ERROR: {', '.join(e.errors[:2])}"
            lines.append(line)
        return "\n".join(lines)


class Reflection(BaseModel):
    """Structured self-reflection at one review step."""

    iteration: int
    completed_count: int = 0
    failed_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    completeness: float = 0.0
    quality: float = 0.0
    issues: list[str] = Field(default_factory=list)
    should_change_strategy: bool = False
    strategy_change: str = ""
    recommendation: str = "synthesize"  # synthesize | replan | ask_user | abort
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProgressLedger(BaseModel):
    """
    Tracks reflection across iterations.

    The evaluate step adds reflections. The decide step reads them
    to determine: done, replan, ask user, or abort.
    """

    reflections: list[Reflection] = Field(default_factory=list)

    def add(self, reflection: Reflection) -> None:
        self.reflections.append(reflection)

    @property
    def latest(self) -> Reflection | None:
        return self.reflections[-1] if self.reflections else None

    @property
    def is_making_progress(self) -> bool:
        """Compare last two iterations — are we improving?"""
        if len(self.reflections) < 2:
            return True
        curr = self.reflections[-1]
        prev = self.reflections[-2]
        return curr.completed_count > prev.completed_count or curr.failed_count < prev.failed_count

    def for_prompt(self) -> str:
        """Format for inclusion in LLM prompts."""
        if not self.reflections:
            return "No iterations completed yet."
        lines = []
        for r in self.reflections:
            lines.append(
                f"Iteration {r.iteration}: {r.completed_count} done, "
                f"{r.failed_count} failed, {r.total_tokens} tokens, "
                f"${r.total_cost:.4f}"
            )
            for issue in r.issues:
                lines.append(f"  → {issue}")
            if r.should_change_strategy:
                lines.append(f"  ⚠️ Strategy change: {r.strategy_change}")
            lines.append(f"  Recommendation: {r.recommendation}")
        return "\n".join(lines)


def _summarize_output(output: dict[str, Any], max_len: int = 150) -> str:
    """Create a brief summary of an agent's output for the ledger."""
    if not output:
        return "completed"

    # Try common keys first
    for key in ("answer", "response", "summary", "result"):
        if key in output:
            text = str(output[key])
            return text[:max_len] + ("..." if len(text) > max_len else "")

    # Fallback: first value
    first_val = str(next(iter(output.values())))
    return first_val[:max_len] + ("..." if len(first_val) > max_len else "")
