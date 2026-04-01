"""LangGraph state definition for the orchestrator pipeline."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from mas.schemas.results import AgentResult
from mas.schemas.tasks import Task


def _merge_results(existing: list[AgentResult], new: list[AgentResult]) -> list[AgentResult]:
    """Reducer: append new results to existing."""
    return existing + new


def _merge_tasks(existing: list[Task], new: list[Task]) -> list[Task]:
    """Reducer: merge tasks by ID (update existing, append new)."""
    by_id = {t.id: t for t in existing}
    for t in new:
        by_id[t.id] = t
    return list(by_id.values())


class PipelineState(TypedDict):
    """Full pipeline state flowing through the orchestrator graph."""

    # User request
    user_query: str
    user_context: dict[str, Any]

    # Planning
    plan: Annotated[list[Task], _merge_tasks]
    current_phase: str  # "planning" | "executing" | "reviewing" | "complete"

    # Execution
    pending_tasks: list[Task]
    active_tasks: list[Task]
    completed_task_ids: list[str]
    results: Annotated[list[AgentResult], _merge_results]

    # Messages (LangGraph native)
    messages: Annotated[list[BaseMessage], add_messages]

    # Progress tracking (Magentic-One inspired)
    task_ledger: str  # What needs to be done
    progress_ledger: str  # Self-reflection on progress

    # Control
    iteration: int
    max_iterations: int
    should_replan: bool
    final_output: dict[str, Any]
