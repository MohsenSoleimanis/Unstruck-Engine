"""LangGraph state definition for the orchestrator pipeline."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from mas.schemas.results import AgentResult
from mas.schemas.tasks import Task


def _merge_results(existing: list[AgentResult], new: list[AgentResult]) -> list[AgentResult]:
    return existing + new


def _merge_tasks(existing: list[Task], new: list[Task]) -> list[Task]:
    by_id = {t.id: t for t in existing}
    for t in new:
        by_id[t.id] = t
    return list(by_id.values())


class PipelineState(TypedDict):
    """Full pipeline state — typed, structured, session-aware."""

    # --- User request ---
    user_query: str
    user_context: dict[str, Any]

    # --- Session (persists across messages) ---
    session_id: str
    session_data: dict[str, Any]  # serialized Session state
    session_update: dict[str, Any]  # changes to write back to session

    # --- Planning ---
    plan: Annotated[list[Task], _merge_tasks]
    current_phase: str

    # --- Execution ---
    pending_tasks: list[Task]
    active_tasks: list[Task]
    completed_task_ids: list[str]
    results: Annotated[list[AgentResult], _merge_results]

    # --- Structured ledgers (not strings) ---
    task_ledger: dict[str, Any]  # serialized TaskLedger
    progress_ledger: dict[str, Any]  # serialized ProgressLedger

    # --- Token budget ---
    token_budget: dict[str, Any]  # serialized TokenBudget summary

    # --- Messages ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Control ---
    iteration: int
    max_iterations: int
    should_replan: bool
    final_output: dict[str, Any]
