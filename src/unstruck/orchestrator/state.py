"""LangGraph state — what flows through the orchestrator graph.

Every field is typed. The ledgers are serialized dicts (Pydantic → dict)
because LangGraph TypedDict needs JSON-serializable values.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from unstruck.schemas.result import AgentResult
from unstruck.schemas.task import Task


def _merge_results(existing: list[AgentResult], new: list[AgentResult]) -> list[AgentResult]:
    return existing + new


def _merge_tasks(existing: list[Task], new: list[Task]) -> list[Task]:
    by_id = {t.id: t for t in existing}
    for t in new:
        by_id[t.id] = t
    return list(by_id.values())


class PipelineState(TypedDict):
    """Full state flowing through the orchestrator graph."""

    # --- User input ---
    user_query: str
    user_context: dict[str, Any]

    # --- Session ---
    session_id: str
    session_data: dict[str, Any]  # Loaded session state (docs, history, context)

    # --- Understanding (step 1) ---
    understanding: dict[str, Any]

    # --- Planning (step 3) ---
    plan: Annotated[list[Task], _merge_tasks]

    # --- Execution (step 4) ---
    results: Annotated[list[AgentResult], _merge_results]
    completed_task_ids: list[str]

    # --- Structured ledgers ---
    task_ledger: dict[str, Any]     # Serialized TaskLedger
    progress_ledger: dict[str, Any]  # Serialized ProgressLedger

    # --- Evaluation (step 5) ---
    evaluation: dict[str, Any]

    # --- Decision (step 6) ---
    decision: str  # "synthesize" | "replan" | "ask_user" | "abort"
    decision_message: str  # Message to user if ask_user

    # --- Token budget ---
    budget: dict[str, Any]  # Serialized TokenBudget.to_dict()

    # --- Output ---
    final_output: dict[str, Any]
    session_update: dict[str, Any]  # Changes to write back to session

    # --- Messages (LangGraph native) ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Control ---
    iteration: int
    max_iterations: int
    current_phase: str
