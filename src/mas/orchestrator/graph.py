"""LangGraph orchestrator — the central pipeline graph."""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from mas.agents.registry import AgentRegistry
from mas.llmops.cost_tracker import CostTracker
from mas.llmops.monitoring import HealthMonitor
from mas.llmops.tracing import TracingManager
from mas.memory.knowledge_graph import KnowledgeGraph
from mas.memory.shared import SharedMemory
from mas.orchestrator.planner import Planner
from mas.orchestrator.router import Router
from mas.orchestrator.state import PipelineState
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task, TaskStatus

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mas.a2a.bus import MessageBus
    from mas.tools.mcp_client import MCPToolClient

logger = structlog.get_logger()


def _build_initial_state(user_query: str, context: dict[str, Any] | None = None) -> PipelineState:
    return PipelineState(
        user_query=user_query,
        user_context=context or {},
        plan=[],
        current_phase="planning",
        pending_tasks=[],
        active_tasks=[],
        completed_task_ids=[],
        results=[],
        messages=[HumanMessage(content=user_query)],
        task_ledger="",
        progress_ledger="",
        iteration=0,
        max_iterations=3,
        should_replan=False,
        final_output={},
    )


def build_orchestrator_graph(
    orchestrator_llm: BaseChatModel,
    worker_llm: BaseChatModel,
    registry: AgentRegistry,
    shared_memory: SharedMemory | None = None,
    knowledge_graph: KnowledgeGraph | None = None,
    cost_tracker: CostTracker | None = None,
    tracing: TracingManager | None = None,
    monitor: HealthMonitor | None = None,
    message_bus: MessageBus | None = None,
    mcp_client: MCPToolClient | None = None,
) -> StateGraph:
    """
    Build the main LangGraph orchestrator.

    Graph topology:
        plan -> route_and_execute -> review -> (replan | synthesize)
                     ^                  |
                     |                  | (if should_replan)
                     +------------------+

    Inspired by:
      - Magentic-One: Task Ledger + Progress Ledger
      - Your protocol-engine: parallel extraction -> cross-check -> consistency
      - AdaptOrch: adaptive orchestration structure
    """

    planner = Planner(orchestrator_llm, registry)
    router = Router(
        registry,
        worker_llm,
        cost_tracker=cost_tracker,
        shared_memory=shared_memory,
        knowledge_graph=knowledge_graph,
        message_bus=message_bus,
        mcp_client=mcp_client,
    )

    # --- Node: Plan ---
    async def plan_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.planning", query=state["user_query"][:100])

        if tracing:
            tracing.start_trace("orchestrator", {"query": state["user_query"][:100]})

        tasks = await planner.plan(state["user_query"], state["user_context"])

        task_ledger = "Tasks to complete:\n" + "\n".join(
            f"  [{t.id}] {t.agent_type}: {t.instruction[:80]}" for t in tasks
        )

        return {
            "plan": tasks,
            "pending_tasks": tasks,
            "current_phase": "executing",
            "task_ledger": task_ledger,
            "messages": [AIMessage(content=f"Plan created with {len(tasks)} tasks.")],
        }

    # --- Node: Execute ---
    async def execute_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.executing", task_count=len(state["pending_tasks"]))

        results = await router.execute_plan(state["plan"], state["results"])

        # Note: agents now auto-store results in shared memory via BaseAgent.run()

        completed_ids = [r.task_id for r in results if r.status == ResultStatus.SUCCESS]
        failed = [r for r in results if r.status == ResultStatus.FAILED]

        progress = (
            f"Iteration {state['iteration'] + 1}:\n"
            f"  Completed: {len(completed_ids)} tasks\n"
            f"  Failed: {len(failed)} tasks\n"
        )
        if failed:
            progress += "  Failures:\n" + "\n".join(
                f"    - {r.agent_type}: {r.errors}" for r in failed
            )

        return {
            "results": results,
            "completed_task_ids": state["completed_task_ids"] + completed_ids,
            "current_phase": "reviewing",
            "progress_ledger": state["progress_ledger"] + "\n" + progress,
            "iteration": state["iteration"] + 1,
        }

    # --- Node: Review ---
    async def review_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.reviewing", iteration=state["iteration"])

        total = len(state["plan"])
        completed = len(state["completed_task_ids"])
        failed = [r for r in state["results"] if r.status == ResultStatus.FAILED]

        # Decide if we need to replan
        should_replan = (
            len(failed) > 0
            and state["iteration"] < state["max_iterations"]
        )

        if not should_replan:
            return {
                "should_replan": False,
                "current_phase": "complete",
            }

        return {
            "should_replan": True,
            "current_phase": "executing",
        }

    # --- Node: Replan ---
    async def replan_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.replanning", iteration=state["iteration"])

        completed_summaries = [
            {"task_id": r.task_id, "agent": r.agent_type, "status": r.status.value}
            for r in state["results"]
        ]
        failed_ids = [r.task_id for r in state["results"] if r.status == ResultStatus.FAILED]

        new_tasks = await planner.replan(
            original_query=state["user_query"],
            completed_results=completed_summaries,
            failed_tasks=failed_ids,
            progress_summary=state["progress_ledger"],
        )

        return {
            "plan": state["plan"] + new_tasks,
            "pending_tasks": new_tasks,
            "messages": [AIMessage(content=f"Replanned: {len(new_tasks)} new tasks.")],
        }

    # --- Node: Synthesize ---
    async def synthesize_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.synthesizing")

        successful = [r for r in state["results"] if r.status == ResultStatus.SUCCESS]
        outputs = {r.agent_type: r.output for r in successful}

        # Record pipeline metrics
        if monitor:
            total_cost = sum(r.cost_usd for r in state["results"])
            monitor.record_pipeline_run(
                pipeline_id=state["user_query"][:50],
                duration_ms=sum(r.duration_ms for r in state["results"]),
                task_count=len(state["plan"]),
                success_count=len(successful),
                total_cost=total_cost,
            )

        return {
            "final_output": {
                "query": state["user_query"],
                "results": outputs,
                "task_ledger": state["task_ledger"],
                "progress_ledger": state["progress_ledger"],
                "total_tasks": len(state["plan"]),
                "completed": len(successful),
                "failed": len(state["results"]) - len(successful),
                "iterations": state["iteration"],
            },
            "current_phase": "complete",
            "messages": [AIMessage(content="Pipeline complete.")],
        }

    # --- Conditional edges ---
    def should_replan_or_finish(state: PipelineState) -> str:
        if state["should_replan"]:
            return "replan"
        return "synthesize"

    # --- Build the graph ---
    graph = StateGraph(PipelineState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)
    graph.add_node("replan", replan_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", should_replan_or_finish, {"replan": "replan", "synthesize": "synthesize"})
    graph.add_edge("replan", "execute")
    graph.add_edge("synthesize", END)

    return graph
