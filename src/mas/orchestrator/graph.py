"""LangGraph orchestrator — the central pipeline graph.

Implements:
  - Structured Task/Progress Ledgers (Magentic-One pattern)
  - Token budget management (early synthesis when exhausted)
  - Session-aware execution (skip re-ingestion on follow-ups)
  - LangGraph checkpointing for graceful degradation
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from mas.agents.registry import AgentRegistry
from mas.llmops.cost_tracker import CostTracker
from mas.llmops.monitoring import HealthMonitor
from mas.llmops.tracing import TracingManager
from mas.memory.knowledge_graph import KnowledgeGraph
from mas.memory.shared import SharedMemory
from mas.orchestrator.ledgers import (
    IterationReflection,
    LedgerTask,
    ProgressLedger,
    TaskLedger,
)
from mas.orchestrator.planner import Planner
from mas.orchestrator.router import Router
from mas.orchestrator.state import PipelineState
from mas.orchestrator.token_budget import TokenBudget
from mas.schemas.results import ResultStatus
from mas.session import Session

if TYPE_CHECKING:
    from mas.a2a.bus import MessageBus
    from mas.tools.mcp_client import MCPToolClient

logger = structlog.get_logger()

REVIEW_PROMPT = """You are reviewing the progress of a multi-agent pipeline.

Task Ledger:
{task_ledger}

Progress So Far:
{progress_ledger}

Token Budget: {tokens_consumed}/{tokens_total} ({utilization}% used)

Answer these questions:
1. Is the pipeline making progress toward answering the user's query?
2. Are there any failures that need a different approach?
3. Should I change strategy or continue with the current plan?
4. What should happen next?

Output JSON:
{{
  "insights": ["insight 1", "insight 2"],
  "should_change_strategy": true/false,
  "strategy_change": "description if changing",
  "next_actions": ["action 1"]
}}
"""


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

    Graph: plan → execute → review → (replan | synthesize)
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
        query = state["user_query"]
        context = state["user_context"]
        session_data = state.get("session_data", {})

        logger.info("orchestrator.planning", query=query[:100])

        if tracing:
            tracing.start_trace("orchestrator", {"query": query[:100]})

        # Session awareness: check if document already ingested
        ingested_docs = session_data.get("ingested_docs", {})
        file_path = context.get("file_path", "")
        if file_path and file_path in ingested_docs:
            # Same file being re-sent
            context["already_ingested"] = True
            context["doc_id"] = ingested_docs[file_path]
            logger.info("orchestrator.skip_ingestion", file_path=file_path)
        elif ingested_docs and not file_path:
            # Follow-up question — no file attached but session has documents
            context["already_ingested"] = True
            context["doc_id"] = list(ingested_docs.values())[0]
            logger.info("orchestrator.follow_up_detected", docs=list(ingested_docs.keys()))

        # Conversation history for context
        history = session_data.get("message_history", [])
        if history:
            context["conversation_history"] = history[-6:]  # Last 3 Q&A pairs

        tasks = await planner.plan(query, context)

        # Build structured Task Ledger
        task_ledger = TaskLedger()
        for t in tasks:
            task_ledger.add(t.id, t.agent_type, t.instruction)

        # Initialize token budget
        token_budget = TokenBudget()

        return {
            "plan": tasks,
            "pending_tasks": tasks,
            "current_phase": "executing",
            "task_ledger": task_ledger.model_dump(),
            "progress_ledger": ProgressLedger().model_dump(),
            "token_budget": token_budget.get_summary(),
            "messages": [AIMessage(content=f"Plan created with {len(tasks)} tasks.")],
        }

    # --- Node: Execute ---
    async def execute_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.executing", task_count=len(state.get("pending_tasks", [])))

        # Restore session context into router
        session_data = state.get("session_data", {})
        ctx_data = session_data.get("pipeline_context", {})
        if ctx_data:
            from mas.schemas.context import PipelineContext
            router.pipeline_context = PipelineContext.model_validate(ctx_data)

        results = await router.execute_plan(state["plan"], state["results"])

        # Update task ledger with results
        task_ledger = TaskLedger.model_validate(state["task_ledger"])
        token_budget = TokenBudget()

        for r in results:
            ledger_task = task_ledger.get_task(r.task_id)
            if ledger_task:
                total_tokens = sum(r.token_usage.values()) if r.token_usage else 0
                if r.status == ResultStatus.SUCCESS:
                    summary = json.dumps(r.output, default=str)[:200] if r.output else "completed"
                    ledger_task.mark_completed(summary, total_tokens, r.cost_usd, r.duration_ms)
                elif r.status == ResultStatus.FAILED:
                    ledger_task.mark_failed(r.errors)
                token_budget.record_usage(r.agent_type, total_tokens)

        # Store results in shared memory
        if shared_memory:
            for r in results:
                if r.status == ResultStatus.SUCCESS and r.output:
                    content = json.dumps(r.output, default=str)[:5000]
                    shared_memory.store_result(r.task_id, r.agent_type, content)

        # Store entities in knowledge graph
        if knowledge_graph:
            ctx = router.get_context()
            for entity in ctx.entities:
                knowledge_graph.add_entity(
                    entity.name.lower().replace(" ", "_"),
                    entity.type,
                    entity.properties,
                )
            for rel in ctx.relationships:
                knowledge_graph.add_relationship(rel.source, rel.target, rel.relation, rel.properties)

        completed_ids = [r.task_id for r in results if r.status == ResultStatus.SUCCESS]

        return {
            "results": results,
            "completed_task_ids": list(set(state.get("completed_task_ids", []) + completed_ids)),
            "current_phase": "reviewing",
            "task_ledger": task_ledger.model_dump(),
            "token_budget": token_budget.get_summary(),
            "iteration": state.get("iteration", 0) + 1,
        }

    # --- Node: Review (structured reflection) ---
    async def review_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.reviewing", iteration=state.get("iteration", 0))

        task_ledger = TaskLedger.model_validate(state["task_ledger"])
        progress_ledger = ProgressLedger.model_validate(state["progress_ledger"])
        budget = state.get("token_budget", {})

        has_failures = len(task_ledger.failed) > 0
        can_replan = state.get("iteration", 0) < state.get("max_iterations", 3)
        budget_ok = budget.get("utilization", 0) < 0.85

        if has_failures and can_replan and budget_ok:
            # LLM-powered reflection
            try:
                from mas.utils.parsing import extract_json
                response = await orchestrator_llm.ainvoke([
                    SystemMessage(content="You are a pipeline reviewer. Output valid JSON only."),
                    HumanMessage(content=REVIEW_PROMPT.format(
                        task_ledger=task_ledger.summary(),
                        progress_ledger=progress_ledger.summary(),
                        tokens_consumed=budget.get("consumed", 0),
                        tokens_total=budget.get("total_budget", 50000),
                        utilization=round(budget.get("utilization", 0) * 100, 1),
                    )),
                ])
                reflection_data = extract_json(response.content)
            except Exception:
                reflection_data = {
                    "insights": ["Review failed, continuing with synthesis"],
                    "should_change_strategy": False,
                }

            reflection = IterationReflection(
                iteration=state.get("iteration", 1),
                completed_count=len(task_ledger.completed),
                failed_count=len(task_ledger.failed),
                total_tokens=budget.get("consumed", 0),
                total_cost=sum(t.cost_usd for t in task_ledger.tasks),
                insights=reflection_data.get("insights", []),
                should_change_strategy=reflection_data.get("should_change_strategy", False),
                strategy_change=reflection_data.get("strategy_change", ""),
                next_actions=reflection_data.get("next_actions", []),
            )
            progress_ledger.add_reflection(reflection)

            should_replan = reflection.should_change_strategy or len(task_ledger.failed) > 0
        else:
            should_replan = False

        return {
            "should_replan": should_replan and can_replan and budget_ok,
            "current_phase": "executing" if should_replan else "complete",
            "progress_ledger": progress_ledger.model_dump(),
        }

    # --- Node: Replan ---
    async def replan_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.replanning", iteration=state.get("iteration", 0))

        task_ledger = TaskLedger.model_validate(state["task_ledger"])
        progress_ledger = ProgressLedger.model_validate(state["progress_ledger"])

        completed_summaries = [
            {"task_id": t.task_id, "agent": t.agent_type, "status": t.status, "result": t.result_summary}
            for t in task_ledger.tasks
        ]
        failed_ids = [t.task_id for t in task_ledger.failed]

        new_tasks = await planner.replan(
            original_query=state["user_query"],
            completed_results=completed_summaries,
            failed_tasks=failed_ids,
            progress_summary=progress_ledger.summary(),
        )

        # Add new tasks to ledger
        for t in new_tasks:
            task_ledger.add(t.id, t.agent_type, t.instruction)

        return {
            "plan": state["plan"] + new_tasks,
            "pending_tasks": new_tasks,
            "task_ledger": task_ledger.model_dump(),
            "messages": [AIMessage(content=f"Replanned: {len(new_tasks)} new tasks.")],
        }

    # --- Node: Synthesize ---
    async def synthesize_node(state: PipelineState) -> dict[str, Any]:
        logger.info("orchestrator.synthesizing")

        task_ledger = TaskLedger.model_validate(state["task_ledger"])
        progress_ledger = ProgressLedger.model_validate(state["progress_ledger"])
        budget = state.get("token_budget", {})
        ctx = router.get_context()

        successful = [r for r in state["results"] if r.status == ResultStatus.SUCCESS]
        outputs = {r.task_id: {"agent_type": r.agent_type, "output": r.output} for r in successful}

        if monitor:
            total_cost = sum(r.cost_usd for r in state["results"])
            monitor.record_pipeline_run(
                pipeline_id=state["user_query"][:50],
                duration_ms=sum(r.duration_ms for r in state["results"]),
                task_count=len(state["plan"]),
                success_count=len(successful),
                total_cost=total_cost,
            )

        # Build final output
        final = {
            "query": state["user_query"],
            "results": outputs,
            "analysis": ctx.analysis.model_dump() if ctx.analysis else None,
            "synthesis": ctx.synthesis,
            "rag_response": ctx.rag_response,
            "task_ledger": task_ledger.summary(),
            "progress_ledger": progress_ledger.summary(),
            "total_tasks": len(state["plan"]),
            "completed": len(successful),
            "failed": len(state["results"]) - len(successful),
            "iterations": state.get("iteration", 0),
            "token_budget": budget,
        }

        # Save pipeline context back to session
        session_update = {
            "pipeline_context": ctx.model_dump(),
            "ingested_docs": {},
        }
        if ctx.document and ctx.document.file_path:
            session_update["ingested_docs"] = {ctx.document.file_path: ctx.rag_doc_id or ctx.document.file_path}

        return {
            "final_output": final,
            "current_phase": "complete",
            "session_update": session_update,
            "messages": [AIMessage(content="Pipeline complete.")],
        }

    # --- Conditional edges ---
    def should_replan_or_finish(state: PipelineState) -> str:
        if state.get("should_replan"):
            return "replan"
        return "synthesize"

    # --- Build graph ---
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
