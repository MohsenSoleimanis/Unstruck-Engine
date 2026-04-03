"""Orchestrator Brain — the 7-step LangGraph graph.

Steps:
  1. Understand  — parse intent, detect follow-up
  2. Validate    — input safety, guardrails
  3. Strategize  — create task plan (TaskLedger)
  4. Delegate    — execute tasks via agents
  5. Evaluate    — assess quality (ProgressLedger)
  6. Decide      — synthesize / replan / ask_user / abort
  7. Learn       — update memory, save session

Graph: understand → validate → strategize → delegate → evaluate → decide
         ↑                                                    │
         └──────── replan ←──────────────────────────────────┘
                                                              │
                                              synthesize ←────┘ → END
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from unstruck.config import Config
from unstruck.context import ContextEngine, TokenBudget
from unstruck.hooks import HookEvent, HookManager
from unstruck.orchestrator.ledgers import (
    ProgressLedger,
    Reflection,
    TaskLedger,
)
from unstruck.orchestrator.state import PipelineState
from unstruck.schemas import AgentResult, ResultStatus, Task, TaskPriority

logger = structlog.get_logger()


def build_graph(
    config: Config,
    context_engine: ContextEngine,
    hooks: HookManager,
    orchestrator_llm: BaseChatModel,
    worker_llm: BaseChatModel,
    router: Any | None = None,
) -> StateGraph:
    """
    Build the orchestrator LangGraph.

    The graph is the structure. The intelligence is in the prompts
    (loaded from config) and the Context Engine (manages every LLM call).
    """

    orch_model = config.get_model_tier("orchestrator")["primary"]
    cheap_model = config.get_model_tier("cheap")["primary"]

    # ── Step 1: Understand ──────────────────────────────────────

    async def understand(state: PipelineState) -> dict[str, Any]:
        query = state["user_query"]
        session_data = state.get("session_data", {})

        logger.info("brain.understand", query=query[:80])

        prompt_template = config.load_prompt("orchestrator/understand.md")
        prompt = prompt_template.format(
            user_message=query,
            session_context=_format_session(session_data),
            conversation_history=_format_history(session_data.get("message_history", [])),
        )

        result = await context_engine.call(
            orchestrator_llm,
            system_prompt="You are the Understanding module. Output valid JSON only.",
            user_prompt=prompt,
            agent_id="orchestrator.understand",
            model_name=cheap_model,  # Cheap model for classification
        )

        understanding = _parse_json_safe(result.text, {
            "intent": "unknown",
            "task_type": "unknown",
            "is_follow_up": bool(session_data.get("message_history")),
            "requires_document": bool(state.get("user_context", {}).get("file_path")),
        })

        return {
            "understanding": understanding,
            "current_phase": "validating",
        }

    # ── Step 2: Validate ────────────────────────────────────────

    async def validate(state: PipelineState) -> dict[str, Any]:
        logger.info("brain.validate")

        # Guardrails run via hooks (registered by the guardrails module)
        # The hook system handles prompt injection, PII, ethical checks.
        # If a SessionStart hook blocks, we never get here.
        # Additional validation can be added via hooks without changing this code.

        return {"current_phase": "strategizing"}

    # ── Step 3: Strategize ──────────────────────────────────────

    async def strategize(state: PipelineState) -> dict[str, Any]:
        query = state["user_query"]
        understanding = state.get("understanding", {})
        session_data = state.get("session_data", {})
        user_context = state.get("user_context", {})

        logger.info("brain.strategize")

        # Session awareness
        ingested_docs = session_data.get("ingested_docs", {})
        file_path = user_context.get("file_path", "")
        is_follow_up = understanding.get("is_follow_up", False)

        session_context = ""
        if ingested_docs:
            session_context += f"Already ingested documents: {list(ingested_docs.keys())}\n"
        if is_follow_up and not file_path:
            session_context += "This is a follow-up question — skip document ingestion.\n"

        # Build agent list from config
        agent_list = "\n".join(
            f"  - {name}: {cfg['description']}"
            for name, cfg in config.agents.items()
        )

        prompt_template = config.load_prompt("orchestrator/strategize.md")
        budget_config = config.get_token_budgets()
        prompt = prompt_template.format(
            agent_list=agent_list,
            understanding=json.dumps(understanding, default=str),
            user_query=query,
            session_context=session_context,
            total_budget=budget_config.get("total_budget", 50000),
        )

        result = await context_engine.call(
            orchestrator_llm,
            system_prompt="You are the Strategist. Output a valid JSON array of task objects.",
            user_prompt=prompt,
            agent_id="orchestrator.strategize",
            model_name=orch_model,
        )

        tasks = _parse_task_list(result.text, user_context)

        # Build TaskLedger
        task_ledger = TaskLedger()
        for task in tasks:
            task_ledger.add_task(task)

        # Initialize budget from config
        budget = TokenBudget.from_config(budget_config)

        return {
            "plan": tasks,
            "task_ledger": task_ledger.model_dump(),
            "progress_ledger": ProgressLedger().model_dump(),
            "budget": budget.to_dict(),
            "current_phase": "delegating",
            "messages": [AIMessage(content=f"Plan: {len(tasks)} tasks created.")],
        }

    # ── Step 4: Delegate ────────────────────────────────────────

    async def delegate(state: PipelineState) -> dict[str, Any]:
        """Execute tasks via the Router → agents."""
        plan = state.get("plan", [])
        existing_results = state.get("results", [])

        logger.info("brain.delegate", task_count=len(plan))

        task_ledger = TaskLedger.model_validate(state.get("task_ledger", {}))

        if router is None:
            logger.warning("brain.delegate.no_router")
            return {
                "current_phase": "evaluating",
                "iteration": state.get("iteration", 0) + 1,
                "task_ledger": task_ledger.model_dump(),
            }

        # Execute tasks through the Router (parallel, dependency-aware)
        results = await router.execute_plan(plan, existing_results)

        # Update task ledger with results
        for r in results:
            task_ledger.record_result(r)

        # Update budget tracking
        completed_ids = state.get("completed_task_ids", [])
        new_completed = [r.task_id for r in results if r.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)]

        return {
            "results": results,
            "completed_task_ids": list(set(completed_ids + new_completed)),
            "current_phase": "evaluating",
            "iteration": state.get("iteration", 0) + 1,
            "task_ledger": task_ledger.model_dump(),
        }

    # ── Step 5: Evaluate ────────────────────────────────────────

    async def evaluate(state: PipelineState) -> dict[str, Any]:
        logger.info("brain.evaluate", iteration=state.get("iteration", 0))

        task_ledger = TaskLedger.model_validate(state.get("task_ledger", {}))
        progress_ledger = ProgressLedger.model_validate(state.get("progress_ledger", {}))
        budget = state.get("budget", {})
        results = state.get("results", [])

        # Summarize results for the evaluator
        result_summaries = []
        for r in results:
            result_summaries.append({
                "agent": r.agent_type,
                "status": r.status.value,
                "output_preview": str(r.output)[:200] if r.output else "",
            })

        prompt_template = config.load_prompt("orchestrator/evaluate.md")
        prompt = prompt_template.format(
            task_ledger=task_ledger.for_prompt(),
            agent_results=json.dumps(result_summaries, default=str),
            tokens_used=budget.get("consumed", 0),
            tokens_total=budget.get("total", 50000),
            utilization=round(budget.get("utilization", 0) * 100, 1),
        )

        result = await context_engine.call(
            orchestrator_llm,
            system_prompt="You are the Evaluator. Output valid JSON only.",
            user_prompt=prompt,
            agent_id="orchestrator.evaluate",
            model_name=cheap_model,  # Cheap model for evaluation
        )

        evaluation = _parse_json_safe(result.text, {
            "completeness": task_ledger.completion_rate,
            "quality": 0.5,
            "issues": [],
            "should_change_strategy": False,
            "recommendation": "synthesize" if task_ledger.completion_rate > 0.5 else "replan",
        })

        # Record reflection
        reflection = Reflection(
            iteration=state.get("iteration", 1),
            completed_count=len(task_ledger.completed),
            failed_count=len(task_ledger.failed),
            total_tokens=task_ledger.total_tokens,
            total_cost=task_ledger.total_cost,
            completeness=evaluation.get("completeness", 0),
            quality=evaluation.get("quality", 0),
            issues=evaluation.get("issues", []),
            should_change_strategy=evaluation.get("should_change_strategy", False),
            strategy_change=evaluation.get("strategy_change", ""),
            recommendation=evaluation.get("recommendation", "synthesize"),
        )
        progress_ledger.add(reflection)

        return {
            "evaluation": evaluation,
            "progress_ledger": progress_ledger.model_dump(),
            "current_phase": "deciding",
        }

    # ── Step 6: Decide ──────────────────────────────────────────

    async def decide(state: PipelineState) -> dict[str, Any]:
        evaluation = state.get("evaluation", {})
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 5)
        budget = state.get("budget", {})

        recommendation = evaluation.get("recommendation", "synthesize")
        budget_ok = budget.get("utilization", 0) < 0.85
        can_retry = iteration < max_iter

        if recommendation == "replan" and can_retry and budget_ok:
            decision = "replan"
        elif recommendation == "ask_user":
            decision = "ask_user"
        elif recommendation == "abort":
            decision = "abort"
        else:
            decision = "synthesize"

        logger.info("brain.decide", decision=decision, iteration=iteration)

        return {
            "decision": decision,
            "decision_message": evaluation.get("message_to_user", ""),
        }

    # ── Step 7a: Synthesize (final output) ──────────────────────

    async def synthesize(state: PipelineState) -> dict[str, Any]:
        logger.info("brain.synthesize")

        task_ledger = TaskLedger.model_validate(state.get("task_ledger", {}))
        progress_ledger = ProgressLedger.model_validate(state.get("progress_ledger", {}))
        results = state.get("results", [])

        # Gather all successful outputs
        outputs = {}
        for r in results:
            if r.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL):
                outputs[r.task_id] = {
                    "agent_type": r.agent_type,
                    "output": r.output,
                }

        final = {
            "query": state["user_query"],
            "results": outputs,
            "task_ledger": task_ledger.for_prompt(),
            "progress_ledger": progress_ledger.for_prompt(),
            "total_tasks": len(task_ledger.entries),
            "completed": len(task_ledger.completed),
            "failed": len(task_ledger.failed),
            "iterations": state.get("iteration", 0),
            "budget": state.get("budget", {}),
        }

        # Session update — save pipeline state for follow-ups
        session_update = {
            "task_ledger": task_ledger.model_dump(),
        }

        return {
            "final_output": final,
            "session_update": session_update,
            "current_phase": "complete",
            "messages": [AIMessage(content="Pipeline complete.")],
        }

    # ── Step 7b: Replan ─────────────────────────────────────────

    async def replan(state: PipelineState) -> dict[str, Any]:
        logger.info("brain.replan", iteration=state.get("iteration", 0))

        task_ledger = TaskLedger.model_validate(state.get("task_ledger", {}))
        progress_ledger = ProgressLedger.model_validate(state.get("progress_ledger", {}))

        prompt_template = config.load_prompt("orchestrator/strategize.md")
        agent_list = "\n".join(
            f"  - {name}: {cfg['description']}"
            for name, cfg in config.agents.items()
        )

        replan_context = (
            f"Previous plan results:\n{task_ledger.for_prompt()}\n\n"
            f"Progress:\n{progress_ledger.for_prompt()}\n\n"
            f"Generate ONLY the remaining tasks needed. Do not repeat completed work."
        )

        prompt = prompt_template.format(
            agent_list=agent_list,
            understanding=json.dumps(state.get("understanding", {})),
            user_query=state["user_query"],
            session_context=replan_context,
            total_budget=state.get("budget", {}).get("remaining", 25000),
        )

        result = await context_engine.call(
            orchestrator_llm,
            system_prompt="You are the Strategist replanning. Output a valid JSON array.",
            user_prompt=prompt,
            agent_id="orchestrator.replan",
            model_name=orch_model,
        )

        new_tasks = _parse_task_list(result.text, state.get("user_context", {}))
        for task in new_tasks:
            task_ledger.add_task(task)

        return {
            "plan": state.get("plan", []) + new_tasks,
            "task_ledger": task_ledger.model_dump(),
            "current_phase": "delegating",
            "messages": [AIMessage(content=f"Replanned: {len(new_tasks)} new tasks.")],
        }

    # ── Routing logic ───────────────────────────────────────────

    def route_decision(state: PipelineState) -> str:
        decision = state.get("decision", "synthesize")
        if decision == "replan":
            return "replan"
        # ask_user and abort both go to synthesize (with appropriate output)
        return "synthesize"

    # ── Build the graph ─────────────────────────────────────────

    graph = StateGraph(PipelineState)

    graph.add_node("understand", understand)
    graph.add_node("validate", validate)
    graph.add_node("strategize", strategize)
    graph.add_node("delegate", delegate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("decide", decide)
    graph.add_node("synthesize", synthesize)
    graph.add_node("replan", replan)

    graph.set_entry_point("understand")
    graph.add_edge("understand", "validate")
    graph.add_edge("validate", "strategize")
    graph.add_edge("strategize", "delegate")
    graph.add_edge("delegate", "evaluate")
    graph.add_edge("evaluate", "decide")
    graph.add_conditional_edges("decide", route_decision, {
        "replan": "replan",
        "synthesize": "synthesize",
    })
    graph.add_edge("replan", "delegate")
    graph.add_edge("synthesize", END)

    return graph


# ── Helpers ─────────────────────────────────────────────────────


def _parse_json_safe(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON from LLM output. Returns fallback on failure."""
    import re

    text = text.strip()

    # Try raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    logger.warning("brain.json_parse_failed", text_preview=text[:200])
    return fallback


def _parse_task_list(text: str, user_context: dict[str, Any]) -> list[Task]:
    """Parse LLM output into Task objects with proper dependency resolution."""
    raw = _parse_json_safe(text, [])
    if not isinstance(raw, list):
        raw = [raw] if isinstance(raw, dict) else []

    tasks: list[Task] = []
    raw_deps: list[list] = []

    for item in raw:
        if not isinstance(item, dict) or "agent_type" not in item:
            continue

        raw_deps.append(item.get("dependencies", []))

        # Merge user_context into every task (file_path, etc.)
        task_context = {**user_context, **item.get("context", {})}

        tasks.append(Task(
            agent_type=item["agent_type"],
            instruction=item.get("instruction", ""),
            context=task_context,
            dependencies=[],  # Resolved below
            priority=TaskPriority(item.get("priority", "medium")),
            token_budget=item.get("token_budget", 8000),
        ))

    # Resolve index-based dependencies to task IDs
    for task, deps in zip(tasks, raw_deps):
        resolved = []
        for dep in deps:
            if isinstance(dep, int) and 0 <= dep < len(tasks):
                resolved.append(tasks[dep].id)
            elif isinstance(dep, str):
                resolved.append(dep)
        task.dependencies = resolved

    return tasks


def _format_session(session_data: dict[str, Any]) -> str:
    """Format session data for prompts."""
    parts = []
    if session_data.get("ingested_docs"):
        parts.append(f"Ingested documents: {list(session_data['ingested_docs'].keys())}")
    if session_data.get("message_history"):
        parts.append(f"Previous messages: {len(session_data['message_history'])}")
    return "\n".join(parts) if parts else "New session — no prior context."


def _format_history(history: list[dict[str, str]], max_messages: int = 6) -> str:
    """Format conversation history for prompts."""
    if not history:
        return "No conversation history."
    recent = history[-max_messages:]
    lines = []
    for msg in recent:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")[:300]
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)
