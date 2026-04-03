"""Query endpoints — sync and SSE streaming."""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from unstruck.hooks import HookEvent

logger = structlog.get_logger()

router = APIRouter(prefix="/api", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    context: dict[str, Any] = {}
    max_iterations: int = Field(default=3, ge=1, le=10)
    conversation_id: str | None = None


@router.post("/query")
async def query_sync(request: Request, body: QueryRequest):
    """Synchronous query — waits for pipeline completion."""
    p = request.app.state.platform
    start = time.perf_counter()

    try:
        # Session
        session_id = body.conversation_id or "default"
        session = p.session_manager.get(session_id)
        session.add_message("user", body.query)

        # Fire SessionStart hook
        await p.hooks.fire(HookEvent.SESSION_START, {"session_id": session_id})

        # Build orchestrator state
        from unstruck.orchestrator.brain import build_graph
        graph = build_graph(
            config=p.config,
            context_engine=p.context_engine,
            hooks=p.hooks,
            orchestrator_llm=_get_orchestrator_llm(p),
            worker_llm=_get_worker_llm(p),
        )
        compiled = graph.compile()

        initial_state = _build_initial_state(body, session)
        result = await compiled.ainvoke(initial_state)

        # Save session
        final = result.get("final_output", {})
        session.update_state(result.get("session_update", {}))
        session.save()

        # Fire SessionEnd hook
        await p.hooks.fire(HookEvent.SESSION_END, {"session_id": session_id})

        duration_ms = int((time.perf_counter() - start) * 1000)
        return {
            "status": "success",
            "query": body.query,
            "results": final,
            "cost": p.cost_tracker.get_summary(),
            "duration_ms": duration_ms,
        }

    except Exception:
        logger.exception("query.failed", query=body.query[:100])
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest):
    """SSE streaming — yields events as the pipeline executes."""
    p = request.app.state.platform

    async def generate():
        start = time.perf_counter()
        session_id = body.conversation_id or "default"
        session = p.session_manager.get(session_id)
        session.add_message("user", body.query)

        try:
            await p.hooks.fire(HookEvent.SESSION_START, {"session_id": session_id})

            from unstruck.orchestrator.brain import build_graph
            graph = build_graph(
                config=p.config,
                context_engine=p.context_engine,
                hooks=p.hooks,
                orchestrator_llm=_get_orchestrator_llm(p),
                worker_llm=_get_worker_llm(p),
            )
            compiled = graph.compile()
            initial_state = _build_initial_state(body, session)

            # Stream via LangGraph astream (yields per-node updates)
            async for chunk in compiled.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if not isinstance(node_output, dict):
                        continue

                    if node_name == "understand":
                        yield _sse("phase", {"phase": "understanding"})

                    elif node_name == "validate":
                        yield _sse("phase", {"phase": "validating"})

                    elif node_name == "strategize":
                        yield _sse("phase", {"phase": "planning"})
                        plan = node_output.get("plan", [])
                        if plan:
                            tasks = [
                                {"id": getattr(t, "id", ""), "agent_type": getattr(t, "agent_type", ""), "instruction": getattr(t, "instruction", "")[:80]}
                                for t in plan
                            ]
                            yield _sse("plan", {"tasks": tasks})

                    elif node_name == "delegate":
                        yield _sse("phase", {"phase": "executing"})

                    elif node_name == "evaluate":
                        yield _sse("phase", {"phase": "evaluating"})
                        evaluation = node_output.get("evaluation", {})
                        if evaluation:
                            yield _sse("evaluation", evaluation)

                    elif node_name == "decide":
                        decision = node_output.get("decision", "")
                        yield _sse("decision", {"decision": decision})

                    elif node_name == "replan":
                        yield _sse("phase", {"phase": "replanning"})

                    elif node_name == "synthesize":
                        yield _sse("phase", {"phase": "complete"})
                        final = node_output.get("final_output", {})
                        duration_ms = int((time.perf_counter() - start) * 1000)
                        yield _sse("done", {
                            "output": final,
                            "cost": p.cost_tracker.get_summary(),
                            "duration_ms": duration_ms,
                        })

                        # Save session
                        session.update_state(node_output.get("session_update", {}))
                        session.save()

            await p.hooks.fire(HookEvent.SESSION_END, {"session_id": session_id})

        except Exception as e:
            logger.exception("query_stream.failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Helpers ─────────────────────────────────────────────────────

def _build_initial_state(body: QueryRequest, session) -> dict[str, Any]:
    session_data = session.to_dict()

    # Session awareness
    user_context = dict(body.context)
    ingested_docs = session_data.get("ingested_docs", {})
    file_path = user_context.get("file_path", "")

    if file_path and file_path in ingested_docs:
        user_context["already_ingested"] = True
    elif ingested_docs and not file_path:
        user_context["already_ingested"] = True

    return {
        "user_query": body.query,
        "user_context": user_context,
        "session_id": session.session_id,
        "session_data": session_data,
        "understanding": {},
        "plan": [],
        "results": [],
        "completed_task_ids": [],
        "task_ledger": {},
        "progress_ledger": {},
        "evaluation": {},
        "decision": "",
        "decision_message": "",
        "budget": {},
        "final_output": {},
        "session_update": {},
        "messages": [],
        "iteration": 0,
        "max_iterations": body.max_iterations,
        "current_phase": "starting",
    }


def _get_orchestrator_llm(platform):
    tier = platform.config.get_model_tier("orchestrator")
    model = tier["primary"]
    if "claude" in model:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0)


def _get_worker_llm(platform):
    tier = platform.config.get_model_tier("worker")
    model = tier["primary"]
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
