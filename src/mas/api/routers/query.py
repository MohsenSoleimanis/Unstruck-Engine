"""Query router — synchronous and SSE streaming endpoints."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mas.utils.security import sanitize_filename

logger = structlog.get_logger()

router = APIRouter(prefix="/api", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    context: dict[str, Any] = {}
    max_iterations: int = Field(default=3, ge=1, le=10)
    conversation_id: str | None = None


class QueryResponse(BaseModel):
    status: str
    query: str
    results: dict[str, Any]
    cost: dict[str, Any]
    duration_ms: int


def _get_pipeline():
    from mas.api.server import app
    return app.state.pipeline


def _build_initial_state(query: str, context: dict, max_iterations: int) -> dict:
    return {
        "user_query": query,
        "user_context": context,
        "session_id": "",
        "session_data": {},
        "session_update": {},
        "plan": [],
        "current_phase": "planning",
        "pending_tasks": [],
        "active_tasks": [],
        "completed_task_ids": [],
        "results": [],
        "messages": [],
        "task_ledger": {},
        "progress_ledger": {},
        "token_budget": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_replan": False,
        "final_output": {},
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Synchronous query — waits for full pipeline completion."""
    pipeline = _get_pipeline()
    start = time.perf_counter()

    try:
        result = await pipeline.run(
            query=request.query,
            context=request.context,
            max_iterations=request.max_iterations,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)

        return QueryResponse(
            status="success",
            query=request.query,
            results=result,
            cost=pipeline.cost_tracker.get_summary(),
            duration_ms=duration_ms,
        )
    except Exception:
        logger.exception("query.failed", query=request.query[:100])
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    SSE streaming — yields events as the pipeline executes.

    LangGraph astream() yields {node_name: node_output} per node.
    We map each node to appropriate SSE events.
    """
    pipeline = _get_pipeline()

    async def event_generator():
        stream_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        session_id = request.conversation_id or stream_id

        try:
            # Use session-aware run path
            session = pipeline.session_manager.get(session_id)
            session.add_message("user", request.query)

            session_data = {
                "pipeline_context": session.pipeline_context.model_dump(),
                "ingested_docs": session.ingested_docs,
                "message_history": session.get_recent_history(6),
            }

            initial_state = _build_initial_state(request.query, request.context, request.max_iterations)
            initial_state["session_id"] = session_id
            initial_state["session_data"] = session_data
            initial_state["session_update"] = {}

            # astream with stream_mode="updates" yields {node_name: output_dict}
            async for chunk in pipeline._compiled.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if not isinstance(node_output, dict):
                        continue

                    if node_name == "plan":
                        yield _sse("phase", {"phase": "planning"})
                        plan = node_output.get("plan", [])
                        if plan:
                            tasks = []
                            for t in plan:
                                tasks.append({
                                    "id": t.id if hasattr(t, "id") else str(t),
                                    "agent_type": t.agent_type if hasattr(t, "agent_type") else "",
                                    "instruction": (t.instruction if hasattr(t, "instruction") else "")[:80],
                                })
                            yield _sse("plan", {"tasks": tasks})
                        yield _sse("phase", {"phase": "executing"})

                    elif node_name == "execute":
                        results = node_output.get("results", [])
                        for r in results:
                            yield _sse("task_complete", {
                                "task_id": r.task_id if hasattr(r, "task_id") else "",
                                "agent_type": r.agent_type if hasattr(r, "agent_type") else "",
                                "status": r.status.value if hasattr(r, "status") else "unknown",
                                "duration_ms": getattr(r, "duration_ms", 0),
                                "cost_usd": getattr(r, "cost_usd", 0),
                            })

                    elif node_name == "review":
                        yield _sse("phase", {"phase": "reviewing"})

                    elif node_name == "replan":
                        yield _sse("phase", {"phase": "replanning"})

                    elif node_name == "synthesize":
                        yield _sse("phase", {"phase": "complete"})
                        final = node_output.get("final_output", {})
                        duration_ms = int((time.perf_counter() - start) * 1000)
                        yield _sse("done", {
                            "output": final,
                            "cost": pipeline.cost_tracker.get_summary(),
                            "duration_ms": duration_ms,
                        })

                        # Save session after completion
                        session_update = node_output.get("session_update", {})
                        if session_update:
                            if "pipeline_context" in session_update:
                                from mas.schemas.context import PipelineContext
                                session.update_context(PipelineContext.model_validate(session_update["pipeline_context"]))
                            if "ingested_docs" in session_update:
                                for path, doc_id in session_update["ingested_docs"].items():
                                    session.register_document(path, doc_id)
                        analysis = final.get("analysis", {})
                        answer = (analysis.get("answer", "") if analysis else final.get("rag_response", ""))
                        if answer:
                            session.add_message("assistant", str(answer)[:2000])
                        session.save()

        except Exception as e:
            logger.exception("query_stream.failed", stream_id=stream_id)
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/query/file")
async def query_file(query: str, file: UploadFile = File(...)):
    """Upload a file and run a query against it."""
    from mas.config import get_config

    pipeline = _get_pipeline()
    config = get_config()

    safe_name = sanitize_filename(file.filename or "upload")
    upload_dir = config.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name

    content = await file.read()
    file_path.write_bytes(content)

    start = time.perf_counter()
    try:
        result = await pipeline.run(query=query, context={"file_path": str(file_path)})
        duration_ms = int((time.perf_counter() - start) * 1000)
        return {
            "status": "success",
            "query": query,
            "file": safe_name,
            "results": result,
            "cost": pipeline.cost_tracker.get_summary(),
            "duration_ms": duration_ms,
        }
    except Exception:
        logger.exception("query_file.failed", query=query[:100], file=safe_name)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
