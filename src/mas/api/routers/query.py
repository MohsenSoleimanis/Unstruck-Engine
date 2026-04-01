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

from mas.api.events import broadcaster
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
    SSE streaming — yields real-time events as the pipeline executes.

    Event types: phase, plan, task_start, task_complete, token, cost, done, error
    """
    pipeline = _get_pipeline()

    async def event_generator():
        stream_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()

        try:
            initial_state = {
                "user_query": request.query,
                "user_context": request.context,
                "plan": [],
                "current_phase": "planning",
                "pending_tasks": [],
                "active_tasks": [],
                "completed_task_ids": [],
                "results": [],
                "messages": [],
                "task_ledger": "",
                "progress_ledger": "",
                "iteration": 0,
                "max_iterations": request.max_iterations,
                "should_replan": False,
                "final_output": {},
            }

            prev_phase = ""
            prev_result_count = 0

            async for state_chunk in pipeline._compiled.astream(initial_state):
                state = state_chunk if isinstance(state_chunk, dict) else {}

                # Phase change
                phase = state.get("current_phase", "")
                if phase and phase != prev_phase:
                    yield _sse("phase", {"phase": phase, "iteration": state.get("iteration", 0)})
                    prev_phase = phase

                # New plan
                plan = state.get("plan", [])
                if plan and prev_phase == "executing":
                    tasks = [{"id": t.id, "agent_type": t.agent_type, "instruction": t.instruction[:80]} for t in plan]
                    yield _sse("plan", {"tasks": tasks})

                # New results
                results = state.get("results", [])
                if len(results) > prev_result_count:
                    for r in results[prev_result_count:]:
                        yield _sse("task_complete", {
                            "task_id": r.task_id,
                            "agent_type": r.agent_type,
                            "status": r.status.value,
                            "duration_ms": r.duration_ms,
                            "tokens": r.token_usage,
                            "cost_usd": r.cost_usd,
                        })
                    prev_result_count = len(results)

                # Final output
                final = state.get("final_output", {})
                if final:
                    duration_ms = int((time.perf_counter() - start) * 1000)
                    yield _sse("done", {
                        "output": final,
                        "cost": pipeline.cost_tracker.get_summary(),
                        "duration_ms": duration_ms,
                    })

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
