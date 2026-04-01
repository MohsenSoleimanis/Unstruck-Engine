"""FastAPI server — production API for the multi-agent system."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mas.config import get_config
from mas.pipeline import MASPipeline
from mas.utils.security import sanitize_filename

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline once at startup, clean up on shutdown."""
    app.state.pipeline = MASPipeline()
    yield


app = FastAPI(
    title="Multi-Agent System API",
    description="Production-scale, data-agnostic multi-agent orchestration system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — configurable, NOT allow-all in production
_config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_pipeline() -> MASPipeline:
    return app.state.pipeline


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    context: dict[str, Any] = {}
    max_iterations: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    status: str
    query: str
    results: dict[str, Any]
    cost: dict[str, Any]
    duration_ms: int


# --- Endpoints ---


@app.get("/health")
async def health():
    pipeline = _get_pipeline()
    return {
        "status": "healthy",
        "agents": pipeline.registry.list_agents(),
        "monitor": pipeline.monitor.get_health(),
    }


@app.get("/agents")
async def list_agents():
    return _get_pipeline().registry.list_agents()


@app.get("/metrics")
async def metrics():
    pipeline = _get_pipeline()
    return {
        "costs": pipeline.cost_tracker.get_summary(),
        "health": pipeline.monitor.get_health(),
        "pipeline_metrics": pipeline.monitor.get_metrics(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
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


@app.post("/query/file")
async def query_file(query: str, file: UploadFile = File(...)):
    """Upload a file and run a query against it."""
    pipeline = _get_pipeline()
    config = get_config()

    # Sanitize filename to prevent path traversal
    safe_name = sanitize_filename(file.filename or "upload")
    upload_dir = config.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name

    content = await file.read()
    file_path.write_bytes(content)

    start = time.perf_counter()
    try:
        result = await pipeline.run(
            query=query,
            context={"file_path": str(file_path)},
        )
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


@app.get("/cost-report")
async def cost_report():
    return _get_pipeline().cost_tracker.get_summary()
