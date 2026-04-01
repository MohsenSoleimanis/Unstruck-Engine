"""FastAPI server — production API for the multi-agent system."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mas.config import get_config
from mas.pipeline import MASPipeline

app = FastAPI(
    title="Multi-Agent System API",
    description="Production-scale, data-agnostic multi-agent orchestration system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-init pipeline
_pipeline: MASPipeline | None = None


def get_pipeline() -> MASPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MASPipeline()
    return _pipeline


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    query: str
    context: dict[str, Any] = {}
    max_iterations: int = 3


class QueryResponse(BaseModel):
    status: str
    query: str
    results: dict[str, Any]
    cost: dict[str, Any]
    duration_ms: int


class FileQueryRequest(BaseModel):
    query: str
    context: dict[str, Any] = {}


# --- Endpoints ---


@app.get("/health")
async def health():
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "agents": pipeline.registry.list_agents(),
        "monitor": pipeline.monitor.get_health(),
    }


@app.get("/agents")
async def list_agents():
    return get_pipeline().registry.list_agents()


@app.get("/metrics")
async def metrics():
    pipeline = get_pipeline()
    return {
        "costs": pipeline.cost_tracker.get_summary(),
        "health": pipeline.monitor.get_health(),
        "pipeline_metrics": pipeline.monitor.get_metrics(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    pipeline = get_pipeline()
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/file")
async def query_file(query: str, file: UploadFile = File(...)):
    """Upload a file and run a query against it."""
    pipeline = get_pipeline()
    config = get_config()

    # Save uploaded file
    upload_dir = config.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
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
            "file": file.filename,
            "results": result,
            "cost": pipeline.cost_tracker.get_summary(),
            "duration_ms": duration_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cost-report")
async def cost_report():
    return get_pipeline().cost_tracker.get_summary()
