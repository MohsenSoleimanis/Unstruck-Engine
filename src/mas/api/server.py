"""FastAPI server — production API with router-based architecture."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mas.api.routers import conversations, files, knowledge_graph, query, ws
from mas.api.storage import ConversationStore
from mas.config import get_config
from mas.pipeline import MASPipeline

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline and storage at startup."""
    config = get_config()
    app.state.pipeline = MASPipeline(config=config)
    app.state.conversation_store = ConversationStore(config.data_dir / "conversations")
    logger.info("server.started", agents=len(app.state.pipeline.registry.list_agents()))
    yield


app = FastAPI(
    title="Multi-Agent System",
    description="Production-scale, data-agnostic multi-agent orchestration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(query.router)
app.include_router(conversations.router)
app.include_router(files.router)
app.include_router(knowledge_graph.router)
app.include_router(ws.router)


# Health + metrics at root level
@app.get("/api/health")
async def health():
    pipeline = app.state.pipeline
    return {
        "status": "healthy",
        "agents": pipeline.registry.list_agents(),
        "monitor": pipeline.monitor.get_health(),
    }


@app.get("/api/metrics")
async def metrics():
    pipeline = app.state.pipeline
    return {
        "costs": pipeline.cost_tracker.get_summary(),
        "health": pipeline.monitor.get_health(),
        "pipeline_metrics": pipeline.monitor.get_metrics(),
    }


@app.get("/api/agents")
async def list_agents():
    return app.state.pipeline.registry.list_agents()


# Serve built frontend in production
_ui_dist = Path(__file__).parent.parent.parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="ui")
