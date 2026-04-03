"""FastAPI server — production API for the platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from unstruck.api.bootstrap import Platform, create_platform
from unstruck.api.routers import conversations, files, health, query

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize platform at startup, cleanup on shutdown."""
    platform = create_platform()
    app.state.platform = platform
    logger.info("server.started",
                agents=platform.agent_registry.agent_count,
                tools=platform.tool_registry.count)
    yield


app = FastAPI(
    title="Unstruck Engine",
    description="Multi-agent orchestration platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(conversations.router)
app.include_router(files.router)

# Serve built frontend in production
_ui_dist = Path(__file__).resolve().parent.parent.parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="ui")
