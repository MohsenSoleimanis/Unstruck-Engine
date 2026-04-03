"""Health and system info endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health(request: Request):
    p = request.app.state.platform
    return {
        "status": "healthy",
        "agents": p.agent_registry.list_agents(),
        "tools": p.tool_registry.list_tools(),
        "rag_available": p.rag_service.is_available,
    }


@router.get("/agents")
async def list_agents(request: Request):
    return request.app.state.platform.agent_registry.list_agents()


@router.get("/tools")
async def list_tools(request: Request):
    return request.app.state.platform.tool_registry.list_tools()


@router.get("/metrics")
async def metrics(request: Request):
    p = request.app.state.platform
    return {
        "costs": p.cost_tracker.get_summary(),
        "context_engine": p.context_engine.stats,
        "audit_entries": p.audit_log.count,
    }
