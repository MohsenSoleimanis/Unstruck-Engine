"""WebSocket router — live agent and system status."""

from __future__ import annotations

import asyncio
import json
import uuid

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mas.api.events import broadcaster

logger = structlog.get_logger()

router = APIRouter(tags=["websocket"])


@router.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Live status WebSocket.

    Pushes: agent activity, pipeline phases, cost updates, health alerts.
    """
    await ws.accept()
    client_id = uuid.uuid4().hex[:8]
    queue = broadcaster.subscribe(client_id)

    try:
        # Send initial state
        from mas.api.server import app
        pipeline = app.state.pipeline

        await ws.send_json({
            "event": "init",
            "data": {
                "agents": pipeline.registry.list_agents(),
                "health": pipeline.monitor.get_health(),
                "costs": pipeline.cost_tracker.get_summary(),
                "kg_stats": pipeline.knowledge_graph.stats,
            },
        })

        # Stream events
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await ws.send_json({"event": "heartbeat"})

    except WebSocketDisconnect:
        logger.debug("ws.disconnected", client_id=client_id)
    except Exception as e:
        logger.warning("ws.error", client_id=client_id, error=str(e))
    finally:
        broadcaster.unsubscribe(client_id)
