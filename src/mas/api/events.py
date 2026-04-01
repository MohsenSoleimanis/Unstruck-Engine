"""Event broadcaster — pub/sub for SSE and WebSocket clients."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger()


class EventBroadcaster:
    """
    Thin pub/sub for pushing real-time events to connected UI clients.

    The pipeline, cost tracker, and message bus publish events here.
    SSE and WebSocket handlers consume them.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    def subscribe(self, client_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Register a new client and return its event queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers[client_id] = queue
        logger.debug("events.subscribed", client_id=client_id)
        return queue

    def unsubscribe(self, client_id: str) -> None:
        """Remove a client."""
        self._subscribers.pop(client_id, None)
        logger.debug("events.unsubscribed", client_id=client_id)

    async def publish(self, event: dict[str, Any]) -> None:
        """Broadcast an event to all subscribers."""
        dead: list[str] = []
        for client_id, queue in self._subscribers.items():
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(client_id)

        for client_id in dead:
            self._subscribers.pop(client_id, None)
            logger.warning("events.client_dropped", client_id=client_id)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Global singleton
broadcaster = EventBroadcaster()
