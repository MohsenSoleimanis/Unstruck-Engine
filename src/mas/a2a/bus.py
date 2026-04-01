"""A2A Message Bus — peer-to-peer inter-agent communication.

Implements the horizontal communication layer from the A2A protocol spec:
  - Agents register with the bus (announce capabilities)
  - Agents send messages to specific agents or broadcast by type
  - Messages are queued per-agent and delivered on receive()
  - Supports request/reply patterns via correlation IDs
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import structlog

from mas.schemas.messages import AgentMessage, MessageRole

logger = structlog.get_logger()


class MessageBus:
    """
    Central message bus for inter-agent communication.

    Supports:
      - Direct messaging: agent_id → agent_id
      - Type-based routing: send to all agents of a given type
      - Broadcast: send to all agents
      - Request/reply: correlated via task_id
      - Message history: full audit trail
    """

    def __init__(self) -> None:
        # agent_id → agent_type mapping
        self._agents: dict[str, str] = {}
        # agent_type → list of agent_ids
        self._type_index: dict[str, list[str]] = defaultdict(list)
        # agent_id → queue of pending messages
        self._queues: dict[str, asyncio.Queue[AgentMessage]] = {}
        # Full message history for observability
        self._history: list[AgentMessage] = []

    def register_agent(self, agent_id: str, agent_type: str) -> None:
        """Register an agent with the bus for message delivery."""
        self._agents[agent_id] = agent_type
        if agent_id not in self._type_index[agent_type]:
            self._type_index[agent_type].append(agent_id)
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
        logger.debug("a2a.registered", agent_id=agent_id, agent_type=agent_type)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the bus."""
        agent_type = self._agents.pop(agent_id, None)
        if agent_type and agent_id in self._type_index[agent_type]:
            self._type_index[agent_type].remove(agent_id)
        self._queues.pop(agent_id, None)

    async def send(self, message: AgentMessage) -> None:
        """
        Send a message. Receiver can be:
          - A specific agent_id (direct)
          - An agent_type (routes to first available agent of that type)
          - "*" for broadcast to all agents
        """
        self._history.append(message)
        receiver = message.receiver

        if receiver == "*":
            # Broadcast
            for agent_id, queue in self._queues.items():
                if agent_id != message.sender:
                    await queue.put(message)
            logger.debug("a2a.broadcast", sender=message.sender, agents=len(self._queues) - 1)

        elif receiver in self._queues:
            # Direct by agent_id
            await self._queues[receiver].put(message)
            logger.debug("a2a.direct", sender=message.sender, receiver=receiver)

        elif receiver in self._type_index:
            # Route by agent_type — deliver to all agents of that type
            targets = self._type_index[receiver]
            for target_id in targets:
                if target_id in self._queues:
                    await self._queues[target_id].put(message)
            logger.debug("a2a.type_route", sender=message.sender, type=receiver, targets=len(targets))

        else:
            logger.warning("a2a.no_recipient", sender=message.sender, receiver=receiver)

    async def receive(self, agent_id: str) -> list[dict[str, Any]]:
        """Receive all pending messages for an agent (non-blocking)."""
        if agent_id not in self._queues:
            return []

        messages = []
        queue = self._queues[agent_id]
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                messages.append({
                    "id": msg.id,
                    "sender": msg.sender,
                    "role": msg.role.value,
                    "content": msg.content,
                    "data": msg.data,
                    "task_id": msg.task_id,
                    "timestamp": msg.timestamp.isoformat(),
                })
            except asyncio.QueueEmpty:
                break

        return messages

    async def request(self, message: AgentMessage, timeout: float = 30.0) -> AgentMessage | None:
        """
        Send a message and wait for a reply (request/reply pattern).

        Waits for a message back to the sender with the same task_id.
        """
        await self.send(message)

        # Wait for reply
        if message.sender not in self._queues:
            return None

        queue = self._queues[message.sender]
        try:
            reply = await asyncio.wait_for(queue.get(), timeout=timeout)
            return reply
        except asyncio.TimeoutError:
            logger.warning("a2a.request_timeout", sender=message.sender, task_id=message.task_id)
            return None

    def get_agents(self) -> list[dict[str, str]]:
        """List all registered agents."""
        return [
            {"agent_id": aid, "agent_type": atype}
            for aid, atype in self._agents.items()
        ]

    def get_agents_by_type(self, agent_type: str) -> list[str]:
        """Get all agent_ids for a given type."""
        return list(self._type_index.get(agent_type, []))

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent message history for observability."""
        return [
            {
                "id": m.id,
                "sender": m.sender,
                "receiver": m.receiver,
                "content": m.content[:200],
                "task_id": m.task_id,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in self._history[-limit:]
        ]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "registered_agents": len(self._agents),
            "total_messages": len(self._history),
            "pending_messages": sum(q.qsize() for q in self._queues.values()),
        }
