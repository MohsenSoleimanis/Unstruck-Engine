"""A2A Protocol — agent discovery and capability advertisement.

Implements the A2A agent card pattern:
  - Each agent publishes a card describing its capabilities
  - Agents can discover other agents and their skills
  - Supports dynamic agent addition/removal
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentCard(BaseModel):
    """
    A2A Agent Card — describes an agent's capabilities for discovery.

    Based on Google's A2A protocol spec: agents advertise what they can do
    so other agents (or the orchestrator) can find the right agent for a task.
    """

    agent_id: str
    agent_type: str
    description: str
    version: str
    input_types: list[str] = Field(default_factory=list, description="Content types this agent accepts")
    output_types: list[str] = Field(default_factory=list, description="Content types this agent produces")
    skills: list[str] = Field(default_factory=list, description="Specific skills/capabilities")
    status: str = "available"  # available | busy | offline


class A2AProtocol:
    """
    Agent-to-Agent protocol manager.

    Manages agent cards and provides discovery services:
      - Register/update agent cards
      - Find agents by capability, input/output type, or skill
      - Health checking
    """

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def register_card(self, card: AgentCard) -> None:
        """Register or update an agent's capability card."""
        self._cards[card.agent_id] = card

    def remove_card(self, agent_id: str) -> None:
        self._cards.pop(agent_id, None)

    def get_card(self, agent_id: str) -> AgentCard | None:
        return self._cards.get(agent_id)

    def find_by_type(self, agent_type: str) -> list[AgentCard]:
        """Find all agents of a given type."""
        return [c for c in self._cards.values() if c.agent_type == agent_type and c.status == "available"]

    def find_by_skill(self, skill: str) -> list[AgentCard]:
        """Find agents that have a specific skill."""
        return [
            c for c in self._cards.values()
            if skill.lower() in [s.lower() for s in c.skills] and c.status == "available"
        ]

    def find_by_input_type(self, input_type: str) -> list[AgentCard]:
        """Find agents that accept a specific input type."""
        return [
            c for c in self._cards.values()
            if input_type in c.input_types and c.status == "available"
        ]

    def find_by_output_type(self, output_type: str) -> list[AgentCard]:
        """Find agents that produce a specific output type."""
        return [
            c for c in self._cards.values()
            if output_type in c.output_types and c.status == "available"
        ]

    def list_all(self) -> list[dict[str, Any]]:
        """List all registered agent cards."""
        return [c.model_dump() for c in self._cards.values()]

    def set_status(self, agent_id: str, status: str) -> None:
        if agent_id in self._cards:
            self._cards[agent_id].status = status
