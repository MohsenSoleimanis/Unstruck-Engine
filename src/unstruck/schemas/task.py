"""Task — a unit of work the orchestrator assigns to an agent."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """A unit of work assigned by the orchestrator to an agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_type: str = Field(description="Target agent from the registry")
    instruction: str = Field(description="What the agent should do")
    context: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list, description="Task IDs this depends on")
    priority: TaskPriority = TaskPriority.MEDIUM
    token_budget: int = 8000
    status: TaskStatus = TaskStatus.PENDING

    def is_ready(self, completed_ids: set[str]) -> bool:
        """True if all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.dependencies)
