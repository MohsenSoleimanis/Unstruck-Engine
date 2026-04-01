"""Task schemas for orchestrator-to-agent communication."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """A unit of work assigned by the orchestrator to an agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    agent_type: str = Field(description="Target agent type from the registry")
    instruction: str = Field(description="What the agent should do")
    context: dict[str, Any] = Field(default_factory=dict, description="Input data / context")
    dependencies: list[str] = Field(default_factory=list, description="Task IDs this depends on")
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_ready(self, completed_ids: set[str]) -> bool:
        return all(dep in completed_ids for dep in self.dependencies)
