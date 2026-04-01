"""Inter-agent message schemas (A2A-inspired)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    AGENT = "agent"
    SYSTEM = "system"
    USER = "user"


class AgentMessage(BaseModel):
    """Message passed between agents or between orchestrator and agents."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: str
    receiver: str
    role: MessageRole
    content: str
    data: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
