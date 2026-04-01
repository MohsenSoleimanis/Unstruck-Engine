"""Agent result schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentResult(BaseModel):
    """Output from an agent after completing a task."""

    task_id: str
    agent_id: str
    agent_type: str
    status: ResultStatus
    output: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list, description="File paths or URIs produced")
    errors: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
