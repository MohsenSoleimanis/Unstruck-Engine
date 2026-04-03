"""AgentResult — what an agent returns after executing a task."""

from __future__ import annotations

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
    errors: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
