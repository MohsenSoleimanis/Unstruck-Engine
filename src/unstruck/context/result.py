"""Result from a Context Engine LLM call."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ContextEngineResult:
    """Immutable result from a Context Engine call."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    blocked: bool = False
    block_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def success(self) -> bool:
        return not self.blocked and bool(self.text)
