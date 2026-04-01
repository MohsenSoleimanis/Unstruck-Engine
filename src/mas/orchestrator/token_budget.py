"""Token budget management — controls context size and spending per pipeline run.

No more arbitrary character slicing. Token counting via tiktoken,
budget allocation per agent, early synthesis when budget exhausted.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

# Lazy-loaded tiktoken encoder
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text. Falls back to word-based estimate if tiktoken unavailable."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    enc = _get_encoder()
    if enc:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    # Fallback: character-based estimate
    max_chars = max_tokens * 4
    return text[:max_chars]


class TokenBudget:
    """
    Tracks and enforces token budgets for a pipeline run.

    - Total budget: max tokens across all agents
    - Per-agent budget: max tokens for a single agent call
    - Context budget: max tokens for context injection into an agent
    - Triggers early synthesis when budget is mostly consumed
    """

    def __init__(
        self,
        total_budget: int = 50000,
        per_agent_budget: int = 8000,
        context_budget: int = 12000,
        synthesis_threshold: float = 0.85,
    ) -> None:
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self.context_budget = context_budget
        self.synthesis_threshold = synthesis_threshold
        self._consumed: dict[str, int] = {}  # agent_type → tokens consumed
        self._total_consumed = 0

    def record_usage(self, agent_type: str, tokens: int) -> None:
        """Record tokens consumed by an agent."""
        self._consumed[agent_type] = self._consumed.get(agent_type, 0) + tokens
        self._total_consumed += tokens
        logger.debug(
            "token_budget.recorded",
            agent=agent_type,
            tokens=tokens,
            total=self._total_consumed,
            remaining=self.remaining,
        )

    @property
    def consumed(self) -> int:
        return self._total_consumed

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self._total_consumed)

    @property
    def utilization(self) -> float:
        return self._total_consumed / self.total_budget if self.total_budget > 0 else 0

    def can_continue(self) -> bool:
        """False if budget is exhausted — should trigger early synthesis."""
        return self.utilization < self.synthesis_threshold

    def allocate_for_agent(self, agent_type: str) -> int:
        """Return max tokens this agent should use."""
        return min(self.per_agent_budget, self.remaining)

    def truncate_context(self, text: str) -> str:
        """Truncate context to fit within context budget."""
        return truncate_to_tokens(text, self.context_budget)

    def get_summary(self) -> dict[str, int | float]:
        return {
            "total_budget": self.total_budget,
            "consumed": self._total_consumed,
            "remaining": self.remaining,
            "utilization": round(self.utilization, 3),
            "by_agent": dict(self._consumed),
        }
