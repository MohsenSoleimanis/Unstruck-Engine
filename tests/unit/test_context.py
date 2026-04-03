"""Tests for the Context Engine — tokens, budget, and engine."""

import pytest

from unstruck.context.tokens import count_tokens, truncate_to_tokens
from unstruck.context.budget import TokenBudget
from unstruck.context.result import ContextEngineResult


# --- Token counting ---

class TestTokens:
    def test_count_tokens_returns_positive(self):
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_count_tokens_longer_text_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many more words in it")
        assert long > short

    def test_truncate_to_tokens_short_text_unchanged(self):
        text = "Hello"
        result = truncate_to_tokens(text, 100)
        assert result == text

    def test_truncate_to_tokens_long_text_truncated(self):
        text = "word " * 1000  # ~1000 tokens
        result = truncate_to_tokens(text, 10)
        assert len(result) < len(text)
        assert count_tokens(result) <= 10


# --- Token budget ---

class TestBudget:
    def test_initial_state(self):
        budget = TokenBudget(total=10000)
        assert budget.consumed == 0
        assert budget.remaining == 10000
        assert budget.utilization == 0.0
        assert budget.can_continue() is True

    def test_record_updates_consumed(self):
        budget = TokenBudget(total=10000)
        budget.record("agent_a", 3000)
        assert budget.consumed == 3000
        assert budget.remaining == 7000

    def test_multiple_records_accumulate(self):
        budget = TokenBudget(total=10000)
        budget.record("agent_a", 3000)
        budget.record("agent_b", 2000)
        budget.record("agent_a", 1000)
        assert budget.consumed == 6000
        assert budget.remaining == 4000

    def test_can_continue_false_at_threshold(self):
        budget = TokenBudget(total=10000, synthesis_threshold=0.85)
        budget.record("agent_a", 8600)  # 86% > 85%
        assert budget.can_continue() is False

    def test_allocate_respects_per_agent_limit(self):
        budget = TokenBudget(total=50000, per_agent=8000)
        assert budget.allocate("agent_a") == 8000

    def test_allocate_respects_remaining_budget(self):
        budget = TokenBudget(total=10000, per_agent=8000)
        budget.record("agent_a", 5000)
        assert budget.allocate("agent_b") == 5000  # min(8000, 5000 remaining)

    def test_allocate_zero_when_exhausted(self):
        budget = TokenBudget(total=100)
        budget.record("agent_a", 100)
        assert budget.allocate("agent_b") == 0

    def test_to_dict(self):
        budget = TokenBudget(total=10000)
        budget.record("agent_a", 3000)
        d = budget.to_dict()
        assert d["total"] == 10000
        assert d["consumed"] == 3000
        assert d["remaining"] == 7000
        assert d["by_agent"]["agent_a"] == 3000

    def test_from_config(self):
        config = {
            "total_budget": 30000,
            "per_agent_budget": 5000,
            "context_budget": 10000,
            "synthesis_threshold": 0.90,
        }
        budget = TokenBudget.from_config(config)
        assert budget.total == 30000
        assert budget.per_agent == 5000
        assert budget.context_limit == 10000
        assert budget.synthesis_threshold == 0.90


# --- ContextEngineResult ---

class TestResult:
    def test_success_result(self):
        r = ContextEngineResult(text="Hello", input_tokens=10, output_tokens=5, cost_usd=0.001)
        assert r.success is True
        assert r.total_tokens == 15
        assert r.blocked is False

    def test_blocked_result(self):
        r = ContextEngineResult(blocked=True, block_reason="Budget exceeded")
        assert r.success is False
        assert r.blocked is True
        assert r.block_reason == "Budget exceeded"

    def test_empty_text_not_success(self):
        r = ContextEngineResult(text="")
        assert r.success is False

    def test_immutable(self):
        r = ContextEngineResult(text="Hello")
        with pytest.raises(AttributeError):
            r.text = "Modified"
