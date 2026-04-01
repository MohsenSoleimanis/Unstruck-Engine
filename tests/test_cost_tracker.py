"""Tests for cost tracker."""

from mas.llmops.cost_tracker import CostTracker


def test_cost_tracking():
    tracker = CostTracker()
    tracker.record(
        agent_id="ext_001",
        agent_type="extractor",
        task_id="t1",
        token_usage={"input_tokens": 1000, "output_tokens": 500},
        model="gpt-4o-mini",
    )

    summary = tracker.get_summary()
    assert summary["session"]["total_tokens"] == 1500
    assert summary["session"]["num_calls"] == 1
    assert summary["session"]["total_cost_usd"] > 0
    assert "extractor" in summary["by_agent"]


def test_cost_tracking_multiple():
    tracker = CostTracker()
    for i in range(5):
        tracker.record(
            agent_id=f"agent_{i}",
            agent_type="researcher",
            task_id=f"t{i}",
            token_usage={"input_tokens": 100, "output_tokens": 50},
            model="gpt-4o-mini",
        )

    summary = tracker.get_summary()
    assert summary["session"]["num_calls"] == 5
    assert summary["session"]["total_tokens"] == 750
