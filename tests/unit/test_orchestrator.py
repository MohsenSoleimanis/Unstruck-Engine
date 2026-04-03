"""Tests for the orchestrator — ledgers, state, and brain structure."""

import pytest

from unstruck.schemas import AgentResult, ResultStatus, Task, TaskPriority, TaskStatus
from unstruck.orchestrator.ledgers import (
    LedgerEntry,
    ProgressLedger,
    Reflection,
    TaskLedger,
)
from unstruck.orchestrator.brain import _parse_json_safe, _parse_task_list


# ── Task schema ─────────────────────────────────────────────────

class TestTask:
    def test_create_task(self):
        task = Task(agent_type="analyst", instruction="Analyze this")
        assert task.id  # Auto-generated
        assert task.agent_type == "analyst"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM

    def test_task_is_ready_no_deps(self):
        task = Task(agent_type="analyst", instruction="x")
        assert task.is_ready(set()) is True

    def test_task_is_ready_with_deps(self):
        task = Task(agent_type="analyst", instruction="x", dependencies=["task_1"])
        assert task.is_ready(set()) is False
        assert task.is_ready({"task_1"}) is True


# ── TaskLedger ──────────────────────────────────────────────────

class TestTaskLedger:
    def test_add_task(self):
        ledger = TaskLedger()
        task = Task(agent_type="analyst", instruction="Analyze")
        ledger.add_task(task)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].agent_type == "analyst"
        assert ledger.entries[0].status == TaskStatus.PENDING

    def test_record_success(self):
        ledger = TaskLedger()
        task = Task(agent_type="analyst", instruction="Analyze")
        ledger.add_task(task)

        result = AgentResult(
            task_id=task.id,
            agent_id="analyst_1",
            agent_type="analyst",
            status=ResultStatus.SUCCESS,
            output={"answer": "The study is Phase III"},
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            duration_ms=5000,
        )
        ledger.record_result(result)

        assert ledger.entries[0].status == TaskStatus.COMPLETED
        assert "Phase III" in ledger.entries[0].result_summary
        assert ledger.entries[0].cost_usd == 0.001

    def test_record_failure(self):
        ledger = TaskLedger()
        task = Task(agent_type="analyst", instruction="Analyze")
        ledger.add_task(task)

        result = AgentResult(
            task_id=task.id,
            agent_id="analyst_1",
            agent_type="analyst",
            status=ResultStatus.FAILED,
            errors=["No context available"],
        )
        ledger.record_result(result)

        assert ledger.entries[0].status == TaskStatus.FAILED
        assert "No context" in ledger.entries[0].errors[0]

    def test_completion_rate(self):
        ledger = TaskLedger()
        for i in range(4):
            ledger.add_task(Task(agent_type="analyst", instruction=f"Task {i}"))

        ledger.entries[0].status = TaskStatus.COMPLETED
        ledger.entries[1].status = TaskStatus.COMPLETED
        ledger.entries[2].status = TaskStatus.FAILED

        assert ledger.completion_rate == 0.5  # 2/4

    def test_properties(self):
        ledger = TaskLedger()
        for i in range(3):
            ledger.add_task(Task(agent_type="analyst", instruction=f"Task {i}"))

        ledger.entries[0].status = TaskStatus.COMPLETED
        ledger.entries[0].cost_usd = 0.01
        ledger.entries[0].tokens_used = 1000
        ledger.entries[1].status = TaskStatus.FAILED

        assert len(ledger.pending) == 1
        assert len(ledger.completed) == 1
        assert len(ledger.failed) == 1
        assert ledger.total_cost == 0.01
        assert ledger.total_tokens == 1000

    def test_for_prompt(self):
        ledger = TaskLedger()
        ledger.add_task(Task(agent_type="analyst", instruction="Analyze the document"))
        ledger.entries[0].status = TaskStatus.COMPLETED
        ledger.entries[0].result_summary = "Found 5 endpoints"

        text = ledger.for_prompt()
        assert "analyst" in text
        assert "Analyze" in text
        assert "5 endpoints" in text
        assert "✅" in text


# ── ProgressLedger ──────────────────────────────────────────────

class TestProgressLedger:
    def test_add_reflection(self):
        ledger = ProgressLedger()
        r = Reflection(iteration=1, completed_count=3, failed_count=1)
        ledger.add(r)
        assert len(ledger.reflections) == 1
        assert ledger.latest.completed_count == 3

    def test_is_making_progress(self):
        ledger = ProgressLedger()
        ledger.add(Reflection(iteration=1, completed_count=1, failed_count=2))
        ledger.add(Reflection(iteration=2, completed_count=3, failed_count=1))
        assert ledger.is_making_progress is True

    def test_not_making_progress(self):
        ledger = ProgressLedger()
        ledger.add(Reflection(iteration=1, completed_count=2, failed_count=1))
        ledger.add(Reflection(iteration=2, completed_count=2, failed_count=2))
        assert ledger.is_making_progress is False

    def test_for_prompt(self):
        ledger = ProgressLedger()
        ledger.add(Reflection(
            iteration=1,
            completed_count=2,
            failed_count=1,
            issues=["Agent X timed out"],
            recommendation="replan",
        ))
        text = ledger.for_prompt()
        assert "Iteration 1" in text
        assert "timed out" in text
        assert "replan" in text


# ── Brain helpers ───────────────────────────────────────────────

class TestBrainHelpers:
    def test_parse_json_safe_valid(self):
        result = _parse_json_safe('{"key": "value"}', {})
        assert result == {"key": "value"}

    def test_parse_json_safe_fenced(self):
        result = _parse_json_safe('```json\n{"key": "value"}\n```', {})
        assert result == {"key": "value"}

    def test_parse_json_safe_invalid_returns_fallback(self):
        result = _parse_json_safe("not json at all", {"fallback": True})
        assert result == {"fallback": True}

    def test_parse_task_list_basic(self):
        raw = '[{"agent_type": "analyst", "instruction": "Analyze this"}]'
        tasks = _parse_task_list(raw, {})
        assert len(tasks) == 1
        assert tasks[0].agent_type == "analyst"

    def test_parse_task_list_with_deps(self):
        raw = """[
            {"agent_type": "rag_ingest", "instruction": "Ingest", "dependencies": []},
            {"agent_type": "analyst", "instruction": "Analyze", "dependencies": [0]}
        ]"""
        tasks = _parse_task_list(raw, {})
        assert len(tasks) == 2
        assert tasks[1].dependencies == [tasks[0].id]  # Index 0 → first task's ID

    def test_parse_task_list_merges_user_context(self):
        raw = '[{"agent_type": "rag_ingest", "instruction": "Ingest"}]'
        tasks = _parse_task_list(raw, {"file_path": "test.pdf"})
        assert tasks[0].context["file_path"] == "test.pdf"

    def test_parse_task_list_invalid_returns_empty(self):
        tasks = _parse_task_list("not json", {})
        assert tasks == []
