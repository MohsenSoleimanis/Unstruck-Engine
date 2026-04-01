"""Tests for schema models."""

from mas.schemas.messages import AgentMessage, MessageRole
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task, TaskPriority, TaskStatus


def test_task_creation():
    task = Task(
        agent_type="extractor",
        instruction="Extract endpoints from section 3",
        context={"content": "some text"},
    )
    assert task.id is not None
    assert task.status == TaskStatus.PENDING
    assert task.priority == TaskPriority.MEDIUM


def test_task_is_ready():
    t1 = Task(agent_type="parser", instruction="Parse doc")
    t2 = Task(agent_type="extractor", instruction="Extract", dependencies=[t1.id])

    assert t1.is_ready(set())
    assert not t2.is_ready(set())
    assert t2.is_ready({t1.id})


def test_agent_result():
    result = AgentResult(
        task_id="abc",
        agent_id="ext_001",
        agent_type="extractor",
        status=ResultStatus.SUCCESS,
        output={"key": "value"},
        token_usage={"input_tokens": 100, "output_tokens": 50},
        cost_usd=0.001,
    )
    assert result.status == ResultStatus.SUCCESS
    assert result.cost_usd == 0.001


def test_agent_message():
    msg = AgentMessage(
        sender="orchestrator",
        receiver="extractor_001",
        role=MessageRole.ORCHESTRATOR,
        content="Execute task",
        task_id="t1",
    )
    assert msg.role == MessageRole.ORCHESTRATOR
