"""Tests for A2A message bus and protocol."""

import asyncio

import pytest

from mas.a2a.bus import MessageBus
from mas.a2a.protocol import A2AProtocol, AgentCard
from mas.schemas.messages import AgentMessage, MessageRole


@pytest.fixture
def bus():
    b = MessageBus()
    b.register_agent("agent_1", "ingestion")
    b.register_agent("agent_2", "analyst")
    b.register_agent("agent_3", "analyst")
    return b


def test_register_agents(bus):
    agents = bus.get_agents()
    assert len(agents) == 3
    assert bus.get_agents_by_type("analyst") == ["agent_2", "agent_3"]


@pytest.mark.asyncio
async def test_direct_message(bus):
    msg = AgentMessage(
        sender="agent_1",
        receiver="agent_2",
        role=MessageRole.AGENT,
        content="Here is the parsed data",
        data={"items": [1, 2, 3]},
    )
    await bus.send(msg)

    received = await bus.receive("agent_2")
    assert len(received) == 1
    assert received[0]["content"] == "Here is the parsed data"
    assert received[0]["data"]["items"] == [1, 2, 3]

    # agent_1 should have no messages
    assert await bus.receive("agent_1") == []


@pytest.mark.asyncio
async def test_type_routing(bus):
    msg = AgentMessage(
        sender="agent_1",
        receiver="analyst",  # type-based routing
        role=MessageRole.AGENT,
        content="Analyze this",
    )
    await bus.send(msg)

    # Both analyst agents should receive it
    r2 = await bus.receive("agent_2")
    r3 = await bus.receive("agent_3")
    assert len(r2) == 1
    assert len(r3) == 1


@pytest.mark.asyncio
async def test_broadcast(bus):
    msg = AgentMessage(
        sender="agent_1",
        receiver="*",
        role=MessageRole.AGENT,
        content="Broadcast message",
    )
    await bus.send(msg)

    # All agents except sender should get it
    assert len(await bus.receive("agent_2")) == 1
    assert len(await bus.receive("agent_3")) == 1
    assert len(await bus.receive("agent_1")) == 0


@pytest.mark.asyncio
async def test_message_history(bus):
    for i in range(5):
        await bus.send(AgentMessage(
            sender="agent_1", receiver="agent_2",
            role=MessageRole.AGENT, content=f"msg {i}",
        ))

    history = bus.get_history()
    assert len(history) == 5


def test_stats(bus):
    s = bus.stats
    assert s["registered_agents"] == 3
    assert s["total_messages"] == 0


def test_a2a_protocol_discovery():
    protocol = A2AProtocol()
    protocol.register_card(AgentCard(
        agent_id="img_1", agent_type="image_processor",
        description="Processes images", version="0.1.0",
        input_types=["image/png", "image/jpeg"],
        output_types=["entities", "description"],
        skills=["OCR", "visual_analysis"],
    ))
    protocol.register_card(AgentCard(
        agent_id="tbl_1", agent_type="table_processor",
        description="Processes tables", version="0.1.0",
        input_types=["table"],
        output_types=["entities", "summary"],
        skills=["data_analysis"],
    ))

    # Find by type
    assert len(protocol.find_by_type("image_processor")) == 1

    # Find by skill
    assert len(protocol.find_by_skill("OCR")) == 1
    assert len(protocol.find_by_skill("data_analysis")) == 1

    # Find by input type
    assert len(protocol.find_by_input_type("image/png")) == 1
    assert len(protocol.find_by_input_type("table")) == 1

    # Find by output type
    assert len(protocol.find_by_output_type("entities")) == 2
