"""Tests for agent registry and base agent."""

from mas.agents.registry import AgentRegistry, registry


def test_registry_register_and_list():
    # The global registry should have agents auto-registered via decorators
    import mas.agents.protocol  # noqa: F401
    import mas.agents.rag  # noqa: F401
    import mas.agents.research  # noqa: F401

    agents = registry.list_agents()
    agent_types = {a["agent_type"] for a in agents}

    assert "extractor" in agent_types
    assert "explorer" in agent_types
    assert "reviewer" in agent_types
    assert "parser" in agent_types
    assert "retriever" in agent_types
    assert "multimodal" in agent_types
    assert "researcher" in agent_types
    assert "synthesizer" in agent_types


def test_registry_has():
    import mas.agents.protocol  # noqa: F401

    assert registry.has("extractor")
    assert not registry.has("nonexistent_agent")


def test_registry_contains():
    import mas.agents.protocol  # noqa: F401

    assert "extractor" in registry
    assert "fake" not in registry
