"""Tests for agent registry and agent registration."""

from mas.agents.registry import registry


def test_registry_all_agents_registered():
    """All RAG-Anything-inspired agents should auto-register via decorators."""
    import mas.agents.ingestion  # noqa: F401
    import mas.agents.kg  # noqa: F401
    import mas.agents.modal  # noqa: F401
    import mas.agents.reasoning  # noqa: F401
    import mas.agents.retrieval  # noqa: F401

    agents = registry.list_agents()
    agent_types = {a["agent_type"] for a in agents}

    # Ingestion pipeline
    assert "ingestion" in agent_types
    assert "separator" in agent_types
    assert "chunker" in agent_types

    # Modal processors
    assert "image_processor" in agent_types
    assert "table_processor" in agent_types
    assert "text_processor" in agent_types

    # Knowledge graph
    assert "kg_builder" in agent_types
    assert "kg_query" in agent_types

    # Retrieval
    assert "embedder" in agent_types
    assert "hybrid_retriever" in agent_types

    # Reasoning
    assert "analyst" in agent_types
    assert "synthesizer" in agent_types


def test_registry_has():
    import mas.agents.ingestion  # noqa: F401

    assert registry.has("ingestion")
    assert not registry.has("nonexistent_agent")


def test_registry_agent_count():
    import mas.agents.ingestion  # noqa: F401
    import mas.agents.kg  # noqa: F401
    import mas.agents.modal  # noqa: F401
    import mas.agents.reasoning  # noqa: F401
    import mas.agents.retrieval  # noqa: F401

    agents = registry.list_agents()
    assert len(agents) >= 12  # 3 ingestion + 3 modal + 2 kg + 2 retrieval + 2 reasoning
