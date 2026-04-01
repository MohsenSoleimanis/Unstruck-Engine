"""Tests for memory layer."""

import time

from mas.memory.knowledge_graph import KnowledgeGraph
from mas.memory.local import LocalMemory


def test_local_memory_set_get():
    mem = LocalMemory(namespace="test")
    mem.set("key1", "value1")
    assert mem.get("key1") == "value1"
    assert mem.get("missing", "default") == "default"


def test_local_memory_ttl():
    mem = LocalMemory(namespace="test", default_ttl=1)
    mem.set("key1", "value1", ttl=0)  # Expires immediately
    time.sleep(0.1)
    assert mem.get("key1") is None


def test_local_memory_context():
    mem = LocalMemory(namespace="test")
    mem.set("a", 1)
    mem.set("b", 2)
    ctx = mem.get_context()
    assert ctx == {"a": 1, "b": 2}


def test_knowledge_graph_entities():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Drug", {"name": "Aspirin"})
    kg.add_entity("e2", "Condition", {"name": "Headache"})
    kg.add_relationship("e1", "e2", "TREATS")

    entity = kg.get_entity("e1")
    assert entity["entity_type"] == "Drug"
    assert entity["name"] == "Aspirin"

    neighbors = kg.get_neighbors("e1")
    assert len(neighbors) == 1
    assert neighbors[0]["entity_id"] == "e2"


def test_knowledge_graph_search():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Drug", {"name": "Aspirin"})
    kg.add_entity("e2", "Drug", {"name": "Ibuprofen"})
    kg.add_entity("e3", "Condition", {"name": "Pain"})

    drugs = kg.search_entities(entity_type="Drug")
    assert len(drugs) == 2

    assert kg.stats["nodes"] == 3


def test_knowledge_graph_subgraph():
    kg = KnowledgeGraph()
    kg.add_entity("a", "Node")
    kg.add_entity("b", "Node")
    kg.add_entity("c", "Node")
    kg.add_relationship("a", "b", "LINKS")
    kg.add_relationship("b", "c", "LINKS")

    sub = kg.get_subgraph("a", depth=1)
    assert len(sub["nodes"]) == 2  # a and b only (depth=1)
