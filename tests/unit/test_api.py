"""Tests for the API layer — bootstrap, server, and routers."""

import pytest

from unstruck.api.bootstrap import Platform, create_platform
from unstruck.config import Config


class TestBootstrap:
    def test_create_platform(self):
        """The platform bootstraps without errors and all layers are wired."""
        platform = create_platform()

        assert isinstance(platform, Platform)
        assert platform.config is not None
        assert platform.hooks is not None
        assert platform.context_engine is not None
        assert platform.agent_registry is not None
        assert platform.tool_registry is not None
        assert platform.router is not None
        assert platform.session_manager is not None
        assert platform.rag_service is not None
        assert platform.cost_tracker is not None
        assert platform.audit_log is not None
        assert platform.evaluator is not None
        assert platform.permissions is not None

    def test_agents_registered(self):
        platform = create_platform()
        agents = platform.agent_registry.list_agents()
        types = [a["agent_type"] for a in agents]
        assert "analyst" in types
        assert "synthesizer" in types
        assert "kg_reasoner" in types

    def test_tools_registered(self):
        platform = create_platform()
        tools = platform.tool_registry.list_tools()
        names = [t["name"] for t in tools]
        assert "fs_read" in names
        assert "http_get" in names
        assert "db_query" in names
        assert "rag_ingest" in names
        assert "rag_query" in names

    def test_hooks_wired(self):
        platform = create_platform()
        # Cost tracker, audit log, evaluator, guardrails, permissions, cache
        # All register hooks — total should be > 10
        assert platform.hooks.total_handlers > 10

    def test_session_manager_works(self):
        platform = create_platform()
        session = platform.session_manager.get("test_session")
        assert session.session_id == "test_session"


class TestServerStructure:
    def test_app_imports(self):
        from unstruck.api.server import app
        assert app.title == "Unstruck Engine"

    def test_routes_exist(self):
        from unstruck.api.server import app
        paths = [route.path for route in app.routes if hasattr(route, "path")]
        assert "/api/health" in paths
        assert "/api/agents" in paths
        assert "/api/tools" in paths
        assert "/api/metrics" in paths
        assert "/api/query" in paths
        assert "/api/query/stream" in paths
        assert "/api/conversations" in paths
        assert "/api/files" in paths
        assert "/api/files/upload" in paths
