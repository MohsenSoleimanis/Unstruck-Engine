"""Tests for the agent layer — registry, permissions, router."""

import asyncio

import pytest

from unstruck.agents.registry import AgentRegistry
from unstruck.agents.permissions import PermissionSystem
from unstruck.agents.base import BaseAgent
from unstruck.agents.builtin import AnalystAgent, SynthesizerAgent, KGReasonerAgent
from unstruck.config import Config
from unstruck.context import ContextEngine, TokenBudget
from unstruck.hooks import HookEvent, HookManager, HookResult
from unstruck.schemas import AgentResult, ResultStatus, Task


# ── Test agent for unit tests ───────────────────────────────────

class EchoAgent(BaseAgent):
    """Simple agent that echoes its instruction. For testing."""
    agent_type = "echo"

    async def execute(self, task: Task) -> AgentResult:
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={"echo": task.instruction},
        )


class FailAgent(BaseAgent):
    """Agent that always fails. For testing error handling."""
    agent_type = "fail"

    async def execute(self, task: Task) -> AgentResult:
        raise RuntimeError("Intentional failure")


# ── Registry ────────────────────────────────────────────────────

class TestRegistry:
    def test_register_and_create(self):
        registry = AgentRegistry()
        registry.register(EchoAgent)

        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        agent = registry.create("echo", engine)
        assert isinstance(agent, EchoAgent)
        assert agent.agent_type == "echo"

    def test_register_unknown_raises(self):
        registry = AgentRegistry()
        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        with pytest.raises(ValueError, match="Unknown agent type"):
            registry.create("nonexistent", engine)

    def test_has(self):
        registry = AgentRegistry()
        registry.register(EchoAgent)
        assert registry.has("echo") is True
        assert registry.has("nonexistent") is False
        assert "echo" in registry

    def test_list_agents(self):
        registry = AgentRegistry()
        registry.register(EchoAgent, config={"description": "Echoes input", "version": "1.0"})
        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0]["agent_type"] == "echo"
        assert agents[0]["description"] == "Echoes input"

    def test_load_from_config(self):
        registry = AgentRegistry()
        registry.register(EchoAgent)
        registry.load_from_config({
            "echo": {"description": "Updated description", "model_tier": "cheap"},
        })
        agents = registry.list_agents()
        assert agents[0]["description"] == "Updated description"
        assert agents[0]["model_tier"] == "cheap"

    def test_builtin_agents_have_correct_types(self):
        assert AnalystAgent.agent_type == "analyst"
        assert SynthesizerAgent.agent_type == "synthesizer"
        assert KGReasonerAgent.agent_type == "kg_reasoner"


# ── Base agent ──────────────────────────────────────────────────

class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_run_success(self):
        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        agent = EchoAgent(agent_id="echo_1", context_engine=engine)
        task = Task(agent_type="echo", instruction="Hello world")
        result = await agent.run(task)

        assert result.status == ResultStatus.SUCCESS
        assert result.output["echo"] == "Hello world"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_error_returns_failed_not_raises(self):
        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        agent = FailAgent(agent_id="fail_1", context_engine=engine)
        task = Task(agent_type="fail", instruction="Do something")

        # Should NOT raise — errors-as-feedback
        result = await agent.run(task)
        assert result.status == ResultStatus.FAILED
        assert "Intentional failure" in result.errors[0]


# ── Permissions ─────────────────────────────────────────────────

class TestPermissions:
    @pytest.mark.asyncio
    async def test_allow_when_tool_in_allowed_list(self):
        config = Config()
        perms = PermissionSystem(config)

        result = await perms._check_permission(
            HookEvent.PRE_TOOL_USE,
            {"agent_type": "kg_reasoner", "tool_name": "rag_query", "user_role": "admin"},
        )
        assert result.action.value == "allow"

    @pytest.mark.asyncio
    async def test_block_when_tool_not_in_allowed_list(self):
        config = Config()
        perms = PermissionSystem(config)

        result = await perms._check_permission(
            HookEvent.PRE_TOOL_USE,
            {"agent_type": "kg_reasoner", "tool_name": "fs_write", "user_role": "admin"},
        )
        assert result.action.value == "block"
        assert "not allowed" in result.reason

    @pytest.mark.asyncio
    async def test_allow_when_agent_has_no_restrictions(self):
        config = Config()
        perms = PermissionSystem(config)

        # analyst has allowed_tools: [] which means "all tools"
        result = await perms._check_permission(
            HookEvent.PRE_TOOL_USE,
            {"agent_type": "analyst", "tool_name": "rag_query", "user_role": "admin"},
        )
        assert result.action.value == "allow"

    @pytest.mark.asyncio
    async def test_block_by_role(self):
        config = Config()
        perms = PermissionSystem(config)

        # viewer role can only use rag_query and fs_read
        result = await perms._check_permission(
            HookEvent.PRE_TOOL_USE,
            {"agent_type": "analyst", "tool_name": "fs_write", "user_role": "viewer"},
        )
        assert result.action.value == "block"
        assert "Role" in result.reason

    def test_get_visible_tools(self):
        config = Config()
        perms = PermissionSystem(config)

        # kg_reasoner has allowed_tools: [rag_query]
        visible = perms.get_visible_tools("kg_reasoner")
        assert visible == ["rag_query"]

        # analyst has allowed_tools: [] → sees all tools
        visible = perms.get_visible_tools("analyst")
        assert len(visible) > 1

    @pytest.mark.asyncio
    async def test_registers_as_hook(self):
        config = Config()
        hooks = HookManager()
        perms = PermissionSystem(config)
        perms.register_hooks(hooks)

        assert hooks.handler_count(HookEvent.PRE_TOOL_USE) == 1

    def test_denial_tracking(self):
        config = Config()
        perms = PermissionSystem(config)
        assert perms.denial_count == 0


# ── Router ──────────────────────────────────────────────────────

class TestRouter:
    @pytest.mark.asyncio
    async def test_execute_single_task(self):
        from unstruck.agents.router import Router

        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        registry = AgentRegistry()
        registry.register(EchoAgent)

        router = Router(registry, engine)
        task = Task(agent_type="echo", instruction="Test")
        result = await router.execute_task(task)

        assert result.status == ResultStatus.SUCCESS
        assert result.output["echo"] == "Test"

    @pytest.mark.asyncio
    async def test_execute_plan_with_deps(self):
        from unstruck.agents.router import Router

        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        registry = AgentRegistry()
        registry.register(EchoAgent)

        router = Router(registry, engine)

        task1 = Task(agent_type="echo", instruction="First")
        task2 = Task(agent_type="echo", instruction="Second", dependencies=[task1.id])

        results = await router.execute_plan([task1, task2])
        assert len(results) == 2
        assert all(r.status == ResultStatus.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_execute_plan_deadlock(self):
        from unstruck.agents.router import Router

        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        registry = AgentRegistry()
        registry.register(EchoAgent)

        router = Router(registry, engine)

        # Circular dependency → deadlock
        task1 = Task(agent_type="echo", instruction="A", dependencies=["nonexistent"])
        results = await router.execute_plan([task1])
        assert results[0].status == ResultStatus.FAILED
        assert "Deadlock" in results[0].errors[0]

    @pytest.mark.asyncio
    async def test_error_agent_returns_failed(self):
        from unstruck.agents.router import Router

        config = Config()
        hooks = HookManager()
        budget = TokenBudget()
        engine = ContextEngine(config, hooks, budget)

        registry = AgentRegistry()
        registry.register(FailAgent)

        router = Router(registry, engine)
        task = Task(agent_type="fail", instruction="Crash")
        result = await router.execute_task(task)

        assert result.status == ResultStatus.FAILED
        assert "Intentional failure" in result.errors[0]
