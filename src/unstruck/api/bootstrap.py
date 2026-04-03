"""Platform bootstrap — wires all layers together.

This is the ONE place where all components are created and connected.
No layer knows about any other layer's internals — they connect
through hooks, registries, and the config system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

from unstruck.agents import AgentRegistry, AnalystAgent, KGReasonerAgent, PermissionSystem, Router, SynthesizerAgent
from unstruck.config import Config
from unstruck.context import ContextEngine, TokenBudget
from unstruck.hooks import HookManager
from unstruck.llmops import AuditLog, CostTracker, OnlineEvaluator
from unstruck.memory import AgentCache, SessionManager
from unstruck.rag import RAGService, register_rag_tools
from unstruck.tools import Guardrails, ToolRegistry
from unstruck.tools.builtin import register_builtin_tools

logger = structlog.get_logger()


class Platform:
    """
    The assembled platform — all layers wired together.

    Created once at server startup. Provides access to all services.
    """

    def __init__(
        self,
        config: Config,
        hooks: HookManager,
        context_engine: ContextEngine,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        router: Router,
        session_manager: SessionManager,
        rag_service: RAGService,
        cost_tracker: CostTracker,
        audit_log: AuditLog,
        evaluator: OnlineEvaluator,
        permissions: PermissionSystem,
    ) -> None:
        self.config = config
        self.hooks = hooks
        self.context_engine = context_engine
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self.router = router
        self.session_manager = session_manager
        self.rag_service = rag_service
        self.cost_tracker = cost_tracker
        self.audit_log = audit_log
        self.evaluator = evaluator
        self.permissions = permissions


def create_platform(env: str = "default") -> Platform:
    """
    Bootstrap the entire platform. Returns a fully wired Platform instance.

    Order matters:
      1. Load .env
      2. Config
      3. Hooks
      4. LLMOps (cost, audit, eval) → register as hooks
      5. Guardrails → register as hooks
      6. Permissions → register as hooks
      7. Memory (cache) → register as hooks
      8. Context Engine (uses hooks)
      9. Tool registry + built-in tools + RAG tools
      10. Agent registry + built-in agents
      11. Router (uses registry + context engine)
      12. Session manager
    """
    load_dotenv()

    # 1. Config
    config = Config(env=env)
    logger.info("bootstrap.config_loaded", env=env, log_level=config.log_level)

    # 2. Hooks
    hooks = HookManager()

    # 3. LLMOps — register as hooks FIRST (they observe everything)
    pricing = config.models.get("pricing", {})
    role_config = config.permissions.get("roles", {}).get("user", {})
    cost_ceiling = role_config.get("max_cost_per_session", 1.0)

    cost_tracker = CostTracker(pricing=pricing, ceiling_usd=cost_ceiling)
    cost_tracker.register_hooks(hooks)

    audit_log = AuditLog()
    audit_log.register_hooks(hooks)

    evaluator = OnlineEvaluator()
    evaluator.register_hooks(hooks)

    # 4. Guardrails → register as hooks
    guardrails = Guardrails(config.guardrails)
    guardrails.register_hooks(hooks)

    # 5. Permissions → register as hooks
    permissions = PermissionSystem(config)
    permissions.register_hooks(hooks)

    # 6. Memory cache → register as hooks
    agent_cache = AgentCache()
    agent_cache.register_hooks(hooks)

    logger.info("bootstrap.hooks_registered", total=hooks.total_handlers)

    # 7. Context Engine
    budget_config = config.get_token_budgets()
    budget = TokenBudget.from_config(budget_config)
    context_engine = ContextEngine(config, hooks, budget)

    # 8. Tool registry
    tool_registry = ToolRegistry()
    data_dir = str(config.data_dir)
    register_builtin_tools(tool_registry, config.tools, data_dir=data_dir)

    # RAG service + tools
    rag_service = RAGService(config)
    register_rag_tools(tool_registry, rag_service)

    # Wire tool registry into Context Engine (for RAG retrieval + ingestion)
    context_engine.set_tool_registry(tool_registry)

    logger.info("bootstrap.tools_registered", count=tool_registry.count)

    # 9. Agent registry
    agent_registry = AgentRegistry()
    agent_registry.load_from_config(config.agents)

    # Register built-in agents
    agent_registry.register(AnalystAgent, config.agents.get("analyst"))
    agent_registry.register(SynthesizerAgent, config.agents.get("synthesizer"))
    agent_registry.register(KGReasonerAgent, config.agents.get("kg_reasoner"))

    # Discover and register plugin agents
    plugins = config.discover_plugins()
    for plugin_name, plugin_config in plugins.items():
        logger.info("bootstrap.plugin_discovered", plugin=plugin_name)
        # Plugin loading would go here — for now just log
        # The plugin system will load agent.py from the plugin dir

    logger.info("bootstrap.agents_registered", count=agent_registry.agent_count)

    # 10. Router
    router = Router(agent_registry, context_engine)

    # 11. Session manager
    session_manager = SessionManager(config.data_dir / "sessions")

    platform = Platform(
        config=config,
        hooks=hooks,
        context_engine=context_engine,
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        router=router,
        session_manager=session_manager,
        rag_service=rag_service,
        cost_tracker=cost_tracker,
        audit_log=audit_log,
        evaluator=evaluator,
        permissions=permissions,
    )

    logger.info("bootstrap.complete")
    return platform
