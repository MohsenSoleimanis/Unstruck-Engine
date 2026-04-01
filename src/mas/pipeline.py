"""Main pipeline — ties together orchestrator, agents, memory, and LLMOps."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
load_dotenv()

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from mas.a2a.bus import MessageBus
from mas.agents.ingestion import ChunkerAgent, ContentSeparatorAgent, IngestionAgent  # noqa: F401
from mas.agents.kg import KGBuilderAgent, KGQueryAgent  # noqa: F401
from mas.agents.modal import ImageProcessor, TableProcessor, TextProcessor  # noqa: F401
from mas.agents.reasoning import AnalystAgent, SynthesizerAgent  # noqa: F401
from mas.agents.registry import registry
from mas.agents.retrieval import EmbedderAgent, HybridRetrieverAgent  # noqa: F401
from mas.config import MASConfig, get_config
from mas.llmops.cost_tracker import CostTracker
from mas.llmops.evaluation import EvaluationEngine
from mas.llmops.monitoring import HealthMonitor
from mas.llmops.tracing import TracingManager
from mas.memory.knowledge_graph import KnowledgeGraph
from mas.memory.shared import SharedMemory
from mas.orchestrator.graph import build_orchestrator_graph
from mas.tools.mcp_client import MCPToolClient

logger = structlog.get_logger()


class MASPipeline:
    """
    Production pipeline — single entry point for the multi-agent system.

    Wires together all 5 layers:
      - Orchestrator: LangGraph (plan → execute → review → synthesize)
      - Agents: 12 data-agnostic agents via registry
      - Memory: SharedMemory (ChromaDB) + KnowledgeGraph (NetworkX/Neo4j) + LocalMemory
      - Tools: MCP client for external data/tool access
      - LLMOps: tracing, cost tracking, evaluation, monitoring
      - A2A: MessageBus for inter-agent communication
    """

    def __init__(self, config: MASConfig | None = None) -> None:
        self.config = config or get_config()
        self.config.ensure_dirs()

        # LLMOps
        self.cost_tracker = CostTracker(output_dir=self.config.output_dir)
        self.tracing = TracingManager(
            enabled=self.config.llmops.enable_tracing,
            langfuse_public_key=self.config.llmops.langfuse_public_key,
            langfuse_secret_key=self.config.llmops.langfuse_secret_key,
            langfuse_host=self.config.llmops.langfuse_host,
        )
        self.evaluator = EvaluationEngine()
        self.monitor = HealthMonitor()

        # Memory
        self.shared_memory = SharedMemory(
            collection_name=self.config.memory.shared_collection,
            persist_dir=str(self.config.memory.chroma_persist_dir),
        )
        self.knowledge_graph = KnowledgeGraph(
            neo4j_uri=self.config.memory.neo4j_uri if self.config.memory.neo4j_uri != "bolt://localhost:7687" else None,
        )

        # A2A message bus
        self.message_bus = MessageBus()

        # MCP tool client
        self.mcp_client = MCPToolClient()
        self.mcp_client.register_builtin_tools()

        # LLMs
        self.orchestrator_llm = self._create_llm(self.config.llm.orchestrator_model)
        self.worker_llm = self._create_llm(self.config.llm.worker_model)

        # Registry (agents auto-register via decorators on import above)
        self.registry = registry

        # Build orchestrator graph — passes all infrastructure to agents
        self._graph = build_orchestrator_graph(
            orchestrator_llm=self.orchestrator_llm,
            worker_llm=self.worker_llm,
            registry=self.registry,
            shared_memory=self.shared_memory,
            knowledge_graph=self.knowledge_graph,
            cost_tracker=self.cost_tracker,
            tracing=self.tracing,
            monitor=self.monitor,
            message_bus=self.message_bus,
            mcp_client=self.mcp_client,
        )
        self._compiled = self._graph.compile()

        logger.info(
            "pipeline.initialized",
            agents=len(self.registry.list_agents()),
            orchestrator=self.config.llm.orchestrator_model,
            worker=self.config.llm.worker_model,
            shared_memory=True,
            knowledge_graph=True,
            message_bus=True,
            mcp_tools=len(self.mcp_client.list_tools()),
        )

    def _create_llm(self, model: str) -> BaseChatModel:
        """Create an LLM instance based on model name."""
        if "claude" in model or "anthropic" in model:
            return ChatAnthropic(
                model=model,
                temperature=self.config.llm.temperature,
                max_retries=self.config.llm.max_retries,
                timeout=self.config.llm.request_timeout,
            )
        else:
            return ChatOpenAI(
                model=model,
                temperature=self.config.llm.temperature,
                max_retries=self.config.llm.max_retries,
                timeout=self.config.llm.request_timeout,
            )

    async def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """Run the full multi-agent pipeline."""
        from mas.orchestrator.state import PipelineState

        initial_state: PipelineState = {
            "user_query": query,
            "user_context": context or {},
            "plan": [],
            "current_phase": "planning",
            "pending_tasks": [],
            "active_tasks": [],
            "completed_task_ids": [],
            "results": [],
            "messages": [],
            "task_ledger": "",
            "progress_ledger": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "should_replan": False,
            "final_output": {},
        }

        result = await self._compiled.ainvoke(initial_state)

        # Flush tracing
        self.tracing.flush()

        return result.get("final_output", {})
