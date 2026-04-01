"""Main pipeline — ties together orchestrator, agents, memory, session, and LLMOps."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
load_dotenv()

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from mas.a2a.bus import MessageBus
# Core agents
from mas.agents.kg import KGBuilderAgent, KGQueryAgent  # noqa: F401
from mas.agents.reasoning import AnalystAgent, SynthesizerAgent  # noqa: F401
from mas.agents.registry import registry
# RAG-Anything agent
from mas.agents.rag import RAGAnythingAgent, RAGEngine  # noqa: F401
from mas.agents.rag.raganything_agent import set_rag_engine
# Legacy agents (fallback)
from mas.agents.ingestion import ChunkerAgent, ContentSeparatorAgent, IngestionAgent  # noqa: F401
from mas.agents.modal import ImageProcessor, TableProcessor, TextProcessor  # noqa: F401
from mas.agents.retrieval import EmbedderAgent, HybridRetrieverAgent  # noqa: F401
from mas.config import MASConfig, get_config
from mas.llmops.cost_tracker import CostTracker
from mas.llmops.evaluation import EvaluationEngine
from mas.llmops.monitoring import HealthMonitor
from mas.llmops.tracing import TracingManager
from mas.memory.knowledge_graph import KnowledgeGraph
from mas.memory.shared import SharedMemory
from mas.orchestrator.graph import build_orchestrator_graph
from mas.session import SessionManager
from mas.tools.mcp_client import MCPToolClient

logger = structlog.get_logger()


class MASPipeline:
    """
    Production pipeline with session management.

    Key difference from before: the pipeline is session-aware.
    Each conversation gets a Session that persists document state,
    pipeline context, and message history between messages.
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

        # Memory (Layer 3 — persistent)
        self.shared_memory = SharedMemory(
            collection_name=self.config.memory.shared_collection,
            persist_dir=str(self.config.memory.chroma_persist_dir),
        )
        self.knowledge_graph = KnowledgeGraph(
            neo4j_uri=self.config.memory.neo4j_uri if self.config.memory.neo4j_uri != "bolt://localhost:7687" else None,
        )

        # Session manager (conversation persistence)
        self.session_manager = SessionManager(self.config.data_dir / "sessions")

        # A2A message bus
        self.message_bus = MessageBus()

        # MCP tool client
        self.mcp_client = MCPToolClient()
        self.mcp_client.register_builtin_tools()

        # RAG-Anything engine
        self.rag_engine = RAGEngine(
            working_dir=str(self.config.data_dir / "raganything"),
            llm_model=self.config.llm.worker_model,
            embedding_model=self.config.llm.embedding_model,
            embedding_dim=self.config.llm.embedding_dim,
        )
        set_rag_engine(self.rag_engine)

        # LLMs
        self.orchestrator_llm = self._create_llm(self.config.llm.orchestrator_model)
        self.worker_llm = self._create_llm(self.config.llm.worker_model)

        # Registry
        self.registry = registry

        # Build orchestrator graph
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
        )

    def _create_llm(self, model: str) -> BaseChatModel:
        if "claude" in model or "anthropic" in model:
            return ChatAnthropic(
                model=model,
                temperature=self.config.llm.temperature,
                max_retries=self.config.llm.max_retries,
                timeout=self.config.llm.request_timeout,
            )
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
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the pipeline — session-aware.

        If session_id is provided, loads existing session state so:
          - Already-ingested documents are not re-processed
          - Previous messages inform the current query
          - Pipeline context is carried forward
        """
        context = context or {}

        # Load or create session
        session = self.session_manager.get(session_id or "default")

        # Record user message in session
        session.add_message("user", query)

        # Build session data for the graph
        session_data = {
            "pipeline_context": session.pipeline_context.model_dump(),
            "ingested_docs": session.ingested_docs,
            "message_history": session.get_recent_history(6),
        }

        initial_state = {
            "user_query": query,
            "user_context": context,
            "session_id": session.session_id,
            "session_data": session_data,
            "session_update": {},
            "plan": [],
            "current_phase": "planning",
            "pending_tasks": [],
            "active_tasks": [],
            "completed_task_ids": [],
            "results": [],
            "messages": [],
            "task_ledger": {},
            "progress_ledger": {},
            "token_budget": {},
            "iteration": 0,
            "max_iterations": max_iterations,
            "should_replan": False,
            "final_output": {},
        }

        result = await self._compiled.ainvoke(initial_state)

        # Update session with pipeline results
        session_update = result.get("session_update", {})
        if "pipeline_context" in session_update:
            from mas.schemas.context import PipelineContext
            session.update_context(PipelineContext.model_validate(session_update["pipeline_context"]))
        if "ingested_docs" in session_update:
            for path, doc_id in session_update["ingested_docs"].items():
                session.register_document(path, doc_id)

        # Record assistant response in session
        final = result.get("final_output", {})
        analysis = final.get("analysis", {})
        answer = analysis.get("answer", "") if analysis else final.get("rag_response", "")
        if answer:
            session.add_message("assistant", answer[:2000])

        # Save session
        session.save()

        self.tracing.flush()
        return final
