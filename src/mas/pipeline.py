"""Main pipeline — ties together orchestrator, agents, memory, and LLMOps."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from mas.agents.protocol import ExplorerAgent, ExtractorAgent, ReviewerAgent
from mas.agents.rag import MultimodalAgent, ParserAgent, RetrieverAgent
from mas.agents.registry import registry
from mas.agents.research import ResearcherAgent, SynthesizerAgent
from mas.config import MASConfig, get_config
from mas.llmops.cost_tracker import CostTracker
from mas.llmops.evaluation import EvaluationEngine
from mas.llmops.monitoring import HealthMonitor
from mas.llmops.tracing import TracingManager
from mas.memory.knowledge_graph import KnowledgeGraph
from mas.memory.shared import SharedMemory
from mas.orchestrator.graph import build_orchestrator_graph

logger = structlog.get_logger()


class MASPipeline:
    """
    Production pipeline — single entry point for the multi-agent system.

    Wires together:
      - LangGraph orchestrator (plan -> execute -> review -> synthesize)
      - Agent registry with all specialized agents
      - Shared + graph memory
      - LLMOps (tracing, cost tracking, evaluation, monitoring)
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
        self.knowledge_graph = KnowledgeGraph()

        # LLMs
        self.orchestrator_llm = self._create_llm(self.config.llm.orchestrator_model)
        self.worker_llm = self._create_llm(self.config.llm.worker_model)

        # Registry (agents auto-register via decorators)
        self.registry = registry

        # Build orchestrator graph
        self._graph = build_orchestrator_graph(
            orchestrator_llm=self.orchestrator_llm,
            worker_llm=self.worker_llm,
            registry=self.registry,
            shared_memory=self.shared_memory,
            cost_tracker=self.cost_tracker,
            tracing=self.tracing,
            monitor=self.monitor,
        )
        self._compiled = self._graph.compile()

        logger.info(
            "pipeline.initialized",
            agents=len(self.registry.list_agents()),
            orchestrator=self.config.llm.orchestrator_model,
            worker=self.config.llm.worker_model,
        )

    def _create_llm(self, model: str):
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
