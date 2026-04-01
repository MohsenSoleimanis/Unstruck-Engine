"""Base agent interface — all agents inherit from this."""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.language_models import BaseChatModel

from mas.llmops.cost_tracker import CostTracker
from mas.memory.local import LocalMemory
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

if TYPE_CHECKING:
    from mas.a2a.bus import MessageBus
    from mas.memory.knowledge_graph import KnowledgeGraph
    from mas.memory.shared import SharedMemory
    from mas.tools.mcp_client import MCPToolClient

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Contract every agent must fulfill.

    Every agent gets access to:
      - LLM for reasoning
      - SharedMemory for cross-agent knowledge (vector store + task board)
      - KnowledgeGraph for entity/relationship queries
      - MessageBus for peer-to-peer A2A messaging
      - MCPToolClient for external tool/data access
      - LocalMemory for per-agent working cache
      - CostTracker for token/cost accounting
    """

    agent_type: str = "base"
    description: str = "Base agent"
    version: str = "0.1.0"

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        agent_id: str | None = None,
        cost_tracker: CostTracker | None = None,
        shared_memory: SharedMemory | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        message_bus: MessageBus | None = None,
        mcp_client: MCPToolClient | None = None,
    ):
        self.agent_id = agent_id or f"{self.agent_type}_{uuid.uuid4().hex[:6]}"
        self.llm = llm
        self.cost_tracker = cost_tracker
        self.shared_memory = shared_memory
        self.knowledge_graph = knowledge_graph
        self.message_bus = message_bus
        self.mcp_client = mcp_client
        self.local_memory = LocalMemory(namespace=self.agent_id)
        self.logger = logger.bind(agent_id=self.agent_id, agent_type=self.agent_type)

        # Register with message bus if available
        if self.message_bus:
            self.message_bus.register_agent(self.agent_id, self.agent_type)

    @abstractmethod
    async def execute(self, task: Task) -> AgentResult:
        """Run the agent's core logic on a task. Must be implemented by subclasses."""

    async def run(self, task: Task) -> AgentResult:
        """Entry point — wraps execute() with logging, timing, and cost tracking."""
        self.logger.info("agent.start", task_id=task.id, instruction=task.instruction[:100])
        start = time.perf_counter()

        try:
            result = await self.execute(task)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result.duration_ms = elapsed_ms

            if self.cost_tracker:
                self.cost_tracker.record(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task_id=task.id,
                    token_usage=result.token_usage,
                    cost_usd=result.cost_usd,
                    model=getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "unknown"),
                )

            # Store result in shared memory so other agents can access it
            if self.shared_memory and result.status == ResultStatus.SUCCESS:
                self.shared_memory.store_result(
                    task_id=task.id,
                    agent_type=self.agent_type,
                    content=json.dumps(result.output, default=str)[:5000],
                )

            self.logger.info(
                "agent.done",
                task_id=task.id,
                status=result.status,
                duration_ms=elapsed_ms,
                tokens=result.token_usage,
            )
            return result

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self.logger.error("agent.error", task_id=task.id, error=str(e))
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.FAILED,
                errors=[str(e)],
                duration_ms=elapsed_ms,
            )

    # --- Shared memory helpers (available to all agents) ---

    def memory_store(self, key: str, text: str, metadata: dict | None = None) -> None:
        """Store something in shared memory for other agents to find."""
        if self.shared_memory:
            meta = {"agent_type": self.agent_type, "agent_id": self.agent_id}
            if metadata:
                meta.update(metadata)
            self.shared_memory.store(doc_id=f"{self.agent_id}_{key}", text=text, metadata=meta)

    def memory_search(self, query: str, n: int = 5, where: dict | None = None) -> list[dict]:
        """Search shared memory for relevant content from any agent."""
        if self.shared_memory:
            return self.shared_memory.search(query, n_results=n, where=where)
        return []

    def board_post(self, key: str, value: Any) -> None:
        """Post to the shared task board."""
        if self.shared_memory:
            self.shared_memory.post_to_board(key, value)

    def board_read(self, key: str, default: Any = None) -> Any:
        """Read from the shared task board."""
        if self.shared_memory:
            return self.shared_memory.read_board(key, default)
        return default

    # --- Knowledge graph helpers ---

    def kg_add_entity(self, entity_id: str, entity_type: str, props: dict | None = None) -> None:
        """Add an entity to the shared knowledge graph."""
        if self.knowledge_graph:
            self.knowledge_graph.add_entity(entity_id, entity_type, props)

    def kg_add_relationship(self, source: str, target: str, relation: str, props: dict | None = None) -> None:
        """Add a relationship to the shared knowledge graph."""
        if self.knowledge_graph:
            self.knowledge_graph.add_relationship(source, target, relation, props)

    def kg_search(self, entity_type: str | None = None, **filters: Any) -> list[dict]:
        """Search entities in the knowledge graph."""
        if self.knowledge_graph:
            return self.knowledge_graph.search_entities(entity_type, **filters)
        return []

    def kg_neighbors(self, entity_id: str, relation: str | None = None) -> list[dict]:
        """Get neighbors of an entity in the knowledge graph."""
        if self.knowledge_graph:
            return self.knowledge_graph.get_neighbors(entity_id, relation)
        return []

    # --- A2A messaging helpers ---

    async def send_message(self, receiver_type: str, content: str, data: dict | None = None) -> None:
        """Send a message to another agent via the A2A bus."""
        if self.message_bus:
            from mas.schemas.messages import AgentMessage, MessageRole

            msg = AgentMessage(
                sender=self.agent_id,
                receiver=receiver_type,
                role=MessageRole.AGENT,
                content=content,
                data=data or {},
            )
            await self.message_bus.send(msg)

    async def receive_messages(self) -> list[dict]:
        """Receive pending messages from the A2A bus."""
        if self.message_bus:
            return await self.message_bus.receive(self.agent_id)
        return []

    # --- MCP tool helpers ---

    async def call_tool(self, tool_name: str, arguments: dict | None = None) -> Any:
        """Call an external tool via MCP."""
        if self.mcp_client:
            return await self.mcp_client.call_tool(tool_name, arguments or {})
        raise RuntimeError(f"No MCP client available to call tool '{tool_name}'")

    def list_tools(self) -> list[dict]:
        """List available MCP tools."""
        if self.mcp_client:
            return self.mcp_client.list_tools()
        return []

    def get_capabilities(self) -> dict[str, Any]:
        """Return agent card (A2A-inspired) for dynamic discovery."""
        return {
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "description": self.description,
            "version": self.version,
            "has_shared_memory": self.shared_memory is not None,
            "has_knowledge_graph": self.knowledge_graph is not None,
            "has_message_bus": self.message_bus is not None,
            "has_mcp": self.mcp_client is not None,
        }
