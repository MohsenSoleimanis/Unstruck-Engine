"""Context Engine — governs every LLM call in the system.

No agent calls an LLM directly. They go through this engine, which:
  1. Ensures documents are ingested (calls rag_ingest if needed)
  2. Retrieves relevant context (calls rag_query transparently)
  3. Checks token budget (PreLLMCall hook can BLOCK if over ceiling)
  4. Assembles the prompt (system + retrieved context + history + instructions)
  5. Truncates to fit the token budget
  6. Makes the LLM call
  7. Extracts citations and tracks cost (PostLLMCall hook)
  8. Returns the response

Agents never know about RAG. The Context Engine handles all retrieval
transparently. This is where context engineering lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from unstruck.config import Config
from unstruck.context.budget import TokenBudget
from unstruck.context.result import ContextEngineResult
from unstruck.context.tokens import count_tokens, truncate_to_tokens
from unstruck.hooks import HookAction, HookEvent, HookManager, HookResult

if TYPE_CHECKING:
    from unstruck.memory.session import Session
    from unstruck.tools.registry import ToolRegistry

logger = structlog.get_logger()


class ContextEngine:
    """
    Central engine for all LLM interactions and context management.

    Handles:
      - Document ingestion (transparent to agents)
      - Context retrieval from RAG-Anything (transparent to agents)
      - Token budget enforcement
      - Prompt assembly and truncation
      - Hook firing (PreLLMCall, PostLLMCall)
      - Cost tracking
    """

    def __init__(
        self,
        config: Config,
        hooks: HookManager,
        budget: TokenBudget,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._config = config
        self._hooks = hooks
        self._budget = budget
        self._tool_registry = tool_registry
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def set_tool_registry(self, registry: ToolRegistry) -> None:
        """Set the tool registry (called during bootstrap after both are created)."""
        self._tool_registry = registry

    # ── Document Ingestion (transparent to agents) ──────────────

    async def ensure_ingested(self, file_path: str, session: Session | None = None) -> bool:
        """
        Ensure a document is ingested into RAG-Anything.

        Checks session first — if already ingested, skips.
        Calls rag_ingest tool if needed.
        Returns True if document is ready for querying.
        """
        if not file_path:
            return False

        # Check session — already ingested?
        if session and session.has_document(file_path):
            logger.debug("context_engine.already_ingested", file_path=file_path)
            return True

        # Call rag_ingest tool
        if not self._tool_registry or not self._tool_registry.has("rag_ingest"):
            logger.warning("context_engine.no_rag_ingest_tool")
            return False

        logger.info("context_engine.ingesting", file_path=file_path)
        result = await self._tool_registry.call("rag_ingest", file_path=file_path)

        if result.get("indexed"):
            if session:
                session.register_document(file_path, result.get("doc_id", file_path))
                session.save()
            logger.info("context_engine.ingested", file_path=file_path, doc_id=result.get("doc_id"))
            return True

        logger.warning("context_engine.ingest_failed", file_path=file_path, error=result.get("error"))
        return False

    # ── Context Retrieval (transparent to agents) ───────────────

    async def retrieve(self, query: str) -> str:
        """
        Retrieve relevant context for a query from RAG-Anything.

        Returns the retrieved text, or empty string if unavailable.
        """
        if not self._tool_registry or not self._tool_registry.has("rag_query"):
            return ""

        result = await self._tool_registry.call("rag_query", query=query)
        response = result.get("response", "")

        if response and not result.get("error"):
            logger.debug("context_engine.retrieved", query=query[:60], response_len=len(response))
            return response

        if result.get("error"):
            logger.debug("context_engine.retrieve_failed", error=result["error"])

        return ""

    # ── LLM Call (the main entry point for agents) ──────────────

    async def call(
        self,
        llm: BaseChatModel,
        *,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
        retrieve_for: str = "",
        agent_id: str = "unknown",
        model_name: str = "unknown",
    ) -> ContextEngineResult:
        """
        Make an LLM call with full context management.

        If retrieve_for is set and no explicit context provided,
        auto-retrieves from RAG-Anything. Agents never know about RAG.
        """

        # --- 1. Budget check ---
        allocated = self._budget.allocate(agent_id)
        if allocated <= 0:
            return ContextEngineResult(
                text="",
                blocked=True,
                block_reason="Token budget exhausted",
            )

        # --- 2. Auto-retrieve context if requested ---
        if retrieve_for and not context:
            context = await self.retrieve(retrieve_for)

        # --- 3. Assemble the prompt ---
        assembled_context = self._assemble(context, agent_id)

        full_user_prompt = user_prompt
        if assembled_context:
            full_user_prompt = f"{assembled_context}\n\n---\n\n{user_prompt}"

        # --- 4. Fire PreLLMCall hook ---
        hook_context = {
            "agent_id": agent_id,
            "model": model_name,
            "system_prompt": system_prompt,
            "user_prompt": full_user_prompt,
            "token_budget": allocated,
            "budget_utilization": self._budget.utilization,
        }
        pre_result = await self._hooks.fire(HookEvent.PRE_LLM_CALL, hook_context)

        if pre_result.action == HookAction.BLOCK:
            return ContextEngineResult(
                text="",
                blocked=True,
                block_reason=pre_result.reason,
            )

        if pre_result.action == HookAction.MODIFY and pre_result.data:
            cached = pre_result.data.get("_cached_response")
            if cached is not None:
                return ContextEngineResult(text=cached)
            system_prompt = pre_result.data.get("system_prompt", system_prompt)
            full_user_prompt = pre_result.data.get("user_prompt", full_user_prompt)

        # --- 5. Build messages and call LLM ---
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # --- 6. Extract token usage ---
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            input_tokens = count_tokens(system_prompt + full_user_prompt)
            output_tokens = count_tokens(response_text)

        total_tokens = input_tokens + output_tokens

        # --- 7. Record in budget ---
        self._budget.record(agent_id, total_tokens)
        self._call_count += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        # --- 8. Calculate cost ---
        inp_price, out_price = self._config.get_model_pricing(model_name)
        cost = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000

        # --- 9. Fire PostLLMCall hook ---
        post_context = {
            "agent_id": agent_id,
            "model": model_name,
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        }
        await self._hooks.fire(HookEvent.POST_LLM_CALL, post_context)

        return ContextEngineResult(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _assemble(self, context: str, agent_id: str) -> str:
        """Assemble and truncate context to fit the budget."""
        if not context:
            return ""

        max_context_tokens = self._budget.context_limit
        context_tokens = count_tokens(context)

        if context_tokens <= max_context_tokens:
            return context

        truncated = truncate_to_tokens(context, max_context_tokens)
        logger.debug(
            "context_engine.truncated",
            agent_id=agent_id,
            original_tokens=context_tokens,
            truncated_to=max_context_tokens,
        )
        return truncated

    async def compress(self, text: str, llm: BaseChatModel, max_tokens: int = 20000) -> str:
        """Compress text by summarizing it via LLM."""
        pre = await self._hooks.fire(HookEvent.PRE_COMPRESS, {"text": text})
        if pre.action == HookAction.MODIFY and pre.data:
            text = pre.data.get("text", text)

        system = "Summarize the following text concisely, preserving all key facts, numbers, names, and relationships."
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=truncate_to_tokens(text, max_tokens)),
        ])

        return response.content if hasattr(response, "content") else str(response)

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "budget": self._budget.to_dict(),
        }
