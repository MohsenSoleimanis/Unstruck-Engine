"""Context Engine — governs every LLM call in the system.

No agent calls an LLM directly. They go through this engine, which:
  1. Checks token budget (PreLLMCall hook can BLOCK if over ceiling)
  2. Assembles the prompt (system + context + history + instructions)
  3. Truncates to fit the token budget
  4. Makes the LLM call
  5. Extracts citations and tracks cost (PostLLMCall hook)
  6. Returns the response

This is where context engineering lives — not scattered across agents.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from unstruck.config import Config
from unstruck.context.budget import TokenBudget
from unstruck.context.result import ContextEngineResult
from unstruck.context.tokens import count_tokens, truncate_to_tokens
from unstruck.hooks import HookAction, HookEvent, HookManager, HookResult

logger = structlog.get_logger()


class ContextEngine:
    """
    Central engine for all LLM interactions.

    Every agent calls context_engine.call() instead of llm.ainvoke().
    The engine handles budget, assembly, truncation, hooks, and tracking.
    """

    def __init__(
        self,
        config: Config,
        hooks: HookManager,
        budget: TokenBudget,
    ) -> None:
        self._config = config
        self._hooks = hooks
        self._budget = budget
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    async def call(
        self,
        llm: BaseChatModel,
        *,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
        agent_id: str = "unknown",
        model_name: str = "unknown",
        max_response_tokens: int | None = None,
    ) -> ContextEngineResult:
        """
        Make an LLM call with full context management.

        Args:
            llm: The LangChain LLM instance.
            system_prompt: System/role instructions.
            user_prompt: The user's query or agent's task instruction.
            context: Retrieved context to include (from RAG, memory, etc.)
            agent_id: Who is making this call (for budget tracking).
            model_name: Which model (for cost calculation).
            max_response_tokens: Override max tokens for the response.

        Returns:
            ContextEngineResult with the response text, token usage, and cost.
        """

        # --- 1. Budget check ---
        allocated = self._budget.allocate(agent_id)
        if allocated <= 0:
            return ContextEngineResult(
                text="",
                blocked=True,
                block_reason="Token budget exhausted",
            )

        # --- 2. Assemble the prompt ---
        assembled_context = self._assemble(context, agent_id)

        full_user_prompt = user_prompt
        if assembled_context:
            full_user_prompt = f"{assembled_context}\n\n---\n\n{user_prompt}"

        # --- 3. Fire PreLLMCall hook ---
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
            system_prompt = pre_result.data.get("system_prompt", system_prompt)
            full_user_prompt = pre_result.data.get("user_prompt", full_user_prompt)

        # --- 4. Build messages and call LLM ---
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # --- 5. Extract token usage ---
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            # Estimate from text
            input_tokens = count_tokens(system_prompt + full_user_prompt)
            output_tokens = count_tokens(response_text)

        total_tokens = input_tokens + output_tokens

        # --- 6. Record in budget ---
        self._budget.record(agent_id, total_tokens)
        self._call_count += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        # --- 7. Calculate cost ---
        inp_price, out_price = self._config.get_model_pricing(model_name)
        cost = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000

        # --- 8. Fire PostLLMCall hook ---
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

        # Truncate context to fit budget
        truncated = truncate_to_tokens(context, max_context_tokens)
        logger.debug(
            "context_engine.truncated",
            agent_id=agent_id,
            original_tokens=context_tokens,
            truncated_to=max_context_tokens,
        )
        return truncated

    async def compress(self, text: str, llm: BaseChatModel, max_tokens: int = 20000) -> str:
        """
        Compress text by summarizing it via LLM.

        Used when context window is near capacity. Fires PreCompress hook
        so handlers can mark critical info as "do not compress".
        """
        # Fire PreCompress hook
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
