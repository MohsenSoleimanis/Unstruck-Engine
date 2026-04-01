"""Analyst agent — reasons over retrieved context to answer any question."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task
from mas.utils.parsing import extract_json, extract_token_usage

ANALYST_PROMPT = """You are an expert analyst. Answer the question using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context.
2. If the context doesn't contain enough information, say so explicitly.
3. Cite sources by referencing page numbers, chunk IDs, or entity names.
4. Structure your answer clearly.
5. Quantify confidence: high (directly stated), medium (inferred), low (speculative).

Context:
{context}

Question: {question}

Output JSON:
{{
  "answer": "your comprehensive answer",
  "key_points": ["point 1", "point 2"],
  "citations": [{{"text": "quoted source", "source": "page/chunk ref"}}],
  "confidence": "high|medium|low",
  "limitations": ["what the context doesn't cover"]
}}
"""


@registry.register
class AnalystAgent(BaseAgent):
    """
    Reasons over retrieved context to answer questions.

    Domain-agnostic: works with any content type that has been
    processed through the RAG pipeline (text, tables, images, structured data).
    """

    agent_type = "analyst"
    description = "Answers questions by reasoning over retrieved context from any data source"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        question = task.instruction
        pipeline_ctx = self.get_pipeline_context(task)

        # Build context from PipelineContext (typed blackboard)
        context_parts = []

        # RAG-Anything response
        if pipeline_ctx.rag_response:
            context_parts.append(f"[RAG-Anything]\n{pipeline_ctx.rag_response}")

        # Retrieved items
        for item in pipeline_ctx.retrieved:
            context_parts.append(f"[{item.source}] {item.text}")

        # Text aggregate from ingestion
        if pipeline_ctx.text_aggregate:
            context_parts.append(f"[Document Text]\n{pipeline_ctx.text_aggregate[:8000]}")

        # Fallback: raw task.context (backward compat)
        if not context_parts:
            for item in task.context.get("retrieved", []):
                context_parts.append(str(item.get("text", item)))
            extra = task.context.get("extra_context", "")
            if extra:
                context_parts.append(extra)

        context = "\n\n".join(context_parts)

        if not context.strip():
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"answer": "No context available to answer the question.", "confidence": "low"},
            )

        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert analyst. Output valid JSON only."),
            HumanMessage(content=ANALYST_PROMPT.format(context=context[:12000], question=question)),
        ])

        try:
            output = extract_json(response.content)
        except json.JSONDecodeError:
            output = {"answer": response.content, "confidence": "medium"}

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output=output,
            token_usage=extract_token_usage(response),
        )
