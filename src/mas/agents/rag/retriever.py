"""Hybrid retrieval agent — semantic + keyword search with reranking."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

RERANKER_PROMPT = """Score each document's relevance to the query on a scale of 0-10.
Query: {query}

Documents:
{documents}

Output JSON array of objects with "index" and "score" fields, sorted by score descending.
"""


@registry.register
class RetrieverAgent(BaseAgent):
    """
    Hybrid retrieval agent combining semantic search with LLM reranking.

    Inspired by:
      - Protocol-intelligence v30's 3-stage retriever (BM25 + semantic + rerank)
      - RAG-Anything's modality-aware retrieval
    """

    agent_type = "retriever"
    description = "Retrieves relevant content using hybrid semantic + keyword search with LLM reranking"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        query = task.instruction
        content_items = task.context.get("content_items", [])
        top_k = task.context.get("top_k", 5)

        if not content_items:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"retrieved": [], "message": "No content items provided"},
            )

        # Stage 1: Keyword matching (simple BM25-like scoring)
        keyword_scores = self._keyword_score(query, content_items)

        # Stage 2: LLM reranking of top candidates
        candidates = sorted(keyword_scores, key=lambda x: x["score"], reverse=True)[:top_k * 2]

        if candidates:
            reranked = await self._llm_rerank(query, candidates, top_k)
        else:
            reranked = candidates[:top_k]

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "retrieved": reranked[:top_k],
                "total_candidates": len(content_items),
                "query": query,
            },
        )

    def _keyword_score(self, query: str, items: list[dict]) -> list[dict]:
        """Simple keyword frequency scoring."""
        query_terms = set(query.lower().split())
        scored = []
        for i, item in enumerate(items):
            text = str(item.get("content", item.get("data", ""))).lower()
            matches = sum(1 for term in query_terms if term in text)
            score = matches / max(len(query_terms), 1)
            scored.append({"index": i, "score": score, "item": item})
        return scored

    async def _llm_rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Use LLM to rerank candidates."""
        docs_text = "\n".join(
            f"[{i}] {str(c['item'].get('content', c['item'].get('data', '')))[:200]}"
            for i, c in enumerate(candidates)
        )

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a relevance scoring assistant. Output valid JSON only."),
            HumanMessage(content=RERANKER_PROMPT.format(query=query, documents=docs_text)),
        ])

        try:
            raw = response.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            rankings = json.loads(raw)
            reranked = []
            for r in rankings[:top_k]:
                idx = r["index"]
                if 0 <= idx < len(candidates):
                    candidates[idx]["rerank_score"] = r["score"]
                    reranked.append(candidates[idx])
            return reranked
        except (json.JSONDecodeError, KeyError, IndexError):
            return candidates[:top_k]
