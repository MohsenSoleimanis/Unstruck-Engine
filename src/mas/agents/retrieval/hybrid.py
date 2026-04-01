"""Hybrid retriever agent — combines vector search + graph traversal + reranking.

The core RAG query path: vector similarity → graph expansion → LLM rerank.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

RERANK_PROMPT = """Score each document's relevance to the query (0-10).

Query: {query}

Documents:
{documents}

Output JSON array: [{{"index": 0, "score": 8, "reason": "..."}}]
Only output the JSON array, nothing else."""


@registry.register
class HybridRetrieverAgent(BaseAgent):
    """
    Hybrid retrieval: vector similarity + knowledge graph expansion + LLM reranking.

    Pipeline:
      1. Vector search in ChromaDB (semantic similarity)
      2. Graph expansion: find related entities and their chunks
      3. LLM reranking of combined candidates
      4. Return top-k with relevance scores
    """

    agent_type = "hybrid_retriever"
    description = "Hybrid retrieval combining vector search, knowledge graph traversal, and LLM reranking"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        query = task.instruction
        collection = task.context.get("collection", "default")
        top_k = task.context.get("top_k", 10)
        graph_nodes = task.context.get("graph_nodes", [])
        graph_edges = task.context.get("graph_edges", [])

        # Stage 1: Vector search
        vector_results = self._vector_search(query, collection, top_k * 2)

        # Stage 2: Graph expansion (if graph context available)
        graph_results = []
        if graph_nodes:
            graph_results = self._graph_expand(query, graph_nodes, graph_edges)

        # Merge candidates
        candidates = vector_results + graph_results

        if not candidates:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=ResultStatus.PARTIAL,
                output={"retrieved": [], "message": "No results found"},
            )

        # Stage 3: LLM reranking
        reranked = await self._rerank(query, candidates, top_k)

        token_usage = {}
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output={
                "retrieved": reranked,
                "total_candidates": len(candidates),
                "vector_hits": len(vector_results),
                "graph_hits": len(graph_results),
            },
            token_usage=token_usage,
        )

    def _vector_search(self, query: str, collection: str, n: int) -> list[dict[str, Any]]:
        """Search ChromaDB for similar chunks."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path="./data/chroma")
            coll = client.get_or_create_collection(name=collection)
            results = coll.query(query_texts=[query], n_results=min(n, coll.count() or 1))

            hits = []
            for i in range(len(results["ids"][0])):
                hits.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "source": "vector",
                })
            return hits
        except Exception:
            return []

    def _graph_expand(self, query: str, nodes: list[dict], edges: list[dict]) -> list[dict[str, Any]]:
        """Find graph nodes relevant to query and expand neighbors."""
        query_terms = set(query.lower().split())
        relevant = []

        for node in nodes:
            name = str(node.get("name", "")).lower()
            if any(term in name for term in query_terms):
                relevant.append({
                    "id": node.get("id", ""),
                    "text": f"[{node.get('type', '?')}] {node.get('name', '')}: {json.dumps(node.get('properties', {}))}",
                    "metadata": {"type": "graph_entity", "entity_type": node.get("type")},
                    "source": "graph",
                })

        return relevant[:20]

    async def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """LLM-based reranking of candidates."""
        if len(candidates) <= top_k:
            return candidates

        docs_text = "\n".join(
            f"[{i}] {c.get('text', '')[:300]}" for i, c in enumerate(candidates[:30])
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a relevance scoring assistant. Output valid JSON array only."),
                HumanMessage(content=RERANK_PROMPT.format(query=query, documents=docs_text)),
            ])

            raw = response.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            rankings = json.loads(raw)
            reranked = []
            for r in sorted(rankings, key=lambda x: x.get("score", 0), reverse=True)[:top_k]:
                idx = r["index"]
                if 0 <= idx < len(candidates):
                    candidates[idx]["rerank_score"] = r.get("score", 0)
                    reranked.append(candidates[idx])
            return reranked

        except Exception:
            return candidates[:top_k]
