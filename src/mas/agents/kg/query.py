"""Knowledge graph query agent — traverses the graph to answer questions.

Combines graph traversal with semantic search for hybrid retrieval.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.base import BaseAgent
from mas.agents.registry import registry
from mas.schemas.results import AgentResult, ResultStatus
from mas.schemas.tasks import Task

KG_QUERY_PROMPT = """You are a knowledge graph query expert.

Given a user question and a knowledge graph context (entities and relationships),
answer the question using the graph information.

Knowledge graph context:
{kg_context}

Question: {question}

Output JSON:
{{
  "answer": "your detailed answer",
  "relevant_entities": ["entity names used"],
  "reasoning_path": ["entity1 -[relation]-> entity2 -> ..."],
  "confidence": 0.0-1.0,
  "sources": ["modality and page references"]
}}
"""


@registry.register
class KGQueryAgent(BaseAgent):
    """
    Queries the knowledge graph to answer questions.

    Combines:
      - Graph traversal (follow relationships)
      - Entity-based retrieval (find relevant nodes)
      - LLM reasoning over graph context
    """

    agent_type = "kg_query"
    description = "Queries the knowledge graph to answer questions using graph traversal and reasoning"
    version = "0.1.0"

    async def execute(self, task: Task) -> AgentResult:
        question = task.instruction
        graph_nodes = task.context.get("nodes", [])
        graph_edges = task.context.get("edges", [])
        retrieved_chunks = task.context.get("retrieved_chunks", [])

        # Build context from graph and retrieved content
        kg_context = self._format_graph_context(graph_nodes, graph_edges)
        if retrieved_chunks:
            kg_context += "\n\nRetrieved text chunks:\n"
            for chunk in retrieved_chunks[:5]:
                kg_context += f"- {chunk.get('text', str(chunk))[:500]}\n"

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a knowledge graph reasoning expert. Output valid JSON only."),
            HumanMessage(content=KG_QUERY_PROMPT.format(kg_context=kg_context[:8000], question=question)),
        ])

        try:
            raw = response.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            output = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            output = {"answer": response.content, "confidence": 0.5}

        token_usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
            }

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=ResultStatus.SUCCESS,
            output=output,
            token_usage=token_usage,
        )

    def _format_graph_context(self, nodes: list[dict], edges: list[dict]) -> str:
        lines = ["Entities:"]
        for n in nodes[:50]:
            props = n.get("properties", {})
            props_str = f" ({', '.join(f'{k}={v}' for k, v in props.items())})" if props else ""
            lines.append(f"  - [{n.get('type', '?')}] {n.get('name', n.get('id', '?'))}{props_str}")

        lines.append("\nRelationships:")
        for e in edges[:50]:
            lines.append(f"  - {e.get('source', '?')} -[{e.get('relation', '?')}]-> {e.get('target', '?')}")

        return "\n".join(lines)
