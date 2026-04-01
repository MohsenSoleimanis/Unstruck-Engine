"""Session management — persists state across messages in a conversation.

The missing piece: when a user uploads a PDF and asks a question, then
asks a follow-up, the session retains the ingested document, pipeline
context, and message history so the second query doesn't start from zero.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from mas.schemas.context import PipelineContext

logger = structlog.get_logger()


class Session:
    """
    Conversation session — persists between messages.

    Stores:
      - pipeline_context: the typed blackboard (survives across messages)
      - ingested_docs: file_path → doc_id (skip re-ingestion on follow-ups)
      - message_history: previous Q&A pairs (for conversation-aware prompts)
    """

    def __init__(self, session_id: str, storage_dir: Path) -> None:
        self.session_id = session_id
        self._path = storage_dir / f"{session_id}.session.json"
        self.pipeline_context = PipelineContext()
        self.ingested_docs: dict[str, str] = {}  # file_path → doc_id
        self.message_history: list[dict[str, str]] = []

        # Load existing session if it exists
        if self._path.exists():
            self._load()

    def has_document(self, file_path: str) -> bool:
        """Check if a document was already ingested in this session."""
        return file_path in self.ingested_docs

    def register_document(self, file_path: str, doc_id: str = "") -> None:
        """Mark a document as ingested."""
        self.ingested_docs[file_path] = doc_id or file_path
        logger.info("session.doc_registered", session_id=self.session_id, file_path=file_path)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history (for conversation-aware prompts)."""
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_recent_history(self, max_messages: int = 10) -> list[dict[str, str]]:
        """Get recent conversation history for context injection."""
        return self.message_history[-max_messages:]

    def get_history_text(self, max_messages: int = 6) -> str:
        """Format recent history as text for LLM prompts."""
        recent = self.get_recent_history(max_messages)
        if not recent:
            return ""
        parts = []
        for msg in recent:
            role = msg["role"].upper()
            parts.append(f"{role}: {msg['content'][:500]}")
        return "\n\n".join(parts)

    def update_context(self, ctx: PipelineContext) -> None:
        """Update the session's pipeline context."""
        self.pipeline_context = ctx

    def save(self) -> None:
        """Persist session to disk (atomic write)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "pipeline_context": self.pipeline_context.model_dump(),
            "ingested_docs": self.ingested_docs,
            "message_history": self.message_history[-50:],  # Cap at 50 messages
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.replace(self._path)
        logger.debug("session.saved", session_id=self.session_id)

    def _load(self) -> None:
        """Load session from disk."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.pipeline_context = PipelineContext.model_validate(data.get("pipeline_context", {}))
            self.ingested_docs = data.get("ingested_docs", {})
            self.message_history = data.get("message_history", [])
            logger.debug("session.loaded", session_id=self.session_id, docs=len(self.ingested_docs))
        except Exception as e:
            logger.warning("session.load_failed", session_id=self.session_id, error=str(e))


class SessionManager:
    """Manages sessions by conversation ID."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        """Get or create a session."""
        if session_id not in self._cache:
            self._cache[session_id] = Session(session_id, self.storage_dir)
        return self._cache[session_id]

    def save(self, session_id: str) -> None:
        """Persist a session."""
        if session_id in self._cache:
            self._cache[session_id].save()

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        self._cache.pop(session_id, None)
        path = self.storage_dir / f"{session_id}.session.json"
        if path.exists():
            path.unlink()
