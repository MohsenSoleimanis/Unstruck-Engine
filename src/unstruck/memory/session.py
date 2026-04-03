"""Tier 3: Session store — persists state across messages in a conversation.

When a user uploads a PDF and asks a question, then asks a follow-up:
  - Session remembers which documents are ingested (skip re-ingestion)
  - Session carries conversation history (follow-up detection)
  - Session preserves pipeline context (retrieved info persists)

Stored as JSON files — one per conversation. Atomic writes (tmp + rename).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class Session:
    """
    A conversation session — persists between messages.

    Stores:
      - ingested_docs: file_path → doc_id (skip re-ingestion)
      - message_history: previous Q&A pairs (for context)
      - pipeline_state: any state the orchestrator wants to preserve
    """

    def __init__(self, session_id: str, storage_dir: Path) -> None:
        self.session_id = session_id
        self._path = storage_dir / f"{session_id}.json"
        self.ingested_docs: dict[str, str] = {}
        self.message_history: list[dict[str, str]] = []
        self.pipeline_state: dict[str, Any] = {}

        if self._path.exists():
            self._load()

    def has_document(self, file_path: str) -> bool:
        return file_path in self.ingested_docs

    def register_document(self, file_path: str, doc_id: str = "") -> None:
        self.ingested_docs[file_path] = doc_id or file_path

    def add_message(self, role: str, content: str) -> None:
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_recent_history(self, max_messages: int = 10) -> list[dict[str, str]]:
        return self.message_history[-max_messages:]

    def get_history_text(self, max_messages: int = 6) -> str:
        """Format recent history as text for LLM prompts."""
        recent = self.get_recent_history(max_messages)
        if not recent:
            return ""
        return "\n\n".join(
            f"{msg['role'].upper()}: {msg['content'][:500]}"
            for msg in recent
        )

    def update_state(self, updates: dict[str, Any]) -> None:
        """Merge updates into pipeline state."""
        self.pipeline_state.update(updates)

    def save(self) -> None:
        """Persist to disk with atomic write (tmp + rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "ingested_docs": self.ingested_docs,
            "message_history": self.message_history[-50:],  # Cap at 50
            "pipeline_state": self.pipeline_state,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.ingested_docs = data.get("ingested_docs", {})
            self.message_history = data.get("message_history", [])
            self.pipeline_state = data.get("pipeline_state", {})
        except Exception as e:
            logger.warning("session.load_failed", session_id=self.session_id, error=str(e))

    def to_dict(self) -> dict[str, Any]:
        """Export session data for the orchestrator."""
        return {
            "ingested_docs": self.ingested_docs,
            "message_history": self.get_recent_history(6),
            "pipeline_state": self.pipeline_state,
        }


class SessionManager:
    """Manages sessions by conversation ID."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._cache:
            self._cache[session_id] = Session(session_id, self.storage_dir)
        return self._cache[session_id]

    def save(self, session_id: str) -> None:
        if session_id in self._cache:
            self._cache[session_id].save()

    def delete(self, session_id: str) -> None:
        self._cache.pop(session_id, None)
        path = self.storage_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self.storage_dir.glob("*.json")]
