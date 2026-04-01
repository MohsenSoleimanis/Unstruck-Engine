"""Conversation persistence — JSON file-based storage."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class ConversationStore:
    """
    Stores conversations as JSON files.

    Atomic writes (temp file + rename) prevent corruption.
    Each conversation: {id, title, created_at, updated_at, messages: [...]}
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        """List all conversations (metadata only, no messages)."""
        conversations = []
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                conversations.append({
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("storage.corrupt_file", path=str(path), error=str(e))
        return conversations

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Load a conversation with all messages."""
        path = self.directory / f"{conversation_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            return None

    def create(self, title: str = "New Chat") -> dict[str, Any]:
        """Create a new empty conversation."""
        now = datetime.now(timezone.utc).isoformat()
        conversation = {
            "id": uuid.uuid4().hex[:12],
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._write(conversation)
        return conversation

    def update(self, conversation_id: str, **updates: Any) -> dict[str, Any] | None:
        """Update conversation fields (title, messages, etc.)."""
        data = self.get(conversation_id)
        if data is None:
            return None
        data.update(updates)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(data)
        return data

    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> dict[str, Any] | None:
        """Append a message to a conversation."""
        data = self.get(conversation_id)
        if data is None:
            return None
        message = {
            "id": uuid.uuid4().hex[:8],
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        data["messages"].append(message)
        data["updated_at"] = message["timestamp"]
        # Auto-title from first user message
        if data["title"] == "New Chat" and role == "user":
            data["title"] = content[:60] + ("..." if len(content) > 60 else "")
        self._write(data)
        return message

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        path = self.directory / f"{conversation_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def _write(self, data: dict[str, Any]) -> None:
        """Atomic write: write to temp file, then rename."""
        path = self.directory / f"{data['id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
