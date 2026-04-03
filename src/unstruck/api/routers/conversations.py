"""Conversation CRUD endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateRequest(BaseModel):
    title: str = "New Chat"


class MessageRequest(BaseModel):
    role: str
    content: str


def _conv_dir(request: Request) -> Path:
    d = request.app.state.platform.config.data_dir / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("")
async def list_conversations(request: Request):
    conv_dir = _conv_dir(request)
    conversations = []
    for path in sorted(conv_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            conversations.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            pass
    return conversations


@router.post("")
async def create_conversation(request: Request, body: CreateRequest):
    conv_dir = _conv_dir(request)
    now = datetime.now(timezone.utc).isoformat()
    conv = {
        "id": uuid.uuid4().hex[:12],
        "title": body.title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _write_conv(conv_dir, conv)
    return conv


@router.get("/{conv_id}")
async def get_conversation(request: Request, conv_id: str):
    conv = _read_conv(_conv_dir(request), conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/{conv_id}")
async def delete_conversation(request: Request, conv_id: str):
    path = _conv_dir(request) / f"{conv_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")
    path.unlink()
    return {"deleted": True}


@router.post("/{conv_id}/messages")
async def add_message(request: Request, conv_id: str, body: MessageRequest):
    conv_dir = _conv_dir(request)
    conv = _read_conv(conv_dir, conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg = {
        "id": uuid.uuid4().hex[:8],
        "role": body.role,
        "content": body.content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    conv["messages"].append(msg)
    conv["updated_at"] = msg["timestamp"]

    if conv["title"] == "New Chat" and body.role == "user":
        conv["title"] = body.content[:60] + ("..." if len(body.content) > 60 else "")

    _write_conv(conv_dir, conv)
    return msg


def _read_conv(conv_dir: Path, conv_id: str) -> dict | None:
    path = conv_dir / f"{conv_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_conv(conv_dir: Path, conv: dict) -> None:
    path = conv_dir / f"{conv['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(conv, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
