"""Conversation CRUD router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _get_store():
    from mas.api.server import app
    return app.state.conversation_store


class CreateRequest(BaseModel):
    title: str = "New Chat"


class UpdateRequest(BaseModel):
    title: str | None = None


class MessageRequest(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = {}


@router.get("")
async def list_conversations():
    return _get_store().list()


@router.post("")
async def create_conversation(request: CreateRequest):
    return _get_store().create(title=request.title)


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    data = _get_store().get(conversation_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return data


@router.put("/{conversation_id}")
async def update_conversation(conversation_id: str, request: UpdateRequest):
    updates = request.model_dump(exclude_none=True)
    data = _get_store().update(conversation_id, **updates)
    if data is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return data


@router.post("/{conversation_id}/messages")
async def add_message(conversation_id: str, request: MessageRequest):
    msg = _get_store().add_message(
        conversation_id, request.role, request.content, request.metadata
    )
    if msg is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return msg


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if not _get_store().delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}
