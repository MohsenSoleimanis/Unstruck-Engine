"""Tests for memory layer — agent cache and session store."""

import time

import pytest

from unstruck.memory.cache import AgentCache
from unstruck.memory.session import Session, SessionManager
from unstruck.hooks import HookAction, HookEvent, HookManager


# ── Agent Cache ─────────────────────────────────────────────────

class TestAgentCache:
    def test_set_and_get(self):
        cache = AgentCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        cache = AgentCache()
        assert cache.get("missing") is None

    def test_ttl_expiration(self):
        from unstruck.memory.cache import _CacheEntry
        cache = AgentCache()
        # Insert an already-expired entry
        cache._store["key1"] = _CacheEntry(value="value1", expires_at=time.time() - 1)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = AgentCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_stats(self):
        cache = AgentCache()
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss

        stats = cache.stats
        assert stats["total_entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_llm_cache_via_hooks(self):
        cache = AgentCache()
        hooks = HookManager()
        cache.register_hooks(hooks)

        # First call — miss
        context = {
            "agent_id": "analyst_1",
            "system_prompt": "You are an analyst.",
            "user_prompt": "What is X?",
        }
        result = await hooks.fire(HookEvent.PRE_LLM_CALL, context)
        assert result.action == HookAction.ALLOW  # No cache hit

        # Simulate PostLLMCall storing the response
        post_context = {**context, "response": "X is a variable."}
        await hooks.fire(HookEvent.POST_LLM_CALL, post_context)

        # Second call — hit
        result = await hooks.fire(HookEvent.PRE_LLM_CALL, context)
        assert result.action == HookAction.MODIFY
        assert result.data["_cached_response"] == "X is a variable."

    @pytest.mark.asyncio
    async def test_tool_cache_via_hooks(self):
        cache = AgentCache()
        hooks = HookManager()
        cache.register_hooks(hooks)

        context = {
            "tool_name": "fs_read",
            "tool_args": {"path": "test.txt"},
        }

        # Miss
        result = await hooks.fire(HookEvent.PRE_TOOL_USE, context)
        assert result.action == HookAction.ALLOW

        # Store
        post_context = {**context, "result": {"content": "hello"}}
        await hooks.fire(HookEvent.POST_TOOL_USE, post_context)

        # Hit
        result = await hooks.fire(HookEvent.PRE_TOOL_USE, context)
        assert result.action == HookAction.MODIFY
        assert result.data["_cached_result"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_registers_four_hooks(self):
        cache = AgentCache()
        hooks = HookManager()
        cache.register_hooks(hooks)

        assert hooks.handler_count(HookEvent.PRE_LLM_CALL) == 1
        assert hooks.handler_count(HookEvent.POST_LLM_CALL) == 1
        assert hooks.handler_count(HookEvent.PRE_TOOL_USE) == 1
        assert hooks.handler_count(HookEvent.POST_TOOL_USE) == 1


# ── Session ─────────────────────────────────────────────────────

class TestSession:
    def test_create_session(self, tmp_path):
        session = Session("test_1", tmp_path)
        assert session.session_id == "test_1"
        assert session.ingested_docs == {}
        assert session.message_history == []

    def test_register_and_check_document(self, tmp_path):
        session = Session("test_1", tmp_path)
        assert session.has_document("doc.pdf") is False

        session.register_document("doc.pdf", "doc_123")
        assert session.has_document("doc.pdf") is True

    def test_add_and_get_messages(self, tmp_path):
        session = Session("test_1", tmp_path)
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        history = session.get_recent_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Hi there"

    def test_get_history_text(self, tmp_path):
        session = Session("test_1", tmp_path)
        session.add_message("user", "What is X?")
        session.add_message("assistant", "X is Y.")

        text = session.get_history_text()
        assert "USER: What is X?" in text
        assert "ASSISTANT: X is Y." in text

    def test_save_and_reload(self, tmp_path):
        # Create and save
        s1 = Session("test_1", tmp_path)
        s1.register_document("doc.pdf", "d1")
        s1.add_message("user", "Hello")
        s1.update_state({"key": "value"})
        s1.save()

        # Reload from disk
        s2 = Session("test_1", tmp_path)
        assert s2.has_document("doc.pdf")
        assert len(s2.message_history) == 1
        assert s2.pipeline_state["key"] == "value"

    def test_to_dict(self, tmp_path):
        session = Session("test_1", tmp_path)
        session.register_document("doc.pdf")
        session.add_message("user", "Hi")

        d = session.to_dict()
        assert "ingested_docs" in d
        assert "message_history" in d
        assert "pipeline_state" in d


# ── SessionManager ──────────────────────────────────────────────

class TestSessionManager:
    def test_get_creates_session(self, tmp_path):
        mgr = SessionManager(tmp_path)
        session = mgr.get("s1")
        assert session.session_id == "s1"

    def test_get_returns_same_instance(self, tmp_path):
        mgr = SessionManager(tmp_path)
        s1 = mgr.get("s1")
        s2 = mgr.get("s1")
        assert s1 is s2

    def test_save_and_list(self, tmp_path):
        mgr = SessionManager(tmp_path)
        s = mgr.get("s1")
        s.add_message("user", "test")
        mgr.save("s1")

        assert "s1" in mgr.list_sessions()

    def test_delete(self, tmp_path):
        mgr = SessionManager(tmp_path)
        s = mgr.get("s1")
        s.add_message("user", "test")
        mgr.save("s1")

        mgr.delete("s1")
        assert "s1" not in mgr.list_sessions()
