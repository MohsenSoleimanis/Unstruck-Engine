"""Tests for the hook system."""

import pytest

from unstruck.hooks import HookAction, HookEvent, HookManager, HookResult


@pytest.fixture
def hooks():
    return HookManager()


@pytest.mark.asyncio
async def test_fire_with_no_handlers(hooks):
    result = await hooks.fire(HookEvent.PRE_LLM_CALL, {"prompt": "hello"})
    assert result.action == HookAction.ALLOW


@pytest.mark.asyncio
async def test_allow_handler(hooks):
    async def allow_all(event, context):
        return HookResult.allow()

    hooks.register(HookEvent.PRE_TOOL_USE, allow_all)
    result = await hooks.fire(HookEvent.PRE_TOOL_USE, {"tool": "fs_read"})
    assert result.action == HookAction.ALLOW


@pytest.mark.asyncio
async def test_block_handler(hooks):
    async def block_writes(event, context):
        if context.get("tool") == "fs_write":
            return HookResult.block("Write operations not allowed")
        return HookResult.allow()

    hooks.register(HookEvent.PRE_TOOL_USE, block_writes)

    result = await hooks.fire(HookEvent.PRE_TOOL_USE, {"tool": "fs_write"})
    assert result.action == HookAction.BLOCK
    assert "not allowed" in result.reason

    result = await hooks.fire(HookEvent.PRE_TOOL_USE, {"tool": "fs_read"})
    assert result.action == HookAction.ALLOW


@pytest.mark.asyncio
async def test_modify_handler(hooks):
    async def add_metadata(event, context):
        return HookResult.modify({"extra": "added_by_hook"})

    hooks.register(HookEvent.POST_LLM_CALL, add_metadata)
    result = await hooks.fire(HookEvent.POST_LLM_CALL, {"response": "hello"})
    assert result.action == HookAction.MODIFY
    assert result.data["extra"] == "added_by_hook"
    assert result.data["response"] == "hello"


@pytest.mark.asyncio
async def test_handler_order_preserved(hooks):
    """Handlers run in registration order. BLOCK stops the chain."""
    call_order = []

    async def first(event, context):
        call_order.append("first")
        return HookResult.allow()

    async def second(event, context):
        call_order.append("second")
        return HookResult.block("stopped")

    async def third(event, context):
        call_order.append("third")
        return HookResult.allow()

    hooks.register(HookEvent.SESSION_START, first)
    hooks.register(HookEvent.SESSION_START, second)
    hooks.register(HookEvent.SESSION_START, third)

    result = await hooks.fire(HookEvent.SESSION_START, {})
    assert result.action == HookAction.BLOCK
    assert call_order == ["first", "second"]  # third never called


@pytest.mark.asyncio
async def test_modify_chains(hooks):
    """Multiple MODIFY handlers accumulate changes."""
    async def add_a(event, context):
        return HookResult.modify({"a": 1})

    async def add_b(event, context):
        assert context.get("a") == 1  # sees previous modification
        return HookResult.modify({"b": 2})

    hooks.register(HookEvent.PRE_LLM_CALL, add_a)
    hooks.register(HookEvent.PRE_LLM_CALL, add_b)

    result = await hooks.fire(HookEvent.PRE_LLM_CALL, {})
    assert result.action == HookAction.MODIFY
    assert result.data["a"] == 1
    assert result.data["b"] == 2


@pytest.mark.asyncio
async def test_handler_error_does_not_crash(hooks):
    """Hook errors are logged and skipped — never crash the pipeline."""
    async def buggy(event, context):
        raise RuntimeError("handler bug")

    async def healthy(event, context):
        return HookResult.modify({"healthy": True})

    hooks.register(HookEvent.POST_TOOL_USE, buggy)
    hooks.register(HookEvent.POST_TOOL_USE, healthy)

    result = await hooks.fire(HookEvent.POST_TOOL_USE, {})
    # Buggy handler is skipped, healthy handler runs
    assert result.action == HookAction.MODIFY
    assert result.data["healthy"] is True


def test_register_and_count(hooks):
    async def handler(event, context):
        return HookResult.allow()

    hooks.register(HookEvent.PRE_LLM_CALL, handler)
    hooks.register(HookEvent.POST_LLM_CALL, handler)

    assert hooks.handler_count(HookEvent.PRE_LLM_CALL) == 1
    assert hooks.handler_count(HookEvent.POST_LLM_CALL) == 1
    assert hooks.handler_count(HookEvent.PRE_TOOL_USE) == 0
    assert hooks.total_handlers == 2


def test_unregister(hooks):
    async def handler(event, context):
        return HookResult.allow()

    hooks.register(HookEvent.SESSION_START, handler)
    assert hooks.handler_count(HookEvent.SESSION_START) == 1

    hooks.unregister(HookEvent.SESSION_START, handler)
    assert hooks.handler_count(HookEvent.SESSION_START) == 0


@pytest.mark.asyncio
async def test_all_events_exist(hooks):
    """Every HookEvent can be fired without error."""
    for event in HookEvent:
        result = await hooks.fire(event, {})
        assert result.action == HookAction.ALLOW
