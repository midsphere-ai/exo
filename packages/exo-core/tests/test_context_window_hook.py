"""Tests for CONTEXT_WINDOW hook integration in _apply_context_windowing."""

from __future__ import annotations

import types
from typing import Any

from exo.agent import _apply_context_windowing, _ContextAction
from exo.context.hook import ContextWindowHook
from exo.context.info import ContextWindowInfo
from exo.hooks import HookManager, HookPoint
from exo.types import AssistantMessage, SystemMessage, UserMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(overflow: str = "hook", history_rounds: int = 20, limit: int = 20, **kw: Any):
    defaults = dict(
        overflow=overflow,
        history_rounds=history_rounds,
        limit=limit,
        summary_threshold=10,
        offload_threshold=50,
        keep_recent=5,
        token_pressure=0.8,
        token_budget_trigger=0.8,
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def _make_context(overflow: str = "hook", **kw: Any):
    return types.SimpleNamespace(config=_make_config(overflow=overflow, **kw))


def _build_msg_list(n_user: int = 5, n_assistant: int = 5, system: bool = True) -> list:
    """Build a message list with the requested number of user/assistant pairs."""
    msgs: list = []
    if system:
        msgs.append(SystemMessage(content="You are a helpful assistant."))
    for i in range(max(n_user, n_assistant)):
        if i < n_user:
            msgs.append(UserMessage(content=f"user message {i}"))
        if i < n_assistant:
            msgs.append(AssistantMessage(content=f"assistant message {i}"))
    return msgs


# ---------------------------------------------------------------------------
# 1. overflow="hook" fires CONTEXT_WINDOW hook
# ---------------------------------------------------------------------------


async def test_overflow_hook_fires_context_window_hook():
    called = []

    async def trim_hook(*, messages: list, info: ContextWindowInfo, **_: Any):
        called.append(True)
        # Keep system + last 3 non-system messages
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        system = [m for m in messages if isinstance(m, SystemMessage)]
        keep = non_system[-3:]
        messages.clear()
        messages.extend(system + keep)

    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, trim_hook)

    msg_list = _build_msg_list(n_user=5, n_assistant=5, system=True)
    original_len = len(msg_list)
    assert original_len == 11  # 1 system + 5 user + 5 assistant

    ctx = _make_context(overflow="hook")
    result, _actions = await _apply_context_windowing(msg_list, ctx, provider=None, hook_manager=hm)

    assert called == [True]
    # 1 system + 3 non-system kept
    assert len(result) == 4


# ---------------------------------------------------------------------------
# 2. overflow="hook" with no hooks is a no-op
# ---------------------------------------------------------------------------


async def test_overflow_hook_no_hooks_is_noop():
    msg_list = _build_msg_list(n_user=5, n_assistant=5, system=True)
    original_len = len(msg_list)

    ctx = _make_context(overflow="hook")
    # No hook_manager at all
    _result, _actions = await _apply_context_windowing(
        msg_list, ctx, provider=None, hook_manager=None
    )

    assert len(_result) == original_len
    assert _actions == []


# ---------------------------------------------------------------------------
# 3. overflow="hook" hook receives correct ContextWindowInfo
# ---------------------------------------------------------------------------


async def test_overflow_hook_receives_correct_info():
    captured: list[ContextWindowInfo] = []

    async def capture_hook(*, info: ContextWindowInfo, **_: Any):
        captured.append(info)

    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, capture_hook)

    msg_list = _build_msg_list(n_user=3, n_assistant=3, system=True)
    ctx = _make_context(overflow="hook")

    await _apply_context_windowing(
        msg_list,
        ctx,
        provider=None,
        hook_manager=hm,
        step=2,
        max_steps=10,
        agent_name="test-agent",
        model_name="openai:gpt-4o",
    )

    assert len(captured) == 1
    info = captured[0]
    assert info.step == 2
    assert info.max_steps == 10
    assert info.agent_name == "test-agent"
    assert info.overflow == "hook"
    assert info.force is False
    assert info.is_initial is False


# ---------------------------------------------------------------------------
# 4. overflow="summarize" with CONTEXT_WINDOW hook fires after built-in
# ---------------------------------------------------------------------------


async def test_summarize_hook_fires_after_builtin():
    """When overflow=summarize and a CONTEXT_WINDOW hook is registered,
    the hook fires AFTER the built-in windowing logic.
    The info.total_messages should reflect the post-windowing count.
    """
    captured: list[ContextWindowInfo] = []

    async def observe_hook(*, info: ContextWindowInfo, **_: Any):
        captured.append(info)

    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, observe_hook)

    # Build 25+ non-system messages that exceed history_rounds=20
    msg_list = _build_msg_list(n_user=15, n_assistant=15, system=True)
    ctx = _make_context(overflow="summarize", history_rounds=20)

    _result, _actions = await _apply_context_windowing(
        msg_list, ctx, provider=None, hook_manager=hm
    )

    assert len(captured) == 1
    info = captured[0]
    # After built-in windowing, total_messages should be <= 20 + system
    assert info.total_messages <= 21  # 20 non-system + 1 system


# ---------------------------------------------------------------------------
# 5. overflow="truncate" with CONTEXT_WINDOW hook fires augmentation after
#    built-in truncation
# ---------------------------------------------------------------------------


async def test_truncate_fires_augmentation_hook():
    captured: list[ContextWindowInfo] = []

    async def observe_hook(*, info: ContextWindowInfo, **_: Any):
        captured.append(info)

    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, observe_hook)

    # Build messages that exceed history_rounds=10
    msg_list = _build_msg_list(n_user=8, n_assistant=8, system=True)
    ctx = _make_context(overflow="truncate", history_rounds=10)

    result, actions = await _apply_context_windowing(msg_list, ctx, provider=None, hook_manager=hm)

    # Augmentation hook fires after truncation
    assert len(captured) == 1
    # After truncation: 10 non-system + 1 system = 11
    assert captured[0].total_messages == 11

    assert len(result) == 11

    # Verify there was a truncation action
    truncate_actions = [a for a in actions if a.action == "truncate"]
    assert len(truncate_actions) == 1


# ---------------------------------------------------------------------------
# 6. overflow="none" does NOT fire hook
# ---------------------------------------------------------------------------


async def test_overflow_none_does_not_fire_hook():
    called = []

    async def should_not_fire(**_: Any):
        called.append(True)

    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, should_not_fire)

    msg_list = _build_msg_list(n_user=5, n_assistant=5, system=True)
    ctx = _make_context(overflow="none")

    _result, _actions = await _apply_context_windowing(
        msg_list, ctx, provider=None, hook_manager=hm
    )

    assert called == []


# ---------------------------------------------------------------------------
# 7. hook can append to actions list
# ---------------------------------------------------------------------------


async def test_hook_appends_to_actions_list():
    async def add_action(*, actions: list, **_: Any):
        actions.append(_ContextAction("custom", 10, 5))

    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, add_action)

    msg_list = _build_msg_list(n_user=3, n_assistant=3, system=True)
    ctx = _make_context(overflow="hook")

    _, actions = await _apply_context_windowing(msg_list, ctx, provider=None, hook_manager=hm)

    custom_actions = [a for a in actions if a.action == "custom"]
    assert len(custom_actions) == 1
    assert custom_actions[0].before_count == 10
    assert custom_actions[0].after_count == 5


# ---------------------------------------------------------------------------
# 8. hook receives provider
# ---------------------------------------------------------------------------


async def test_hook_receives_provider():
    captured_provider: list = []

    async def capture_provider(*, provider: Any = None, **_: Any):
        captured_provider.append(provider)

    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, capture_provider)

    msg_list = _build_msg_list(n_user=3, n_assistant=3, system=True)
    ctx = _make_context(overflow="hook")

    sentinel_provider = object()
    await _apply_context_windowing(msg_list, ctx, provider=sentinel_provider, hook_manager=hm)

    assert len(captured_provider) == 1
    assert captured_provider[0] is sentinel_provider


# ---------------------------------------------------------------------------
# 9. ContextWindowHook ABC works as callable
# ---------------------------------------------------------------------------


async def test_context_window_hook_abc_callable():
    calls: list[ContextWindowInfo] = []

    class RecordingHook(ContextWindowHook):
        async def window(self, *, info: ContextWindowInfo, **extra: Any) -> None:
            calls.append(info)

    hook_instance = RecordingHook()
    hm = HookManager()
    hm.add(HookPoint.CONTEXT_WINDOW, hook_instance)

    msg_list = _build_msg_list(n_user=3, n_assistant=3, system=True)
    ctx = _make_context(overflow="hook")

    await _apply_context_windowing(msg_list, ctx, provider=None, hook_manager=hm, agent_name="abc")

    assert len(calls) == 1
    assert calls[0].agent_name == "abc"
