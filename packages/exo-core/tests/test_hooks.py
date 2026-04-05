"""Tests for exo.hooks — async hook system."""

import pytest

from exo.hooks import HookManager, HookPoint, run_hooks

# --- HookPoint enum ---


class TestHookPoint:
    def test_member_count(self) -> None:
        assert len(HookPoint) == 8

    def test_all_values_distinct(self) -> None:
        values = [p.value for p in HookPoint]
        assert len(values) == len(set(values))

    def test_expected_names(self) -> None:
        names = {p.name for p in HookPoint}
        assert names == {
            "START",
            "FINISHED",
            "ERROR",
            "PRE_LLM_CALL",
            "POST_LLM_CALL",
            "PRE_TOOL_CALL",
            "POST_TOOL_CALL",
            "CONTEXT_WINDOW",
        }


# --- Run hooks ---


class TestRun:
    async def test_hook_called_with_kwargs(self) -> None:
        mgr = HookManager()
        calls: list[dict] = []

        async def hook(**data: object) -> None:
            calls.append(dict(data))

        mgr.add(HookPoint.START, hook)
        await mgr.run(HookPoint.START, agent="bot", step=1)
        assert calls == [{"agent": "bot", "step": 1}]

    async def test_multiple_hooks_in_order(self) -> None:
        mgr = HookManager()
        order: list[str] = []

        async def first(**data: object) -> None:
            order.append("first")

        async def second(**data: object) -> None:
            order.append("second")

        mgr.add(HookPoint.START, first)
        mgr.add(HookPoint.START, second)
        await mgr.run(HookPoint.START)
        assert order == ["first", "second"]

    async def test_empty_point_is_noop(self) -> None:
        mgr = HookManager()
        # Should not raise
        await mgr.run(HookPoint.FINISHED, value=42)

    async def test_exception_propagates(self) -> None:
        mgr = HookManager()

        async def bad_hook(**data: object) -> None:
            raise RuntimeError("hook failed")

        mgr.add(HookPoint.ERROR, bad_hook)
        with pytest.raises(RuntimeError, match="hook failed"):
            await mgr.run(HookPoint.ERROR)


# --- Point isolation ---


class TestPointIsolation:
    async def test_different_points_are_isolated(self) -> None:
        mgr = HookManager()
        start_calls: list[str] = []
        finish_calls: list[str] = []

        async def on_start(**data: object) -> None:
            start_calls.append("start")

        async def on_finish(**data: object) -> None:
            finish_calls.append("finish")

        mgr.add(HookPoint.START, on_start)
        mgr.add(HookPoint.FINISHED, on_finish)

        await mgr.run(HookPoint.START)
        assert start_calls == ["start"]
        assert finish_calls == []

        await mgr.run(HookPoint.FINISHED)
        assert finish_calls == ["finish"]
        assert start_calls == ["start"]  # still just one call


# --- Remove ---


class TestRemove:
    async def test_remove_works(self) -> None:
        mgr = HookManager()
        calls: list[int] = []

        async def hook(**data: object) -> None:
            calls.append(1)

        mgr.add(HookPoint.START, hook)
        await mgr.run(HookPoint.START)
        assert len(calls) == 1

        mgr.remove(HookPoint.START, hook)
        await mgr.run(HookPoint.START)
        assert len(calls) == 1  # not called again

    async def test_remove_idempotent(self) -> None:
        mgr = HookManager()

        async def hook(**data: object) -> None:
            pass

        # Should not raise even if hook was never registered
        mgr.remove(HookPoint.START, hook)

    async def test_remove_first_occurrence_only(self) -> None:
        mgr = HookManager()
        calls: list[int] = []

        async def hook(**data: object) -> None:
            calls.append(1)

        mgr.add(HookPoint.START, hook)
        mgr.add(HookPoint.START, hook)  # registered twice
        mgr.remove(HookPoint.START, hook)  # remove first
        await mgr.run(HookPoint.START)
        assert len(calls) == 1  # second registration still fires


# --- has_hooks ---


class TestHasHooks:
    def test_false_when_empty(self) -> None:
        mgr = HookManager()
        assert mgr.has_hooks(HookPoint.START) is False

    def test_true_after_add(self) -> None:
        mgr = HookManager()

        async def hook(**data: object) -> None:
            pass

        mgr.add(HookPoint.START, hook)
        assert mgr.has_hooks(HookPoint.START) is True

    def test_false_after_remove(self) -> None:
        mgr = HookManager()

        async def hook(**data: object) -> None:
            pass

        mgr.add(HookPoint.START, hook)
        mgr.remove(HookPoint.START, hook)
        assert mgr.has_hooks(HookPoint.START) is False


# --- Clear ---


class TestClear:
    async def test_clear_removes_all(self) -> None:
        mgr = HookManager()
        calls: list[str] = []

        async def hook_a(**data: object) -> None:
            calls.append("a")

        async def hook_b(**data: object) -> None:
            calls.append("b")

        mgr.add(HookPoint.START, hook_a)
        mgr.add(HookPoint.FINISHED, hook_b)
        mgr.clear()

        await mgr.run(HookPoint.START)
        await mgr.run(HookPoint.FINISHED)
        assert calls == []
        assert mgr.has_hooks(HookPoint.START) is False
        assert mgr.has_hooks(HookPoint.FINISHED) is False


# --- run_hooks convenience ---


class TestRunHooks:
    async def test_delegates_to_manager(self) -> None:
        mgr = HookManager()
        calls: list[dict] = []

        async def hook(**data: object) -> None:
            calls.append(dict(data))

        mgr.add(HookPoint.PRE_LLM_CALL, hook)
        await run_hooks(mgr, HookPoint.PRE_LLM_CALL, model="gpt-4o")
        assert calls == [{"model": "gpt-4o"}]
