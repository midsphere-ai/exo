"""Tests for Rail ABC, RailAction, RetryRequest, RailAbortError, and RailManager."""

from __future__ import annotations

import pytest

from exo.hooks import HookManager, HookPoint
from exo.rail import Rail, RailAbortError, RailAction, RailManager, RetryRequest
from exo.rail_types import InvokeInputs, ModelCallInputs, RailContext, ToolCallInputs
from exo.types import ExoError

# ---------------------------------------------------------------------------
# Concrete rail subclass for testing
# ---------------------------------------------------------------------------


class EchoRail(Rail):
    """Returns a pre-configured action for testing."""

    def __init__(
        self,
        name: str,
        action: RailAction | None = RailAction.CONTINUE,
        *,
        priority: int = 50,
    ) -> None:
        super().__init__(name, priority=priority)
        self.action = action
        self.last_ctx: RailContext | None = None

    async def handle(self, ctx: RailContext) -> RailAction | None:
        self.last_ctx = ctx
        return self.action


# ---------------------------------------------------------------------------
# RailAction
# ---------------------------------------------------------------------------


class TestRailAction:
    def test_members(self) -> None:
        assert set(RailAction) == {
            RailAction.CONTINUE,
            RailAction.SKIP,
            RailAction.RETRY,
            RailAction.ABORT,
        }

    def test_values(self) -> None:
        assert RailAction.CONTINUE == "continue"
        assert RailAction.SKIP == "skip"
        assert RailAction.RETRY == "retry"
        assert RailAction.ABORT == "abort"

    def test_is_str(self) -> None:
        for action in RailAction:
            assert isinstance(action, str)


# ---------------------------------------------------------------------------
# RetryRequest
# ---------------------------------------------------------------------------


class TestRetryRequest:
    def test_defaults(self) -> None:
        req = RetryRequest()
        assert req.delay == 0.0
        assert req.max_retries == 1
        assert req.reason == ""

    def test_custom_values(self) -> None:
        req = RetryRequest(delay=1.5, max_retries=3, reason="rate limited")
        assert req.delay == 1.5
        assert req.max_retries == 3
        assert req.reason == "rate limited"

    def test_frozen(self) -> None:
        req = RetryRequest()
        with pytest.raises(AttributeError):
            req.delay = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RailAbortError
# ---------------------------------------------------------------------------


class TestRailAbortError:
    def test_inherits_exo_error(self) -> None:
        assert issubclass(RailAbortError, ExoError)
        assert issubclass(RailAbortError, Exception)

    def test_message_without_reason(self) -> None:
        err = RailAbortError("safety_rail")
        assert str(err) == "Rail 'safety_rail' aborted"
        assert err.rail_name == "safety_rail"
        assert err.reason == ""

    def test_message_with_reason(self) -> None:
        err = RailAbortError("safety_rail", reason="injection detected")
        assert str(err) == "Rail 'safety_rail' aborted: injection detected"
        assert err.reason == "injection detected"

    def test_raise_and_catch(self) -> None:
        with pytest.raises(RailAbortError, match="blocked"):
            raise RailAbortError("blocker", reason="blocked")


# ---------------------------------------------------------------------------
# Rail ABC
# ---------------------------------------------------------------------------


class TestRailABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Rail("base")  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        rail = EchoRail("echo")
        assert rail.name == "echo"
        assert rail.priority == 50

    def test_custom_priority(self) -> None:
        rail = EchoRail("high", priority=10)
        assert rail.priority == 10

    def test_repr(self) -> None:
        rail = EchoRail("my_rail", priority=25)
        assert repr(rail) == "EchoRail(name='my_rail', priority=25)"

    @pytest.mark.asyncio
    async def test_handle_continue(self) -> None:
        rail = EchoRail("cont", action=RailAction.CONTINUE)
        ctx = RailContext(
            agent=None,
            event=HookPoint.START,
            inputs=InvokeInputs(input="hello"),
        )
        result = await rail.handle(ctx)
        assert result == RailAction.CONTINUE
        assert rail.last_ctx is ctx

    @pytest.mark.asyncio
    async def test_handle_skip(self) -> None:
        rail = EchoRail("skip", action=RailAction.SKIP)
        ctx = RailContext(
            agent=None,
            event=HookPoint.PRE_LLM_CALL,
            inputs=ModelCallInputs(messages=[{"role": "user", "content": "hi"}]),
        )
        result = await rail.handle(ctx)
        assert result == RailAction.SKIP

    @pytest.mark.asyncio
    async def test_handle_retry(self) -> None:
        rail = EchoRail("retry", action=RailAction.RETRY)
        ctx = RailContext(
            agent=None,
            event=HookPoint.POST_LLM_CALL,
            inputs=ModelCallInputs(messages=[]),
        )
        result = await rail.handle(ctx)
        assert result == RailAction.RETRY

    @pytest.mark.asyncio
    async def test_handle_abort(self) -> None:
        rail = EchoRail("abort", action=RailAction.ABORT)
        ctx = RailContext(
            agent=None,
            event=HookPoint.PRE_TOOL_CALL,
            inputs=ToolCallInputs(tool_name="dangerous"),
        )
        result = await rail.handle(ctx)
        assert result == RailAction.ABORT

    @pytest.mark.asyncio
    async def test_handle_none(self) -> None:
        rail = EchoRail("none", action=None)
        ctx = RailContext(
            agent=None,
            event=HookPoint.START,
            inputs=InvokeInputs(input="test"),
        )
        result = await rail.handle(ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Priority ordering (sort behavior)
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_sort_by_priority(self) -> None:
        rails = [
            EchoRail("low", priority=100),
            EchoRail("high", priority=10),
            EchoRail("mid", priority=50),
        ]
        sorted_rails = sorted(rails, key=lambda r: r.priority)
        assert [r.name for r in sorted_rails] == ["high", "mid", "low"]

    def test_equal_priority_preserves_order(self) -> None:
        rails = [
            EchoRail("first", priority=50),
            EchoRail("second", priority=50),
            EchoRail("third", priority=50),
        ]
        sorted_rails = sorted(rails, key=lambda r: r.priority)
        assert [r.name for r in sorted_rails] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# RailManager
# ---------------------------------------------------------------------------


class TestRailManager:
    def test_add_and_clear(self) -> None:
        mgr = RailManager()
        mgr.add(EchoRail("a"))
        mgr.add(EchoRail("b"))
        assert len(mgr._rails) == 2
        mgr.clear()
        assert len(mgr._rails) == 0

    def test_remove(self) -> None:
        mgr = RailManager()
        rail = EchoRail("a")
        mgr.add(rail)
        mgr.remove(rail)
        assert len(mgr._rails) == 0

    def test_remove_not_found(self) -> None:
        mgr = RailManager()
        with pytest.raises(ValueError):
            mgr.remove(EchoRail("missing"))

    @pytest.mark.asyncio
    async def test_empty_manager_returns_continue(self) -> None:
        mgr = RailManager()
        action = await mgr.run(HookPoint.START, agent=None, input="hello")
        assert action == RailAction.CONTINUE

    @pytest.mark.asyncio
    async def test_all_continue(self) -> None:
        mgr = RailManager()
        mgr.add(EchoRail("a", action=RailAction.CONTINUE))
        mgr.add(EchoRail("b", action=RailAction.CONTINUE))
        action = await mgr.run(HookPoint.START, agent=None, input="hello")
        assert action == RailAction.CONTINUE

    @pytest.mark.asyncio
    async def test_none_treated_as_continue(self) -> None:
        mgr = RailManager()
        mgr.add(EchoRail("a", action=None))
        action = await mgr.run(HookPoint.START, agent=None, input="hello")
        assert action == RailAction.CONTINUE

    @pytest.mark.asyncio
    async def test_first_non_continue_returned(self) -> None:
        mgr = RailManager()
        mgr.add(EchoRail("a", action=RailAction.CONTINUE, priority=10))
        mgr.add(EchoRail("b", action=RailAction.SKIP, priority=20))
        mgr.add(EchoRail("c", action=RailAction.ABORT, priority=30))
        action = await mgr.run(HookPoint.START, agent=None, input="hello")
        assert action == RailAction.SKIP

    # -- Priority ordering ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_priority_ordering(self) -> None:
        """Rails run in ascending priority order (lower number first)."""
        call_order: list[str] = []

        class TrackingRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                call_order.append(self.name)
                return RailAction.CONTINUE

        mgr = RailManager()
        mgr.add(TrackingRail("low", priority=100))
        mgr.add(TrackingRail("high", priority=10))
        mgr.add(TrackingRail("mid", priority=50))

        await mgr.run(HookPoint.START, agent=None, input="test")
        assert call_order == ["high", "mid", "low"]

    # -- Cross-rail extra dict -----------------------------------------------

    @pytest.mark.asyncio
    async def test_cross_rail_extra_dict(self) -> None:
        """Rails share the same extra dict within one invocation."""

        class WriterRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                ctx.extra["writer_was_here"] = True
                return RailAction.CONTINUE

        class ReaderRail(Rail):
            def __init__(self) -> None:
                super().__init__("reader", priority=20)
                self.saw_key = False

            async def handle(self, ctx: RailContext) -> RailAction | None:
                self.saw_key = ctx.extra.get("writer_was_here", False)
                return RailAction.CONTINUE

        reader = ReaderRail()
        mgr = RailManager()
        mgr.add(WriterRail("writer", priority=10))
        mgr.add(reader)

        await mgr.run(HookPoint.START, agent=None, input="test")
        assert reader.saw_key is True

    @pytest.mark.asyncio
    async def test_extra_dict_fresh_per_invocation(self) -> None:
        """Each run() call gets a fresh extra dict."""

        class AppendRail(Rail):
            def __init__(self) -> None:
                super().__init__("appender", priority=10)
                self.seen_counts: list[int] = []

            async def handle(self, ctx: RailContext) -> RailAction | None:
                count = ctx.extra.get("count", 0)
                self.seen_counts.append(count)
                ctx.extra["count"] = count + 1
                return RailAction.CONTINUE

        rail = AppendRail()
        mgr = RailManager()
        mgr.add(rail)

        await mgr.run(HookPoint.START, agent=None, input="a")
        await mgr.run(HookPoint.START, agent=None, input="b")
        # Each invocation starts with a fresh extra, so count is always 0
        assert rail.seen_counts == [0, 0]

    # -- Abort propagation ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_abort_stops_subsequent_rails(self) -> None:
        """When a rail returns ABORT, subsequent rails are NOT executed."""
        third = EchoRail("third", action=RailAction.CONTINUE, priority=30)
        mgr = RailManager()
        mgr.add(EchoRail("first", action=RailAction.CONTINUE, priority=10))
        mgr.add(EchoRail("second", action=RailAction.ABORT, priority=20))
        mgr.add(third)

        action = await mgr.run(HookPoint.START, agent=None, input="test")
        assert action == RailAction.ABORT
        # Third rail should not have been called
        assert third.last_ctx is None

    @pytest.mark.asyncio
    async def test_skip_stops_subsequent_rails(self) -> None:
        """SKIP also stops subsequent rails."""
        second = EchoRail("second", action=RailAction.CONTINUE, priority=20)
        mgr = RailManager()
        mgr.add(EchoRail("first", action=RailAction.SKIP, priority=10))
        mgr.add(second)

        action = await mgr.run(HookPoint.START, agent=None, input="test")
        assert action == RailAction.SKIP
        assert second.last_ctx is None

    # -- Input type mapping --------------------------------------------------

    @pytest.mark.asyncio
    async def test_pre_llm_call_builds_model_call_inputs(self) -> None:
        rail = EchoRail("check")
        mgr = RailManager()
        mgr.add(rail)
        msgs = [{"role": "user", "content": "hi"}]

        await mgr.run(HookPoint.PRE_LLM_CALL, agent=None, messages=msgs)
        assert rail.last_ctx is not None
        assert isinstance(rail.last_ctx.inputs, ModelCallInputs)
        assert rail.last_ctx.inputs.messages == msgs

    @pytest.mark.asyncio
    async def test_pre_tool_call_builds_tool_call_inputs(self) -> None:
        rail = EchoRail("check")
        mgr = RailManager()
        mgr.add(rail)

        await mgr.run(
            HookPoint.PRE_TOOL_CALL,
            agent=None,
            tool_name="my_tool",
            arguments={"x": 1},
        )
        assert rail.last_ctx is not None
        assert isinstance(rail.last_ctx.inputs, ToolCallInputs)
        assert rail.last_ctx.inputs.tool_name == "my_tool"
        assert rail.last_ctx.inputs.arguments == {"x": 1}

    @pytest.mark.asyncio
    async def test_start_builds_invoke_inputs(self) -> None:
        rail = EchoRail("check")
        mgr = RailManager()
        mgr.add(rail)

        await mgr.run(HookPoint.START, agent=None, input="hello")
        assert rail.last_ctx is not None
        assert isinstance(rail.last_ctx.inputs, InvokeInputs)
        assert rail.last_ctx.inputs.input == "hello"

    @pytest.mark.asyncio
    async def test_agent_passed_to_context(self) -> None:
        rail = EchoRail("check")
        mgr = RailManager()
        mgr.add(rail)
        sentinel = object()

        await mgr.run(HookPoint.START, agent=sentinel, input="test")
        assert rail.last_ctx is not None
        assert rail.last_ctx.agent is sentinel

    # -- HookManager compatibility -------------------------------------------

    @pytest.mark.asyncio
    async def test_hook_for_registers_on_hook_manager(self) -> None:
        """RailManager can be registered as a single hook on HookManager."""
        mgr = RailManager()
        mgr.add(EchoRail("pass", action=RailAction.CONTINUE))

        hook_mgr = HookManager()
        hook_mgr.add(HookPoint.PRE_LLM_CALL, mgr.hook_for(HookPoint.PRE_LLM_CALL))

        # Should not raise — all rails CONTINUE
        await hook_mgr.run(HookPoint.PRE_LLM_CALL, agent=None, messages=[])

    @pytest.mark.asyncio
    async def test_hook_for_abort_raises(self) -> None:
        """hook_for raises RailAbortError when a rail returns ABORT."""
        mgr = RailManager()
        mgr.add(EchoRail("blocker", action=RailAction.ABORT))

        hook_mgr = HookManager()
        hook_mgr.add(HookPoint.PRE_LLM_CALL, mgr.hook_for(HookPoint.PRE_LLM_CALL))

        with pytest.raises(RailAbortError, match="aborted"):
            await hook_mgr.run(HookPoint.PRE_LLM_CALL, agent=None, messages=[])

    @pytest.mark.asyncio
    async def test_existing_hooks_unaffected(self) -> None:
        """Existing hooks on HookManager still fire alongside rail hooks."""
        call_log: list[str] = []

        async def plain_hook(**_: object) -> None:
            call_log.append("plain")

        mgr = RailManager()

        class LogRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                call_log.append("rail")
                return RailAction.CONTINUE

        mgr.add(LogRail("logger", priority=50))

        hook_mgr = HookManager()
        hook_mgr.add(HookPoint.PRE_LLM_CALL, plain_hook)
        hook_mgr.add(HookPoint.PRE_LLM_CALL, mgr.hook_for(HookPoint.PRE_LLM_CALL))

        await hook_mgr.run(HookPoint.PRE_LLM_CALL, agent=None, messages=[])
        assert call_log == ["plain", "rail"]
