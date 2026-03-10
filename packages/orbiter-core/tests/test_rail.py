"""Tests for Rail ABC, RailAction, RetryRequest, and RailAbortError."""

from __future__ import annotations

import pytest

from orbiter.hooks import HookPoint
from orbiter.rail import Rail, RailAbortError, RailAction, RetryRequest
from orbiter.rail_types import InvokeInputs, ModelCallInputs, RailContext, ToolCallInputs
from orbiter.types import OrbiterError


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
    def test_inherits_orbiter_error(self) -> None:
        assert issubclass(RailAbortError, OrbiterError)
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
