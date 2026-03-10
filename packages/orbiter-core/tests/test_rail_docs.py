"""Tests verifying that code examples in docs/porting-guide/rails.md typecheck and run."""

from __future__ import annotations

from orbiter.hooks import HookPoint
from orbiter.rail import Rail, RailAction, RailManager
from orbiter.rail_types import RailContext, ToolCallInputs


# --- Example 1: Block a dangerous tool ---


class BlockDangerousTool(Rail):
    """Block execution of a specific tool by name."""

    def __init__(self, blocked_tool: str) -> None:
        super().__init__("block_tool", priority=10)
        self.blocked_tool = blocked_tool

    async def handle(self, ctx: RailContext) -> RailAction | None:
        if (
            ctx.event == HookPoint.PRE_TOOL_CALL
            and isinstance(ctx.inputs, ToolCallInputs)
            and ctx.inputs.tool_name == self.blocked_tool
        ):
            return RailAction.ABORT
        return RailAction.CONTINUE


# --- Example 2: Cross-rail state sharing ---


class RateCounterRail(Rail):
    """Write rate-limit state for downstream rails."""

    def __init__(self) -> None:
        super().__init__("rate_counter", priority=10)

    async def handle(self, ctx: RailContext) -> RailAction | None:
        ctx.extra["calls_remaining"] = 5
        return RailAction.CONTINUE


class RateLoggerRail(Rail):
    """Read rate-limit state from upstream rails."""

    def __init__(self) -> None:
        super().__init__("rate_logger", priority=80)

    async def handle(self, ctx: RailContext) -> RailAction | None:
        remaining = ctx.extra.get("calls_remaining", "unknown")
        print(f"Rate limit remaining: {remaining}")
        return RailAction.CONTINUE


# --- Tests ---

import pytest


@pytest.mark.asyncio
async def test_block_dangerous_tool_aborts() -> None:
    rail = BlockDangerousTool("rm_rf")
    ctx = RailContext(
        agent=None,
        event=HookPoint.PRE_TOOL_CALL,
        inputs=ToolCallInputs(tool_name="rm_rf", arguments={}),
    )
    action = await rail.handle(ctx)
    assert action == RailAction.ABORT


@pytest.mark.asyncio
async def test_block_dangerous_tool_continues_for_safe_tool() -> None:
    rail = BlockDangerousTool("rm_rf")
    ctx = RailContext(
        agent=None,
        event=HookPoint.PRE_TOOL_CALL,
        inputs=ToolCallInputs(tool_name="read_file", arguments={}),
    )
    action = await rail.handle(ctx)
    assert action == RailAction.CONTINUE


@pytest.mark.asyncio
async def test_cross_rail_state_sharing() -> None:
    manager = RailManager()
    manager.add(RateCounterRail())
    manager.add(RateLoggerRail())

    action = await manager.run(
        HookPoint.PRE_LLM_CALL, agent=None, messages=[]
    )
    assert action == RailAction.CONTINUE


@pytest.mark.asyncio
async def test_rail_manager_priority_ordering() -> None:
    """Verify rails run in priority order."""
    call_order: list[str] = []

    class TrackingRail(Rail):
        async def handle(self, ctx: RailContext) -> RailAction | None:
            call_order.append(self.name)
            return RailAction.CONTINUE

    manager = RailManager()
    manager.add(TrackingRail("third", priority=90))
    manager.add(TrackingRail("first", priority=10))
    manager.add(TrackingRail("second", priority=50))

    await manager.run(HookPoint.START, agent=None, input="hello")
    assert call_order == ["first", "second", "third"]
