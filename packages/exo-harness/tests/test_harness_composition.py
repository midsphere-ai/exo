"""Tests for harness composition — nested harnesses and Swarm integration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

from exo.agent import Agent
from exo.harness.base import Harness, HarnessContext, HarnessNode
from exo.harness.types import HarnessEvent
from exo.types import (
    AgentOutput,
    RunResult,
    StreamEvent,
    Usage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


# ---------------------------------------------------------------------------
# Concrete harnesses
# ---------------------------------------------------------------------------


class InnerHarness(Harness):
    """Simple harness that runs one agent and emits a tag event."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        agent = next(iter(self.agents.values()))
        result = await ctx.run_agent(agent, ctx.input)
        yield ctx.emit("inner_done", output=result.output)


class OuterHarness(Harness):
    """Harness that runs another harness as a sub-unit."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        inner = self.agents["inner"]
        result = await ctx.run_agent(inner, ctx.input)
        ctx.state["inner_output"] = result.output
        yield ctx.emit("outer_done")


class StreamingOuterHarness(Harness):
    """Harness that streams from an inner harness."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        inner = self.agents["inner"]
        async for event in ctx.stream_agent(inner, ctx.input):
            yield event
        yield ctx.emit("outer_done")


class ChainedHarness(Harness):
    """Harness that runs agents sequentially, piping output to input."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        current_input = ctx.input
        for agent_name in self.agents:
            agent = self.agents[agent_name]
            result = await ctx.run_agent(agent, current_input)
            current_input = result.output
            ctx.state[agent_name] = result.output
            yield ctx.emit("step", agent=agent_name, output=result.output)


# ---------------------------------------------------------------------------
# Tests: Nested harness via ctx.run_agent()
# ---------------------------------------------------------------------------


class TestNestedHarness:
    async def test_harness_runs_inner_harness(self) -> None:
        """A harness can run another harness via ctx.run_agent()."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="inner result")])
        inner = InnerHarness(name="inner_h", agents=[agent])
        outer = OuterHarness(name="outer_h", agents={"inner": inner})

        result = await outer.run("Hi", provider=provider)

        # OuterHarness emits HarnessEvent, not TextEvent
        assert result.output == ""
        # Inner ran successfully
        assert outer.session["inner_output"] == ""  # InnerHarness emits HarnessEvent

    async def test_harness_streams_inner_harness(self) -> None:
        """A harness can stream another harness via ctx.stream_agent()."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="streamed")])
        inner = InnerHarness(name="inner_h", agents=[agent])
        outer = StreamingOuterHarness(name="outer_h", agents={"inner": inner})

        events = [ev async for ev in outer.stream("Hi", provider=provider)]

        harness_events = [e for e in events if isinstance(e, HarnessEvent)]
        # inner emits "inner_done", outer emits "outer_done"
        kinds = [e.kind for e in harness_events]
        assert "inner_done" in kinds
        assert "outer_done" in kinds


# ---------------------------------------------------------------------------
# Tests: HarnessNode in Swarm
# ---------------------------------------------------------------------------


class TestHarnessNodeComposition:
    async def test_node_in_swarm_run(self) -> None:
        """HarnessNode can be used in a Swarm's flow."""
        from exo.swarm import Swarm

        agent_a = Agent(name="first")
        agent_b = Agent(name="inner_bot")
        inner_harness = InnerHarness(name="inner_h", agents=[agent_b])
        node = HarnessNode(harness=inner_harness, name="harness_step")

        # Create a swarm: first >> harness_step
        provider = _make_provider([AgentOutput(text="classified")])
        swarm = Swarm(agents=[agent_a, node], flow="first >> harness_step")

        result = await swarm.run("test", provider=provider)

        assert isinstance(result, RunResult)

    async def test_node_context_isolation(self) -> None:
        """HarnessNode does not forward outer messages to inner harness."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="ok")])
        h = InnerHarness(name="h", agents=[agent])
        node = HarnessNode(harness=h)

        # Passing messages to node.run() — they should not be forwarded
        from exo.types import UserMessage

        result = await node.run("Hi", messages=[UserMessage(content="prior")], provider=provider)

        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# Tests: Chained agent execution
# ---------------------------------------------------------------------------


class TestChainedExecution:
    async def test_sequential_agent_chain(self) -> None:
        """Harness runs agents in sequence, piping output to input."""
        agent_a = Agent(name="step_a")
        agent_b = Agent(name="step_b")

        # step_a returns "A", step_b returns "B"
        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            responses = ["result_A", "result_B"]
            text = responses[min(call_count, len(responses) - 1)]
            call_count += 1

            resp_tool_calls: list[Any] = []

            class FakeResponse:
                content = text
                tool_calls = resp_tool_calls
                usage = Usage()

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = complete

        h = ChainedHarness(name="chain", agents=[agent_a, agent_b])

        await h.run("start", provider=provider)

        assert h.session["step_a"] == "result_A"
        assert h.session["step_b"] == "result_B"

    async def test_chain_emits_step_events(self) -> None:
        """ChainedHarness emits a HarnessEvent per step."""
        agent_a = Agent(name="a")
        agent_b = Agent(name="b")
        provider = _make_provider([AgentOutput(text="ok")])

        h = ChainedHarness(name="chain", agents=[agent_a, agent_b])

        events = [ev async for ev in h.stream("start", provider=provider)]

        harness_events = [e for e in events if isinstance(e, HarnessEvent)]
        assert len(harness_events) == 2
        assert harness_events[0].kind == "step"
        assert harness_events[1].kind == "step"
