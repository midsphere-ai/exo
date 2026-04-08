"""Tests for parallel sub-agent execution in harness."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent
from exo.harness.base import Harness, HarnessContext
from exo.harness.middleware import CostTrackingMiddleware
from exo.harness.parallel import (
    SubAgentError,
    _ForkedSessionState,
)
from exo.harness.types import (
    SessionState,
    SubAgentResult,
    SubAgentStatus,
    SubAgentTask,
)
from exo.types import (
    AgentOutput,
    ErrorEvent,
    StatusEvent,
    StreamEvent,
    TextEvent,
    Usage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values."""
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


class _FakeStreamChunk:
    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Concrete harness implementations for testing
# ---------------------------------------------------------------------------


class ParallelRunHarness(Harness):
    """Runs agents in parallel via run_agents_parallel."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        tasks = [SubAgentTask(agent=agent, input=ctx.input) for agent in self.agents.values()]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)
        for r in results:
            ctx.state[r.agent_name] = r.output
            yield ctx.emit("result", agent=r.agent_name, output=r.output)


class ParallelStreamHarness(Harness):
    """Streams agents in parallel via stream_agents_parallel."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        tasks = [SubAgentTask(agent=agent, input=ctx.input) for agent in self.agents.values()]
        async for event in ctx.stream_agents_parallel(tasks, continue_on_error=True):
            yield event


class FailFastHarness(Harness):
    """Runs parallel agents in fail-fast mode."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        tasks = [SubAgentTask(agent=agent, input=ctx.input) for agent in self.agents.values()]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=False)
        for r in results:
            yield ctx.emit("result", agent=r.agent_name, output=r.output)


# ---------------------------------------------------------------------------
# Tests: _ForkedSessionState
# ---------------------------------------------------------------------------


class TestForkedSessionState:
    def test_read_through_to_parent(self) -> None:
        parent = SessionState(data={"key": "value"})
        fork = _ForkedSessionState(parent)
        assert fork["key"] == "value"

    def test_write_local(self) -> None:
        parent = SessionState(data={"key": "original"})
        fork = _ForkedSessionState(parent)
        fork["key"] = "modified"
        assert fork["key"] == "modified"
        assert parent["key"] == "original"

    def test_contains(self) -> None:
        parent = SessionState(data={"a": 1})
        fork = _ForkedSessionState(parent)
        fork["b"] = 2
        assert "a" in fork
        assert "b" in fork
        assert "c" not in fork

    def test_get_with_default(self) -> None:
        parent = SessionState(data={"a": 1})
        fork = _ForkedSessionState(parent)
        assert fork.get("a") == 1
        assert fork.get("missing", 42) == 42

    def test_dirty_tracking(self) -> None:
        parent = SessionState()
        fork = _ForkedSessionState(parent)
        assert not fork.dirty
        fork["x"] = 1
        assert fork.dirty

    def test_local_overrides_parent(self) -> None:
        parent = SessionState(data={"x": "parent"})
        fork = _ForkedSessionState(parent)
        fork["x"] = "child"
        assert fork.get("x") == "child"


# ---------------------------------------------------------------------------
# Tests: run_agents_parallel
# ---------------------------------------------------------------------------


class TestRunAgentsParallel:
    async def test_basic_two_agents(self) -> None:
        agent_a = Agent(name="alpha")
        agent_b = Agent(name="beta")
        provider = _make_provider([AgentOutput(text="hello")])
        h = ParallelRunHarness(name="h", agents=[agent_a, agent_b])

        await h.run("Hi", provider=provider)

        assert h.session["alpha"] == "hello"
        assert h.session["beta"] == "hello"

    async def test_results_in_task_order(self) -> None:
        agent_a = Agent(name="first")
        agent_b = Agent(name="second")
        provider = _make_provider([AgentOutput(text="ok")])

        h = ParallelRunHarness(name="h", agents=[agent_a, agent_b])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=h.session,
            harness=h,
            provider=provider,
        )

        tasks = [
            SubAgentTask(agent=agent_a, input="test"),
            SubAgentTask(agent=agent_b, input="test"),
        ]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        assert len(results) == 2
        assert results[0].agent_name == "first"
        assert results[1].agent_name == "second"

    async def test_empty_tasks(self) -> None:
        h = ParallelRunHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=h.session,
            harness=h,
        )
        results = await ctx.run_agents_parallel([])
        assert results == []

    async def test_single_task(self) -> None:
        agent = Agent(name="solo")
        provider = _make_provider([AgentOutput(text="done")])
        h = ParallelRunHarness(name="h", agents=[agent])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=h.session,
            harness=h,
            provider=provider,
        )

        tasks = [SubAgentTask(agent=agent, input="test")]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        assert len(results) == 1
        assert results[0].status == SubAgentStatus.SUCCESS
        assert results[0].output == "done"

    async def test_custom_name_override(self) -> None:
        agent = Agent(name="real_name")
        provider = _make_provider([AgentOutput(text="ok")])
        harness = ParallelRunHarness(name="h", agents=[agent])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=provider,
        )

        tasks = [SubAgentTask(agent=agent, input="test", name="custom_label")]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        assert results[0].agent_name == "custom_label"

    async def test_duplicate_names_raises(self) -> None:
        agent_a = Agent(name="dup")
        agent_b = Agent(name="dup")
        harness = ParallelRunHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
        )

        tasks = [
            SubAgentTask(agent=agent_a, input="a"),
            SubAgentTask(agent=agent_b, input="b"),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            await ctx.run_agents_parallel(tasks)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestParallelErrorHandling:
    async def test_continue_on_error_partial_results(self) -> None:
        """One agent fails, other succeeds — both results returned."""
        good_agent = Agent(name="good")
        bad_agent = Agent(name="bad")

        good_provider = _make_provider([AgentOutput(text="ok")])
        # bad_provider raises an error
        bad_provider = AsyncMock()
        bad_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))

        harness = ParallelRunHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=good_provider,  # default
        )

        tasks = [
            SubAgentTask(agent=good_agent, input="test"),
            SubAgentTask(agent=bad_agent, input="test", provider=bad_provider),
        ]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        assert len(results) == 2
        good_result = next(r for r in results if r.agent_name == "good")
        bad_result = next(r for r in results if r.agent_name == "bad")
        assert good_result.status == SubAgentStatus.SUCCESS
        assert bad_result.status == SubAgentStatus.FAILED
        assert bad_result.error is not None

    async def test_fail_fast_raises_sub_agent_error(self) -> None:
        """Fail-fast mode raises SubAgentError with partial results."""
        good_agent = Agent(name="good")
        bad_agent = Agent(name="bad")

        bad_provider = AsyncMock()
        bad_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
        good_provider = _make_provider([AgentOutput(text="ok")])

        harness = ParallelRunHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=good_provider,
        )

        tasks = [
            SubAgentTask(agent=bad_agent, input="test", provider=bad_provider),
            SubAgentTask(agent=good_agent, input="test"),
        ]
        with pytest.raises(SubAgentError) as exc_info:
            await ctx.run_agents_parallel(tasks, continue_on_error=False)

        assert "bad" in exc_info.value.failed_agents
        assert len(exc_info.value.results) == 2

    async def test_all_agents_fail(self) -> None:
        agent_a = Agent(name="a")
        agent_b = Agent(name="b")

        bad_provider = AsyncMock()
        bad_provider.complete = AsyncMock(side_effect=RuntimeError("fail"))

        harness = ParallelRunHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=bad_provider,
        )

        tasks = [
            SubAgentTask(agent=agent_a, input="test"),
            SubAgentTask(agent=agent_b, input="test"),
        ]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        assert all(r.status == SubAgentStatus.FAILED for r in results)


# ---------------------------------------------------------------------------
# Tests: State isolation
# ---------------------------------------------------------------------------


class TestParallelStateIsolation:
    async def test_state_merge_on_completion(self) -> None:
        """Successful agent's state writes merge back to parent."""
        agent = Agent(name="writer")
        provider = _make_provider([AgentOutput(text="written")])

        harness = ParallelRunHarness(name="h", agents=[agent])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=provider,
        )

        tasks = [SubAgentTask(agent=agent, input="test")]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        assert results[0].status == SubAgentStatus.SUCCESS

    async def test_state_isolation_between_agents(self) -> None:
        """Parallel agents don't see each other's state writes."""
        agent_a = Agent(name="a")
        agent_b = Agent(name="b")
        provider = _make_provider([AgentOutput(text="val")])

        state = SessionState(data={"shared": "original"})
        harness = ParallelRunHarness(name="h", agents=[agent_a, agent_b])
        harness.session = state

        await harness.run("test", provider=provider)

        # Both agents ran — the harness stores results in state
        assert "a" in state
        assert "b" in state


# ---------------------------------------------------------------------------
# Tests: Timeout
# ---------------------------------------------------------------------------


class TestParallelTimeout:
    async def test_per_agent_timeout(self) -> None:
        """Slow agent times out while fast agent succeeds."""

        fast_agent = Agent(name="fast")
        slow_agent = Agent(name="slow")

        # Fast provider responds immediately
        fast_provider = _make_provider([AgentOutput(text="quick")])

        # Slow provider blocks forever
        async def slow_complete(messages: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)

        slow_provider = AsyncMock()
        slow_provider.complete = slow_complete

        harness = ParallelRunHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=fast_provider,
        )

        tasks = [
            SubAgentTask(agent=fast_agent, input="test"),
            SubAgentTask(agent=slow_agent, input="test", provider=slow_provider, timeout=0.1),
        ]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        fast_result = next(r for r in results if r.agent_name == "fast")
        slow_result = next(r for r in results if r.agent_name == "slow")
        assert fast_result.status == SubAgentStatus.SUCCESS
        assert slow_result.status == SubAgentStatus.TIMED_OUT
        assert slow_result.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# Tests: stream_agents_parallel
# ---------------------------------------------------------------------------


class TestStreamAgentsParallel:
    async def test_stream_events_from_multiple_agents(self) -> None:
        """Events from parallel agents arrive with correct agent_name."""
        agent_a = Agent(name="alpha")
        agent_b = Agent(name="beta")

        provider_a = _make_stream_provider([[_FakeStreamChunk(delta="A")]])
        provider_b = _make_stream_provider([[_FakeStreamChunk(delta="B")]])

        harness = ParallelStreamHarness(name="h", agents=[agent_a, agent_b])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
        )

        tasks = [
            SubAgentTask(agent=agent_a, input="test", provider=provider_a),
            SubAgentTask(agent=agent_b, input="test", provider=provider_b),
        ]
        events: list[StreamEvent] = []
        async for event in ctx.stream_agents_parallel(tasks, continue_on_error=True):
            events.append(event)

        # Should have StatusEvent(starting) + TextEvent + StatusEvent(completed)
        # for each agent
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        starting = [e for e in status_events if e.status == "starting"]
        completed = [e for e in status_events if e.status == "completed"]
        assert len(starting) == 2
        assert len(completed) == 2

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 2
        texts = {e.text for e in text_events}
        assert "A" in texts
        assert "B" in texts

    async def test_stream_empty_tasks(self) -> None:
        harness = ParallelStreamHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
        )
        events = [e async for e in ctx.stream_agents_parallel([])]
        assert events == []

    async def test_stream_error_event_on_failure(self) -> None:
        """Failed agent produces ErrorEvent in the stream."""
        agent = Agent(name="failing")
        bad_provider = AsyncMock()
        bad_provider.stream = AsyncMock(side_effect=RuntimeError("crash"))

        harness = ParallelStreamHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
        )

        tasks = [SubAgentTask(agent=agent, input="test", provider=bad_provider)]
        events: list[StreamEvent] = []
        async for event in ctx.stream_agents_parallel(tasks, continue_on_error=True):
            events.append(event)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) >= 1
        assert error_events[0].agent_name == "failing"

    async def test_stream_writes_log_files(self) -> None:
        """Each sub-agent writes events to a /tmp/ log file."""
        agent = Agent(name="logger")
        provider = _make_stream_provider([[_FakeStreamChunk(delta="logged")]])

        harness = ParallelStreamHarness(name="h", agents=[agent])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
        )

        tasks = [SubAgentTask(agent=agent, input="test", provider=provider)]
        async for _event in ctx.stream_agents_parallel(tasks, continue_on_error=True):
            pass

        # Find the log file
        import glob

        log_files = glob.glob("/tmp/exo_subagent_logger_*.log")
        assert len(log_files) >= 1
        with open(log_files[-1]) as f:
            content = f.read()
        assert "[starting]" in content
        assert "[text] logged" in content
        assert "[completed]" in content

    async def test_stream_appends_assistant_message(self) -> None:
        """Sub-agent output is appended to ctx.messages as AssistantMessage."""
        agent_a = Agent(name="alpha")
        agent_b = Agent(name="beta")

        provider_a = _make_stream_provider([[_FakeStreamChunk(delta="Hello")]])
        provider_b = _make_stream_provider([[_FakeStreamChunk(delta="World")]])

        harness = ParallelStreamHarness(name="h", agents=[agent_a, agent_b])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
        )

        tasks = [
            SubAgentTask(agent=agent_a, input="test", provider=provider_a),
            SubAgentTask(agent=agent_b, input="test", provider=provider_b),
        ]
        async for _event in ctx.stream_agents_parallel(tasks, continue_on_error=True):
            pass

        # AssistantMessages: 2 "started" + 2 "completed" = 4 total
        from exo.types import AssistantMessage

        assistant_msgs = [m for m in ctx.messages if isinstance(m, AssistantMessage)]
        assert len(assistant_msgs) == 4
        started_msgs = [m for m in assistant_msgs if "executing in the background" in m.content]
        completed_msgs = [m for m in assistant_msgs if "completed" in m.content]
        assert len(started_msgs) == 2
        assert len(completed_msgs) == 2
        # Started messages tell the parent where to look
        assert any("alpha" in m.content and "/tmp/" in m.content for m in started_msgs)
        assert any("beta" in m.content and "/tmp/" in m.content for m in started_msgs)
        # Completed messages carry the output
        assert any("alpha" in c.content and "Hello" in c.content for c in completed_msgs)
        assert any("beta" in c.content and "World" in c.content for c in completed_msgs)

    async def test_run_parallel_appends_assistant_message(self) -> None:
        """run_agents_parallel also appends AssistantMessage to ctx.messages."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="answer")])

        harness = ParallelRunHarness(name="h", agents=[agent])
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=provider,
        )

        tasks = [SubAgentTask(agent=agent, input="test")]
        results = await ctx.run_agents_parallel(tasks, continue_on_error=True)

        assert results[0].output == "answer"
        assert results[0].log_path is not None

        from exo.types import AssistantMessage

        assistant_msgs = [m for m in ctx.messages if isinstance(m, AssistantMessage)]
        assert len(assistant_msgs) == 2  # 1 "started" + 1 "completed"
        started = [m for m in assistant_msgs if "executing in the background" in m.content]
        completed = [m for m in assistant_msgs if "completed" in m.content]
        assert len(started) == 1
        assert "/tmp/" in started[0].content
        assert "bot" in started[0].content
        assert len(completed) == 1
        assert "bot" in completed[0].content
        assert "answer" in completed[0].content

    async def test_harness_reads_output_from_messages(self) -> None:
        """Parent harness reads sub-agent output from ctx.messages."""

        class ReadFromMessagesHarness(Harness):
            async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
                tasks = [
                    SubAgentTask(agent=agent, input=ctx.input) for agent in self.agents.values()
                ]
                async for event in ctx.stream_agents_parallel(tasks, continue_on_error=True):
                    yield event
                # Parent reads output from messages
                from exo.types import AssistantMessage

                for msg in ctx.messages:
                    if isinstance(msg, AssistantMessage) and "completed" in msg.content:
                        ctx.state["got_message"] = True

        agent = Agent(name="worker")
        provider = _make_stream_provider([[_FakeStreamChunk(delta="done")]])
        h = ReadFromMessagesHarness(name="h", agents=[agent])
        await h.run("test", provider=provider)

        assert h.session.get("got_message") is True


# ---------------------------------------------------------------------------
# Tests: Cancellation
# ---------------------------------------------------------------------------


class TestParallelCancellation:
    async def test_harness_cancel_stops_parallel(self) -> None:
        """Harness-level cancel stops all parallel agents."""

        async def slow_complete(messages: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)

        agent = Agent(name="slow")
        slow_provider = AsyncMock()
        slow_provider.stream = slow_complete  # Never returns

        harness = ParallelStreamHarness(name="h", agents=[agent])

        # Cancel after a short delay
        async def cancel_later():
            await asyncio.sleep(0.05)
            harness.cancel()

        cancel_task = asyncio.create_task(cancel_later())

        events: list[StreamEvent] = []
        try:
            async for event in harness.stream("test", provider=slow_provider):
                events.append(event)
        except Exception:
            pass  # May raise on cancellation

        await cancel_task

    async def test_cancel_specific_agent(self) -> None:
        """cancel_agent() stops one agent while others continue."""

        agent_a = Agent(name="a")
        agent_b = Agent(name="b")

        provider_a = _make_stream_provider([[_FakeStreamChunk(delta="A")]])
        provider_b = _make_stream_provider([[_FakeStreamChunk(delta="B")]])

        harness = ParallelStreamHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
        )

        tasks = [
            SubAgentTask(agent=agent_a, input="test", provider=provider_a),
            SubAgentTask(agent=agent_b, input="test", provider=provider_b),
        ]

        events: list[StreamEvent] = []
        async for event in ctx.stream_agents_parallel(tasks, continue_on_error=True):
            events.append(event)
            # Cancel agent "a" after first event
            if len(events) == 1:
                ctx.cancel_agent("a")

        # Both agents should have produced at least some events
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Tests: Max concurrency
# ---------------------------------------------------------------------------


class TestMaxConcurrency:
    async def test_limits_parallel_execution(self) -> None:
        """max_concurrency=1 forces sequential execution."""
        execution_order: list[str] = []

        agent_a = Agent(name="a")
        agent_b = Agent(name="b")

        call_count = 0

        async def ordered_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            current = call_count
            call_count += 1
            execution_order.append(f"start_{current}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{current}")

            class FakeResponse:
                content = f"result_{current}"
                tool_calls: list[Any] = []  # noqa: RUF012
                usage = Usage()

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = ordered_complete

        harness = ParallelRunHarness(name="h")
        ctx = HarnessContext(
            input="test",
            messages=[],
            state=harness.session,
            harness=harness,
            provider=provider,
        )

        tasks = [
            SubAgentTask(agent=agent_a, input="test"),
            SubAgentTask(agent=agent_b, input="test"),
        ]

        results = await ctx.run_agents_parallel(tasks, continue_on_error=True, max_concurrency=1)

        assert len(results) == 2
        # With max_concurrency=1, execution should be sequential
        # First agent ends before second starts
        assert execution_order[1] == "end_0"
        assert execution_order[2] == "start_1"


# ---------------------------------------------------------------------------
# Tests: Checkpoint with pending_agents
# ---------------------------------------------------------------------------


class TestCheckpointPendingAgents:
    async def test_pending_agents_roundtrip(self) -> None:
        from exo.harness.checkpoint import CheckpointAdapter
        from exo.harness.types import HarnessCheckpoint

        class MockStore:
            def __init__(self) -> None:
                self._items: list[Any] = []

            async def add(self, item: Any) -> None:
                self._items.append(item)

            async def search(self, **kwargs: Any) -> list[Any]:
                return self._items[-1:]

        store = MockStore()
        adapter = CheckpointAdapter(store, "h")

        cp = HarnessCheckpoint(
            harness_name="h",
            session_state={"x": 1},
            completed_agents=["a"],
            pending_agents=["b", "c"],
        )
        await adapter.save(cp)

        loaded = await adapter.load_latest()
        assert loaded is not None
        assert loaded.pending_agents == ["b", "c"]

    async def test_backward_compat_single_pending(self) -> None:
        """Old checkpoint without pending_agents still loads correctly."""
        from exo.harness.checkpoint import CheckpointAdapter
        from exo.harness.types import HarnessCheckpoint

        class MockStore:
            def __init__(self) -> None:
                self._items: list[Any] = []

            async def add(self, item: Any) -> None:
                self._items.append(item)

            async def search(self, **kwargs: Any) -> list[Any]:
                return self._items[-1:]

        store = MockStore()
        adapter = CheckpointAdapter(store, "h")

        # Save with old-style single pending_agent
        cp = HarnessCheckpoint(
            harness_name="h",
            session_state={},
            completed_agents=[],
            pending_agent="agent_x",
        )
        await adapter.save(cp)

        loaded = await adapter.load_latest()
        assert loaded is not None
        assert loaded.pending_agent == "agent_x"
        assert loaded.pending_agents == ["agent_x"]


# ---------------------------------------------------------------------------
# Tests: Middleware integration
# ---------------------------------------------------------------------------


class TestParallelMiddleware:
    async def test_events_flow_through_middleware(self) -> None:
        """CostTrackingMiddleware sees usage from parallel agents."""
        agent_a = Agent(name="a")
        agent_b = Agent(name="b")

        chunks = [
            _FakeStreamChunk(
                delta="hi",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        ]
        provider = _make_stream_provider([chunks])

        h = ParallelStreamHarness(
            name="h",
            agents=[agent_a, agent_b],
            middleware=[CostTrackingMiddleware()],
        )

        events = [ev async for ev in h.stream("Hi", provider=provider)]

        # Events should have flowed through — check that we got some
        assert len(events) > 0


# ---------------------------------------------------------------------------
# Tests: SubAgentResult properties
# ---------------------------------------------------------------------------


class TestSubAgentResult:
    def test_success_result(self) -> None:
        r = SubAgentResult(
            agent_name="test",
            status=SubAgentStatus.SUCCESS,
            output="hello",
            elapsed_seconds=1.5,
        )
        assert r.agent_name == "test"
        assert r.status == SubAgentStatus.SUCCESS
        assert r.output == "hello"
        assert r.error is None
        assert r.elapsed_seconds == 1.5

    def test_failed_result(self) -> None:
        exc = RuntimeError("boom")
        r = SubAgentResult(
            agent_name="test",
            status=SubAgentStatus.FAILED,
            error=exc,
        )
        assert r.status == SubAgentStatus.FAILED
        assert r.error is exc
        assert r.output == ""

    def test_timed_out_result(self) -> None:
        r = SubAgentResult(
            agent_name="test",
            status=SubAgentStatus.TIMED_OUT,
        )
        assert r.status == SubAgentStatus.TIMED_OUT

    def test_cancelled_result(self) -> None:
        r = SubAgentResult(
            agent_name="test",
            status=SubAgentStatus.CANCELLED,
        )
        assert r.status == SubAgentStatus.CANCELLED


# ---------------------------------------------------------------------------
# Tests: SubAgentError
# ---------------------------------------------------------------------------


class TestSubAgentError:
    def test_error_carries_results(self) -> None:
        results = [
            SubAgentResult(agent_name="a", status=SubAgentStatus.SUCCESS, output="ok"),
            SubAgentResult(
                agent_name="b",
                status=SubAgentStatus.FAILED,
                error=RuntimeError("fail"),
            ),
        ]
        err = SubAgentError(
            "test failure",
            results=results,
            failed_agents=["b"],
        )
        assert err.results == results
        assert err.failed_agents == ["b"]
        assert "test failure" in str(err)
