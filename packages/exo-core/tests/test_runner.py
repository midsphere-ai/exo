"""Tests for exo.runner — public run() / run.sync() / run.stream() entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent
from exo.config import ModelConfig
from exo.runner import run
from exo.tool import tool
from exo.types import (
    AgentOutput,
    ErrorEvent,
    RunResult,
    StatusEvent,
    StepEvent,
    StreamEvent,
    TextEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    UsageEvent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Fixtures: mock provider
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


class _RecordingProvider:
    """Provider stub that records completion inputs for assertions."""

    def __init__(self, responses: list[AgentOutput]) -> None:
        self._responses = responses
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append({"messages": list(messages), "kwargs": kwargs})
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage

        return FakeResponse()


# ---------------------------------------------------------------------------
# run() async
# ---------------------------------------------------------------------------


class TestRunAsync:
    async def test_basic_run(self) -> None:
        """run() returns RunResult with agent output."""
        agent = Agent(name="bot", instructions="Be helpful.")
        provider = _make_provider([AgentOutput(text="Hello!")])

        result = await run(agent, "Hi", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "Hello!"
        assert result.steps >= 1

    async def test_run_with_usage(self) -> None:
        """run() propagates token usage to RunResult."""
        agent = Agent(name="bot")
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        provider = _make_provider([AgentOutput(text="ok", usage=usage)])

        result = await run(agent, "test", provider=provider)

        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    async def test_run_includes_messages(self) -> None:
        """run() result includes message history."""
        agent = Agent(name="bot", instructions="You are nice.")
        provider = _make_provider([AgentOutput(text="Sure!")])

        result = await run(agent, "hello", provider=provider)

        assert len(result.messages) > 0

    async def test_run_with_tool(self) -> None:
        """run() handles agents with tools (tool call → text response)."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="greeter", tools=[greet])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="greet", arguments='{"name":"Alice"}')],
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
            AgentOutput(
                text="I greeted Alice for you!",
                usage=Usage(input_tokens=30, output_tokens=15, total_tokens=45),
            ),
        ]
        provider = _make_provider(responses)

        result = await run(agent, "Greet Alice", provider=provider)

        assert result.output == "I greeted Alice for you!"

    async def test_planning_enabled_runs_planner_before_executor(self) -> None:
        """Planner-enabled agents inject a plan before executor tool use."""

        @tool
        def lookup(topic: str) -> str:
            """Look up a topic."""
            return f"Facts about {topic}"

        agent = Agent(
            name="researcher",
            instructions="Execute the task.",
            tools=[lookup],
            planning_enabled=True,
            planning_instructions="Return a short numbered plan.",
        )
        provider = _RecordingProvider(
            [
                AgentOutput(text="1. Look up the facts.\n2. Summarize them."),
                AgentOutput(
                    text="",
                    tool_calls=[ToolCall(id="tc1", name="lookup", arguments='{"topic":"exo"}')],
                ),
                AgentOutput(text="Summary ready."),
            ]
        )

        result = await run(agent, "Research exo", provider=provider)

        assert result.output == "Summary ready."
        assert len(provider.calls) == 3

        planner_call = provider.calls[0]
        executor_call = provider.calls[1]
        planner_tools = {schema["function"]["name"] for schema in planner_call["kwargs"]["tools"]}
        executor_tools = {schema["function"]["name"] for schema in executor_call["kwargs"]["tools"]}

        assert planner_call["messages"][0].content == "Return a short numbered plan."
        assert planner_tools == executor_tools
        assert executor_call["messages"][-1].role == "user"
        assert "Original task:\nResearch exo" in executor_call["messages"][-1].content
        assert "Planner output:\n1. Look up the facts.\n2. Summarize them." in (
            executor_call["messages"][-1].content
        )

    async def test_planner_defaults_to_executor_instructions(self) -> None:
        """Empty planner instructions fall back to the executor instructions."""
        agent = Agent(
            name="planner-bot",
            instructions="Break the work down before acting.",
            planning_enabled=True,
        )
        provider = _RecordingProvider(
            [
                AgentOutput(text="1. Break the task down."),
                AgentOutput(text="Done."),
            ]
        )

        result = await run(agent, "Handle the task", provider=provider)

        assert result.output == "Done."
        assert provider.calls[0]["messages"][0].content == "Break the work down before acting."

    async def test_planner_uses_planning_model_override(self) -> None:
        """Planner model overrides clone the executor provider when possible."""
        call_log: list[str] = []
        responses_by_model = {
            "gpt-4o-mini": [AgentOutput(text="1. Prepare the work.")],
            "gpt-4o": [AgentOutput(text="Finished.")],
        }

        class CloneableProvider:
            def __init__(self, config: ModelConfig) -> None:
                self.config = config

            async def complete(self, messages: Any, **kwargs: Any) -> Any:
                call_log.append(self.config.model_name)
                resp = responses_by_model[self.config.model_name].pop(0)

                class FakeResponse:
                    content = resp.text
                    tool_calls = resp.tool_calls
                    usage = resp.usage

                return FakeResponse()

        provider = CloneableProvider(ModelConfig(provider="openai", model_name="gpt-4o"))
        agent = Agent(
            name="planner-bot",
            model="openai:gpt-4o",
            instructions="Do the task.",
            planning_enabled=True,
            planning_model="openai:gpt-4o-mini",
        )

        result = await run(agent, "Handle the task", provider=provider)

        assert result.output == "Finished."
        assert call_log == ["gpt-4o-mini", "gpt-4o"]


# ---------------------------------------------------------------------------
# run.sync()
# ---------------------------------------------------------------------------


class TestRunSync:
    def test_sync_basic(self) -> None:
        """run.sync() returns RunResult synchronously."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="sync ok")])

        result = run.sync(agent, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "sync ok"

    def test_sync_with_usage(self) -> None:
        """run.sync() propagates usage stats."""
        agent = Agent(name="bot")
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        provider = _make_provider([AgentOutput(text="ok", usage=usage)])

        result = run.sync(agent, "test", provider=provider)

        assert result.usage.input_tokens == 10

    def test_sync_with_tool(self) -> None:
        """run.sync() handles tool-calling agents."""

        @tool
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        agent = Agent(name="calc", tools=[add])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="add", arguments='{"a":2,"b":3}')],
            ),
            AgentOutput(text="The answer is 5."),
        ]
        provider = _make_provider(responses)

        result = run.sync(agent, "What is 2+3?", provider=provider)

        assert result.output == "The answer is 5."


# ---------------------------------------------------------------------------
# Multi-turn via messages param
# ---------------------------------------------------------------------------


class TestRunMultiTurn:
    async def test_prior_messages(self) -> None:
        """run() forwards prior messages for multi-turn conversations."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="Continued!")])

        prior = [UserMessage(content="Earlier message")]
        result = await run(agent, "Continue", messages=prior, provider=provider)

        assert result.output == "Continued!"

    async def test_multi_turn_accumulates(self) -> None:
        """Multiple run() calls with messages param create ongoing conversation."""
        agent = Agent(name="bot", instructions="You are a counter.")
        provider1 = _make_provider([AgentOutput(text="Count: 1")])

        r1 = await run(agent, "Start counting", provider=provider1)
        assert r1.output == "Count: 1"

        # Second turn passes first result's messages
        provider2 = _make_provider([AgentOutput(text="Count: 2")])
        r2 = await run(agent, "Next", messages=r1.messages, provider=provider2)
        assert r2.output == "Count: 2"

    async def test_planning_disabled_bypasses_planner(self) -> None:
        """Agents with planning disabled keep the existing execution path."""
        agent = Agent(name="bot", instructions="Just answer directly.")
        provider = _RecordingProvider([AgentOutput(text="Done.")])

        result = await run(agent, "Handle the task", provider=provider)

        assert result.output == "Done."
        assert len(provider.calls) == 1
        assert provider.calls[0]["messages"][-1].content == "Handle the task"
        assert "Planner output" not in provider.calls[0]["messages"][-1].content


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestRunErrors:
    async def test_no_provider_raises(self) -> None:
        """run() raises when no provider given and auto-resolve fails."""
        agent = Agent(name="bot")

        # Agent.run() raises AgentError → call_runner wraps in CallRunnerError
        from exo._internal.call_runner import CallRunnerError

        with pytest.raises(CallRunnerError):
            await run(agent, "test")

    async def test_provider_error_propagates(self) -> None:
        """Provider errors bubble up through run()."""
        agent = Agent(name="bot")
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("Service down"))

        from exo._internal.call_runner import CallRunnerError

        with pytest.raises(CallRunnerError, match="Call runner failed"):
            await run(agent, "test", provider=provider)

    def test_sync_error_propagates(self) -> None:
        """run.sync() propagates errors from the async path."""
        agent = Agent(name="bot")

        from exo._internal.call_runner import CallRunnerError

        with pytest.raises(CallRunnerError):
            run.sync(agent, "test")


# ---------------------------------------------------------------------------
# Provider auto-resolution
# ---------------------------------------------------------------------------


class TestProviderAutoResolve:
    async def test_explicit_provider_used(self) -> None:
        """When provider is given explicitly, it is used directly."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="explicit")])

        result = await run(agent, "test", provider=provider)

        assert result.output == "explicit"


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Lightweight stream chunk for testing (mirrors StreamChunk fields)."""

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


class _FakeToolCallDelta:
    """Mirrors ToolCallDelta fields for testing."""

    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_stream_provider(
    stream_rounds: list[list[_FakeStreamChunk]],
) -> Any:
    """Create a mock provider with stream() returning pre-defined chunks.

    Each call to stream() consumes the next list of chunks from stream_rounds.
    """
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    # Also give it a complete() for agent._execute_tools (not needed for stream tests
    # but avoids AttributeError if somehow called)
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# run.stream() tests
# ---------------------------------------------------------------------------


class TestRunStream:
    async def test_stream_text_events(self) -> None:
        """run.stream() yields TextEvent for each text delta."""
        agent = Agent(name="bot", instructions="Be nice.")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(delta=" world"),
            _FakeStreamChunk(delta="!", finish_reason="stop"),
        ]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(agent, "Hi", provider=provider):
            events.append(ev)

        assert len(events) == 3
        assert all(isinstance(e, TextEvent) for e in events)
        assert events[0].text == "Hello"  # type: ignore[union-attr]
        assert events[1].text == " world"  # type: ignore[union-attr]
        assert events[2].text == "!"  # type: ignore[union-attr]

    async def test_stream_agent_name(self) -> None:
        """TextEvent includes agent_name."""
        agent = Agent(name="assistant")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].agent_name == "assistant"

    async def test_stream_empty_text(self) -> None:
        """Empty deltas are not yielded as events."""
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta=""),
            _FakeStreamChunk(delta="data"),
            _FakeStreamChunk(delta=""),
        ]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "data"

    async def test_stream_planning_enabled_injects_plan_before_streaming(self) -> None:
        """run.stream() performs the planner pre-pass before executor streaming."""
        plan_calls: list[dict[str, Any]] = []
        stream_calls: list[dict[str, Any]] = []

        class PlanningStreamProvider:
            async def complete(self, messages: Any, **kwargs: Any) -> Any:
                plan_calls.append({"messages": list(messages), "kwargs": kwargs})

                class FakeResponse:
                    content = "1. Gather the facts."
                    tool_calls: ClassVar[list[ToolCall]] = []
                    usage = Usage()

                return FakeResponse()

            async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
                stream_calls.append({"messages": list(messages), "kwargs": kwargs})
                yield _FakeStreamChunk(delta="Done.", finish_reason="stop")

        agent = Agent(
            name="stream-bot",
            instructions="Execute the task.",
            planning_enabled=True,
            planning_instructions="Return a short numbered plan.",
        )
        provider = PlanningStreamProvider()

        events = [ev async for ev in run.stream(agent, "Handle the task", provider=provider)]

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "Done."
        assert plan_calls[0]["messages"][0].content == "Return a short numbered plan."
        assert "Planner output:\n1. Gather the facts." in stream_calls[0]["messages"][-1].content


class TestRunStreamToolCalls:
    async def test_stream_tool_call_events(self) -> None:
        """run.stream() yields ToolCallEvent for tool calls."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])

        # First round: tool call streamed via deltas
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":'),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        # Second round: text response after tool execution
        round2 = [
            _FakeStreamChunk(delta="Done!"),
        ]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "Greet Alice", provider=provider)]

        # Should have: ToolCallEvent, then TextEvent("Done!")
        assert len(events) == 2
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].tool_name == "greet"
        assert events[0].tool_call_id == "tc1"
        assert events[0].agent_name == "bot"
        assert isinstance(events[1], TextEvent)
        assert events[1].text == "Done!"

    async def test_stream_multiple_tool_calls(self) -> None:
        """run.stream() handles multiple parallel tool calls."""

        @tool
        def add(a: int, b: int) -> str:
            """Add numbers."""
            return str(a + b)

        @tool
        def mul(a: int, b: int) -> str:
            """Multiply numbers."""
            return str(a * b)

        agent = Agent(name="calc", tools=[add, mul])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="add"),
                    _FakeToolCallDelta(index=1, id="tc2", name="mul"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"a":2,"b":3}'),
                    _FakeToolCallDelta(index=1, arguments='{"a":4,"b":5}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Results ready.")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "compute", provider=provider)]

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        text_events = [e for e in events if isinstance(e, TextEvent)]

        assert len(tool_events) == 2
        assert {e.tool_name for e in tool_events} == {"add", "mul"}
        assert len(text_events) == 1
        assert text_events[0].text == "Results ready."

    async def test_stream_tool_error_continues(self) -> None:
        """Tool execution errors don't crash the stream."""

        @tool
        def fail_tool() -> str:
            """Always fails."""
            raise ValueError("oops")

        agent = Agent(name="bot", tools=[fail_tool])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="fail_tool"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Tool failed.")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "try", provider=provider)]

        assert any(isinstance(e, ToolCallEvent) for e in events)
        assert any(isinstance(e, TextEvent) and e.text == "Tool failed." for e in events)


# ---------------------------------------------------------------------------
# ToolCallDeltaEvent streaming tests
# ---------------------------------------------------------------------------


class TestRunStreamToolCallDeltas:
    async def test_stream_tool_call_delta_events_detailed(self) -> None:
        """detailed=True emits ToolCallDeltaEvent for each provider delta."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":'),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [
            ev async for ev in run.stream(agent, "Greet Alice", provider=provider, detailed=True)
        ]

        deltas = [e for e in events if isinstance(e, ToolCallDeltaEvent)]
        assert len(deltas) == 3

        # First delta carries id and name
        assert deltas[0].tool_call_id == "tc1"
        assert deltas[0].tool_name == "greet"
        assert deltas[0].arguments_delta == ""
        assert deltas[0].index == 0
        assert deltas[0].agent_name == "bot"

        # Subsequent deltas carry argument fragments
        assert deltas[1].tool_call_id == ""
        assert deltas[1].tool_name == ""
        assert deltas[1].arguments_delta == '{"name":'

        assert deltas[2].arguments_delta == '"Alice"}'

    async def test_tool_call_delta_not_emitted_without_detailed(self) -> None:
        """Default detailed=False produces no ToolCallDeltaEvent."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "Greet Alice", provider=provider)]

        assert not any(isinstance(e, ToolCallDeltaEvent) for e in events)

    async def test_tool_call_event_includes_arguments(self) -> None:
        """ToolCallEvent now carries the fully assembled arguments."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "Greet Alice", provider=provider)]

        tc_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tc_events) == 1
        assert tc_events[0].arguments == '{"name":"Alice"}'

    async def test_parallel_tool_call_deltas(self) -> None:
        """Parallel tool calls emit deltas with correct index values."""

        @tool
        def add(a: int, b: int) -> str:
            """Add numbers."""
            return str(a + b)

        @tool
        def mul(a: int, b: int) -> str:
            """Multiply numbers."""
            return str(a * b)

        agent = Agent(name="calc", tools=[add, mul])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="add"),
                    _FakeToolCallDelta(index=1, id="tc2", name="mul"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"a":2,"b":3}'),
                    _FakeToolCallDelta(index=1, arguments='{"a":4,"b":5}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Results ready.")]
        provider = _make_stream_provider([round1, round2])

        events = [
            ev async for ev in run.stream(agent, "compute", provider=provider, detailed=True)
        ]

        deltas = [e for e in events if isinstance(e, ToolCallDeltaEvent)]
        # 2 deltas in first chunk (one per tool) + 2 in second chunk
        assert len(deltas) == 4

        # First chunk: names and ids
        idx0_deltas = [d for d in deltas if d.index == 0]
        idx1_deltas = [d for d in deltas if d.index == 1]
        assert len(idx0_deltas) == 2
        assert len(idx1_deltas) == 2
        assert idx0_deltas[0].tool_name == "add"
        assert idx1_deltas[0].tool_name == "mul"
        assert idx0_deltas[1].arguments_delta == '{"a":2,"b":3}'
        assert idx1_deltas[1].arguments_delta == '{"a":4,"b":5}'

    async def test_tool_call_delta_event_types_filter(self) -> None:
        """event_types filter can select only tool_call_delta events."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [
            ev
            async for ev in run.stream(
                agent,
                "Greet Alice",
                provider=provider,
                detailed=True,
                event_types={"tool_call_delta"},
            )
        ]

        assert len(events) == 2
        assert all(isinstance(e, ToolCallDeltaEvent) for e in events)


class TestRunStreamCompletion:
    async def test_stream_completes_on_text_only(self) -> None:
        """Stream ends after a text-only response (no tool calls)."""
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta="All done."),
        ]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)

    async def test_stream_no_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run.stream() raises when no provider is available."""
        from exo import runner as runner_mod
        from exo.agent import AgentError

        monkeypatch.setattr(runner_mod, "_resolve_provider", lambda _agent: None)
        agent = Agent(name="bot")

        with pytest.raises(AgentError, match="requires a provider"):
            async for _ in run.stream(agent, "test"):
                pass

    async def test_stream_max_steps(self) -> None:
        """Stream respects max_steps limit."""

        @tool
        def echo() -> str:
            """Echo."""
            return "echoed"

        agent = Agent(name="bot", tools=[echo], max_steps=2)

        # Every round returns tool calls — should stop after max_steps
        tool_round = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="echo"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        provider = _make_stream_provider([tool_round, tool_round, tool_round])

        events = [ev async for ev in run.stream(agent, "loop", provider=provider)]

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        # Should have at most max_steps (2) tool call events
        assert len(tool_events) <= 2

    async def test_stream_text_and_tool_mixed(self) -> None:
        """Stream handles chunks with both text and tool call deltas."""

        @tool
        def compute() -> str:
            """Compute."""
            return "42"

        agent = Agent(name="bot", tools=[compute])

        round1 = [
            _FakeStreamChunk(
                delta="Thinking...",
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="compute"),
                ],
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Answer: 42")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "what?", provider=provider)]

        types = [type(e).__name__ for e in events]
        assert "TextEvent" in types
        assert "ToolCallEvent" in types


# ---------------------------------------------------------------------------
# run.stream() detailed=False backward compatibility
# ---------------------------------------------------------------------------


class TestRunStreamDetailedFalse:
    """Verify that detailed=False (default) only emits TextEvent and ToolCallEvent."""

    async def test_default_no_rich_events(self) -> None:
        """Default (detailed=False) emits only TextEvent — no StatusEvent/StepEvent."""
        agent = Agent(name="bot", instructions="Be nice.")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(delta="!", finish_reason="stop"),
        ]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "Hi", provider=provider)]

        assert all(isinstance(e, TextEvent) for e in events)
        assert len(events) == 2

    async def test_explicit_false_no_rich_events(self) -> None:
        """Explicit detailed=False still only emits TextEvent/ToolCallEvent."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [
            ev async for ev in run.stream(agent, "Greet Alice", provider=provider, detailed=False)
        ]

        for ev in events:
            assert isinstance(ev, (TextEvent, ToolCallEvent))
            assert not isinstance(ev, ToolCallDeltaEvent)


# ---------------------------------------------------------------------------
# run.stream() detailed=True tests
# ---------------------------------------------------------------------------


class TestRunStreamDetailedText:
    """Tests for detailed=True with text-only responses."""

    async def test_detailed_text_emits_status_and_step_events(self) -> None:
        """detailed=True wraps text response with StatusEvent and StepEvent."""
        agent = Agent(name="bot", instructions="Be nice.")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(
                delta="!",
                finish_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(agent, "Hi", provider=provider, detailed=True):
            events.append(ev)

        types = [type(e).__name__ for e in events]

        # StatusEvent('starting') first
        assert isinstance(events[0], StatusEvent)
        assert events[0].status == "starting"
        assert events[0].agent_name == "bot"

        # StepEvent(status='started') next
        assert isinstance(events[1], StepEvent)
        assert events[1].status == "started"
        assert events[1].step_number == 1

        # Text events in the middle
        assert "TextEvent" in types

        # UsageEvent after LLM stream
        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage_events) == 1
        assert usage_events[0].usage.total_tokens == 15
        assert usage_events[0].step_number == 1
        assert usage_events[0].agent_name == "bot"

        # StepEvent(status='completed') near the end
        step_completed = [e for e in events if isinstance(e, StepEvent) and e.status == "completed"]
        assert len(step_completed) == 1
        assert step_completed[0].completed_at is not None
        assert step_completed[0].usage is not None
        assert step_completed[0].usage.total_tokens == 15

        # StatusEvent('completed') at the end
        assert isinstance(events[-1], StatusEvent)
        assert events[-1].status == "completed"

    async def test_detailed_text_event_order(self) -> None:
        """Verify exact event type order for a simple text response."""
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "Hi", provider=provider, detailed=True)]

        type_names = [type(e).__name__ for e in events]
        assert type_names == [
            "StatusEvent",  # starting
            "StepEvent",  # started
            "TextEvent",  # text delta
            "UsageEvent",  # usage after LLM
            "StepEvent",  # completed
            "StatusEvent",  # completed
        ]


class TestRunStreamDetailedToolCalls:
    """Tests for detailed=True with tool call responses."""

    async def test_detailed_tool_call_emits_tool_result_events(self) -> None:
        """detailed=True emits ToolResultEvent for each tool execution."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":"Alice"}'),
                ],
                finish_reason="tool_calls",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [
            ev async for ev in run.stream(agent, "Greet Alice", provider=provider, detailed=True)
        ]

        tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_result_events) == 1
        tre = tool_result_events[0]
        assert tre.tool_name == "greet"
        assert tre.tool_call_id == "tc1"
        assert tre.arguments == {"name": "Alice"}
        assert tre.result == "Hi Alice!"
        assert tre.success is True
        assert tre.error is None
        assert tre.duration_ms > 0
        assert tre.agent_name == "bot"

    async def test_detailed_tool_call_full_event_order(self) -> None:
        """Verify event type order for a tool call + text response."""

        @tool
        def compute() -> str:
            """Compute."""
            return "42"

        agent = Agent(name="bot", tools=[compute])
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="compute"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Result: 42")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "calc", provider=provider, detailed=True)]

        type_names = [type(e).__name__ for e in events]
        # Step 1: tool call
        assert type_names == [
            "StatusEvent",  # starting
            "StepEvent",  # step 1 started
            "ToolCallDeltaEvent",  # delta: id + name
            "ToolCallDeltaEvent",  # delta: arguments
            "UsageEvent",  # usage after LLM call
            "ToolCallEvent",  # tool call notification (complete)
            "ToolResultEvent",  # tool result
            "StepEvent",  # step 1 completed
            # Step 2: text response
            "StepEvent",  # step 2 started
            "TextEvent",  # text delta
            "UsageEvent",  # usage after LLM call
            "StepEvent",  # step 2 completed
            "StatusEvent",  # completed
        ]

    async def test_detailed_failed_tool_result_event(self) -> None:
        """ToolResultEvent captures failure when a tool raises."""

        @tool
        def fail_tool() -> str:
            """Always fails."""
            raise ValueError("oops")

        agent = Agent(name="bot", tools=[fail_tool])
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="fail_tool"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Tool failed.")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "try", provider=provider, detailed=True)]

        tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_result_events) == 1
        tre = tool_result_events[0]
        assert tre.tool_name == "fail_tool"
        assert tre.success is False
        assert tre.error is not None
        assert "oops" in tre.error

    async def test_detailed_multiple_tool_calls(self) -> None:
        """detailed=True emits ToolResultEvent for each parallel tool call."""

        @tool
        def add(a: int, b: int) -> str:
            """Add numbers."""
            return str(a + b)

        @tool
        def mul(a: int, b: int) -> str:
            """Multiply numbers."""
            return str(a * b)

        agent = Agent(name="calc", tools=[add, mul])
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="add"),
                    _FakeToolCallDelta(index=1, id="tc2", name="mul"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"a":2,"b":3}'),
                    _FakeToolCallDelta(index=1, arguments='{"a":4,"b":5}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done.")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "compute", provider=provider, detailed=True)]

        tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_result_events) == 2
        names = {e.tool_name for e in tool_result_events}
        assert names == {"add", "mul"}

        # Check results
        add_result = next(e for e in tool_result_events if e.tool_name == "add")
        assert add_result.result == "5"
        assert add_result.arguments == {"a": 2, "b": 3}

        mul_result = next(e for e in tool_result_events if e.tool_name == "mul")
        assert mul_result.result == "20"
        assert mul_result.arguments == {"a": 4, "b": 5}


class TestRunStreamDetailedUsage:
    """Tests for UsageEvent emission with detailed=True."""

    async def test_usage_event_captures_token_counts(self) -> None:
        """UsageEvent correctly captures token counts from stream chunks."""
        agent = Agent(name="bot", model="gpt-4")
        chunks = [
            _FakeStreamChunk(delta="Hi"),
            _FakeStreamChunk(
                delta="!",
                finish_reason="stop",
                usage=Usage(input_tokens=50, output_tokens=10, total_tokens=60),
            ),
        ]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "Hi", provider=provider, detailed=True)]

        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage_events) == 1
        assert usage_events[0].usage.input_tokens == 50
        assert usage_events[0].usage.output_tokens == 10
        assert usage_events[0].usage.total_tokens == 60
        assert usage_events[0].model == "gpt-4"
        assert usage_events[0].step_number == 1

    async def test_usage_event_zero_when_no_usage(self) -> None:
        """UsageEvent has zero tokens when chunks don't include usage."""
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "Hi", provider=provider, detailed=True)]

        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage_events) == 1
        assert usage_events[0].usage.total_tokens == 0


class TestRunStreamDetailedErrors:
    """Tests for ErrorEvent emission."""

    async def test_error_event_on_provider_error(self) -> None:
        """ErrorEvent emitted when provider stream raises an exception."""
        agent = Agent(name="bot")

        async def failing_stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("connection failed")
            yield

        mock = AsyncMock()
        mock.stream = failing_stream

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="connection failed"):
            async for ev in run.stream(agent, "Hi", provider=mock, detailed=True):
                events.append(ev)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error == "connection failed"
        assert error_events[0].error_type == "RuntimeError"
        assert error_events[0].agent_name == "bot"
        assert error_events[0].step_number == 1
        assert error_events[0].recoverable is False

    async def test_error_event_emitted_without_detailed(self) -> None:
        """ErrorEvent emitted on errors regardless of detailed flag."""
        agent = Agent(name="bot")

        async def failing_stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("oops")
            yield

        mock = AsyncMock()
        mock.stream = failing_stream

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="oops"):
            async for ev in run.stream(agent, "Hi", provider=mock, detailed=False):
                events.append(ev)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error == "oops"

    async def test_error_event_includes_status_when_detailed(self) -> None:
        """When detailed=True, error also emits StatusEvent(status='error')."""
        agent = Agent(name="bot")

        async def failing_stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise ValueError("bad input")
            yield

        mock = AsyncMock()
        mock.stream = failing_stream

        events: list[StreamEvent] = []
        with pytest.raises(ValueError, match="bad input"):
            async for ev in run.stream(agent, "Hi", provider=mock, detailed=True):
                events.append(ev)

        status_events = [e for e in events if isinstance(e, StatusEvent) and e.status == "error"]
        assert len(status_events) == 1
        assert status_events[0].message == "bad input"


class TestRunStreamDetailedStepNumbers:
    """Tests for correct step numbering with detailed=True."""

    async def test_multi_step_step_numbers(self) -> None:
        """Step numbers increment correctly across multiple steps."""

        @tool
        def echo() -> str:
            """Echo."""
            return "echoed"

        agent = Agent(name="bot", tools=[echo])
        tool_round = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="echo"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        text_round = [_FakeStreamChunk(delta="Final answer")]
        provider = _make_stream_provider([tool_round, text_round])

        events = [ev async for ev in run.stream(agent, "go", provider=provider, detailed=True)]

        step_events = [e for e in events if isinstance(e, StepEvent)]
        # Step 1 started + completed, Step 2 started + completed
        assert len(step_events) == 4
        assert step_events[0].step_number == 1
        assert step_events[0].status == "started"
        assert step_events[1].step_number == 1
        assert step_events[1].status == "completed"
        assert step_events[2].step_number == 2
        assert step_events[2].status == "started"
        assert step_events[3].step_number == 2
        assert step_events[3].status == "completed"

        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage_events) == 2
        assert usage_events[0].step_number == 1
        assert usage_events[1].step_number == 2


# ---------------------------------------------------------------------------
# run.stream() event_types filtering
# ---------------------------------------------------------------------------


class TestRunStreamEventFiltering:
    """Tests for event_types parameter — filtering which events are yielded."""

    async def test_filter_text_only(self) -> None:
        """event_types={'text'} yields only TextEvent."""
        agent = Agent(name="bot", instructions="Be nice.")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(delta="!", finish_reason="stop"),
        ]
        provider = _make_stream_provider([chunks])

        events = [
            ev
            async for ev in run.stream(
                agent, "Hi", provider=provider, detailed=True, event_types={"text"}
            )
        ]

        assert len(events) == 2
        assert all(isinstance(e, TextEvent) for e in events)

    async def test_filter_tool_result_only(self) -> None:
        """event_types={'tool_result'} yields only ToolResultEvent."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [
            ev
            async for ev in run.stream(
                agent,
                "Greet Alice",
                provider=provider,
                detailed=True,
                event_types={"tool_result"},
            )
        ]

        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert events[0].tool_name == "greet"

    async def test_filter_multiple_types(self) -> None:
        """event_types={'text', 'tool_call'} yields only those two types."""

        @tool
        def compute() -> str:
            """Compute."""
            return "42"

        agent = Agent(name="bot", tools=[compute])
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="compute"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Result: 42")]
        provider = _make_stream_provider([round1, round2])

        events = [
            ev
            async for ev in run.stream(
                agent,
                "calc",
                provider=provider,
                detailed=True,
                event_types={"text", "tool_call"},
            )
        ]

        for ev in events:
            assert isinstance(ev, (TextEvent, ToolCallEvent))
        assert any(isinstance(e, TextEvent) for e in events)
        assert any(isinstance(e, ToolCallEvent) for e in events)

    async def test_filter_none_passes_all(self) -> None:
        """event_types=None (default) passes all events through."""
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        events = [
            ev
            async for ev in run.stream(
                agent, "Hi", provider=provider, detailed=True, event_types=None
            )
        ]

        type_names = {type(e).__name__ for e in events}
        # Should have StatusEvent, StepEvent, TextEvent, UsageEvent at minimum
        assert "StatusEvent" in type_names
        assert "TextEvent" in type_names

    async def test_filter_empty_set_yields_nothing(self) -> None:
        """event_types=set() yields no events."""
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        events = [
            ev
            async for ev in run.stream(
                agent, "Hi", provider=provider, detailed=True, event_types=set()
            )
        ]

        assert len(events) == 0

    async def test_filter_without_detailed(self) -> None:
        """event_types works with detailed=False (default mode)."""
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(delta=" world"),
        ]
        provider = _make_stream_provider([chunks])

        # Filter to only text — should get TextEvents (same as no filter, since
        # detailed=False only emits TextEvent/ToolCallEvent anyway)
        events = [
            ev async for ev in run.stream(agent, "Hi", provider=provider, event_types={"text"})
        ]

        assert len(events) == 2
        assert all(isinstance(e, TextEvent) for e in events)

    async def test_filter_error_events(self) -> None:
        """Error events are filtered by event_types like any other event."""
        agent = Agent(name="bot")

        async def failing_stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("connection failed")
            yield

        mock = AsyncMock()
        mock.stream = failing_stream

        # Filter to text only — error events should be filtered out
        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="connection failed"):
            async for ev in run.stream(
                agent, "Hi", provider=mock, detailed=True, event_types={"text"}
            ):
                events.append(ev)

        # No events should pass the filter (error event is filtered out)
        assert len(events) == 0

    async def test_filter_preserves_error_when_included(self) -> None:
        """Error events pass through when 'error' is in event_types."""
        agent = Agent(name="bot")

        async def failing_stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("connection failed")
            yield

        mock = AsyncMock()
        mock.stream = failing_stream

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="connection failed"):
            async for ev in run.stream(
                agent, "Hi", provider=mock, detailed=True, event_types={"error"}
            ):
                events.append(ev)

        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)


# ---------------------------------------------------------------------------
# run.stream() streaming event metrics
# ---------------------------------------------------------------------------


class TestRunStreamEventMetrics:
    """Verify that run.stream() with detailed=True records total events emitted."""

    @pytest.fixture(autouse=True)
    def _reset_metrics(self) -> None:
        from unittest.mock import patch as mock_patch

        from exo.observability.metrics import reset_metrics

        reset_metrics()
        with mock_patch("exo.runner.HAS_OTEL", False):
            yield  # type: ignore[misc]

    async def test_detailed_records_event_counts(self) -> None:
        """detailed=True records stream_events_emitted counter."""
        from exo.observability.metrics import get_metrics_snapshot

        agent = Agent(name="bot", instructions="Be nice.")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(
                delta="!",
                finish_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(agent, "Hi", provider=provider, detailed=True):
            events.append(ev)

        snap = get_metrics_snapshot()
        assert "stream_events_emitted" in snap["counters"]
        assert snap["counters"]["stream_events_emitted"] > 0

    async def test_detailed_false_no_event_metrics(self) -> None:
        """detailed=False does not record stream event metrics."""
        from exo.observability.metrics import get_metrics_snapshot

        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="Hello")]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "Hi", provider=provider)]
        assert len(events) == 1

        snap = get_metrics_snapshot()
        assert "stream_events_emitted" not in snap["counters"]

    async def test_event_type_breakdown(self) -> None:
        """Event metrics include type breakdown via STREAM_EVENT_TYPE attribute."""
        from exo.observability.metrics import get_metrics_snapshot

        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(
                delta="!",
                finish_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "Hi", provider=provider, detailed=True)]

        # Should have text, status, step, usage events
        snap = get_metrics_snapshot()
        assert snap["counters"]["stream_events_emitted"] == len(events)


# ---------------------------------------------------------------------------
# run.stream() hook parity — PRE_LLM_CALL / POST_LLM_CALL
# ---------------------------------------------------------------------------


class TestRunStreamHookParity:
    """Verify that _stream() fires PRE_LLM_CALL and POST_LLM_CALL hooks."""

    async def test_stream_fires_pre_llm_call(self) -> None:
        """PRE_LLM_CALL hook fires with messages kwarg."""
        from exo.hooks import HookPoint

        captured: list[dict[str, Any]] = []

        async def on_pre(*, agent: Any, messages: Any, **_: Any) -> None:
            captured.append({"agent": agent, "messages": messages})

        agent = Agent(name="bot", hooks=[(HookPoint.PRE_LLM_CALL, on_pre)])
        chunks = [_FakeStreamChunk(delta="hi")]
        provider = _make_stream_provider([chunks])

        _ = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(captured) == 1
        assert captured[0]["agent"] is agent
        assert len(captured[0]["messages"]) > 0

    async def test_stream_fires_post_llm_call(self) -> None:
        """POST_LLM_CALL hook fires with response containing streamed text."""
        from exo.hooks import HookPoint

        captured: list[Any] = []

        async def on_post(*, agent: Any, response: Any, **_: Any) -> None:
            captured.append(response)

        agent = Agent(name="bot", hooks=[(HookPoint.POST_LLM_CALL, on_post)])
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(delta=" world"),
        ]
        provider = _make_stream_provider([chunks])

        _ = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(captured) == 1
        assert captured[0].content == "Hello world"

    async def test_stream_hook_order(self) -> None:
        """Hooks fire in order: pre_llm then post_llm."""
        from exo.hooks import HookPoint

        events_log: list[str] = []

        async def on_pre(**_: Any) -> None:
            events_log.append("pre_llm")

        async def on_post(**_: Any) -> None:
            events_log.append("post_llm")

        agent = Agent(
            name="bot",
            hooks=[
                (HookPoint.PRE_LLM_CALL, on_pre),
                (HookPoint.POST_LLM_CALL, on_post),
            ],
        )
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        _ = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert events_log == ["pre_llm", "post_llm"]

    async def test_stream_hooks_fire_per_step(self) -> None:
        """Hooks fire once per LLM step (2-step tool run fires hooks twice each)."""
        from exo.hooks import HookPoint

        events_log: list[str] = []

        async def on_pre(**_: Any) -> None:
            events_log.append("pre")

        async def on_post(**_: Any) -> None:
            events_log.append("post")

        @tool
        def echo() -> str:
            """Echo."""
            return "echoed"

        agent = Agent(
            name="bot",
            tools=[echo],
            hooks=[
                (HookPoint.PRE_LLM_CALL, on_pre),
                (HookPoint.POST_LLM_CALL, on_post),
            ],
        )

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="echo"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done")]
        provider = _make_stream_provider([round1, round2])

        _ = [ev async for ev in run.stream(agent, "go", provider=provider)]

        assert events_log == ["pre", "post", "pre", "post"]

    async def test_stream_post_llm_includes_tool_calls(self) -> None:
        """POST_LLM_CALL response has non-empty tool_calls when tools are called."""
        from exo.hooks import HookPoint

        captured: list[Any] = []

        async def on_post(*, agent: Any, response: Any, **_: Any) -> None:
            captured.append(response)

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(
            name="bot",
            tools=[greet],
            hooks=[(HookPoint.POST_LLM_CALL, on_post)],
        )

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Done")]
        provider = _make_stream_provider([round1, round2])

        _ = [ev async for ev in run.stream(agent, "go", provider=provider)]

        # First POST_LLM_CALL should have tool_calls
        assert len(captured) == 2
        assert len(captured[0].tool_calls) == 1
        assert captured[0].tool_calls[0].name == "greet"
        assert captured[0].finish_reason == "tool_calls"
        # Second POST_LLM_CALL should have no tool_calls (text response)
        assert len(captured[1].tool_calls) == 0
        assert captured[1].finish_reason == "stop"

    async def test_stream_pre_llm_error_emits_error_event(self) -> None:
        """PRE_LLM_CALL hook error is caught and yields ErrorEvent."""
        from exo.hooks import HookPoint

        async def bad_hook(**_: Any) -> None:
            raise RuntimeError("hook exploded")

        agent = Agent(name="bot", hooks=[(HookPoint.PRE_LLM_CALL, bad_hook)])
        chunks = [_FakeStreamChunk(delta="hi")]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="hook exploded"):
            async for ev in run.stream(agent, "test", provider=provider):
                events.append(ev)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert "hook exploded" in error_events[0].error

    async def test_stream_post_llm_error_emits_error_event(self) -> None:
        """POST_LLM_CALL hook error is caught and yields ErrorEvent."""
        from exo.hooks import HookPoint

        async def bad_hook(**_: Any) -> None:
            raise RuntimeError("post hook boom")

        agent = Agent(name="bot", hooks=[(HookPoint.POST_LLM_CALL, bad_hook)])
        chunks = [_FakeStreamChunk(delta="hi")]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="post hook boom"):
            async for ev in run.stream(agent, "test", provider=provider):
                events.append(ev)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert "post hook boom" in error_events[0].error
