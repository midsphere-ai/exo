"""Tests for Programmatic Tool Calling (PTC).

All tests use mock providers — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent, AgentError
from exo.hooks import HookPoint
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.ptc import (
    PTC_TOOL_NAME,
    PTCExecutor,
    PTCTool,
    build_tool_signatures,
    get_ptc_eligible_tools,
    schema_to_python_sig,
)
from exo.runner import run
from exo.tool import Tool, tool
from exo.types import (
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
)

# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def search(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return json.dumps([{"title": f"Result for {query}", "rank": i} for i in range(max_results)])


@tool
def failing_tool(msg: str) -> str:
    """A tool that always fails."""
    raise ValueError(f"Intentional error: {msg}")


# ---------------------------------------------------------------------------
# Mock provider helpers
# ---------------------------------------------------------------------------


def _mock_provider(
    content: str = "Done!",
    tool_calls: list[ToolCall] | None = None,
) -> AsyncMock:
    resp = ModelResponse(
        id="resp-1",
        model="test-model",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


def _multi_step_provider(*responses: ModelResponse) -> AsyncMock:
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=list(responses))
    return provider


# ---------------------------------------------------------------------------
# Init & registration tests
# ---------------------------------------------------------------------------


class TestPTCInit:
    def test_ptc_false_by_default(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        assert agent.ptc is False
        assert PTC_TOOL_NAME not in agent.tools

    def test_ptc_true_registers_ptc_tool(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert agent.ptc is True
        assert PTC_TOOL_NAME in agent.tools
        assert isinstance(agent.tools[PTC_TOOL_NAME], PTCTool)

    def test_ptc_timeout_default(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert agent.ptc_timeout == 60

    def test_ptc_timeout_custom(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True, ptc_timeout=120)
        assert agent.ptc_timeout == 120

    def test_ptc_collision_raises(self) -> None:
        @tool(name=PTC_TOOL_NAME)
        def my_tool() -> str:
            """Conflicting name."""
            return "x"

        with pytest.raises(AgentError, match=r"already registered"):
            Agent(name="bot", tools=[my_tool], ptc=True)

    def test_ptc_tool_is_tool_subclass(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert isinstance(agent.tools[PTC_TOOL_NAME], Tool)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestPTCSchemas:
    def test_schemas_exclude_user_tools(self) -> None:
        """PTC-eligible tools should NOT appear as individual schemas."""
        agent = Agent(name="bot", tools=[greet, add], ptc=True)
        schemas = agent.get_tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}

        assert PTC_TOOL_NAME in schema_names
        assert "greet" not in schema_names
        assert "add" not in schema_names
        # retrieve_artifact is always auto-registered and should stay direct
        assert "retrieve_artifact" in schema_names

    def test_schemas_hitl_tools_stay_direct(self) -> None:
        """HITL tools must remain as direct schemas, not PTC-wrapped."""
        agent = Agent(
            name="bot",
            tools=[greet, add],
            hitl_tools=["greet"],
            ptc=True,
        )
        schemas = agent.get_tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}

        assert PTC_TOOL_NAME in schema_names
        assert "greet" in schema_names  # HITL → stays direct
        assert "add" not in schema_names  # not HITL → PTC-wrapped

    def test_description_lists_tool_signatures(self) -> None:
        agent = Agent(name="bot", tools=[greet, search], ptc=True)
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        desc = ptc_tool.description

        assert "async def greet" in desc
        assert "name: str" in desc
        assert "async def search" in desc
        assert "query: str" in desc
        assert "max_results: int" in desc

    def test_description_excludes_framework_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description

        assert "retrieve_artifact" not in desc
        assert PTC_TOOL_NAME not in desc

    def test_description_excludes_hitl_tools(self) -> None:
        agent = Agent(
            name="bot",
            tools=[greet, add],
            hitl_tools=["greet"],
            ptc=True,
        )
        desc = agent.tools[PTC_TOOL_NAME].description

        assert "greet" not in desc  # HITL tools not in PTC namespace
        assert "async def add" in desc

    async def test_dynamic_tool_add_updates_schemas(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)

        # Cache schemas
        schemas1 = agent.get_tool_schemas()
        schema_names1 = {s["function"]["name"] for s in schemas1}
        assert "add" not in schema_names1

        # Add tool dynamically
        await agent.add_tool(add)

        schemas2 = agent.get_tool_schemas()
        schema_names2 = {s["function"]["name"] for s in schemas2}
        # add should still not be in schemas (it's PTC-eligible)
        assert "add" not in schema_names2
        # But should appear in PTC tool description
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "async def add" in desc


# ---------------------------------------------------------------------------
# Signature generation tests
# ---------------------------------------------------------------------------


class TestSignatureGeneration:
    def test_simple_required_param(self) -> None:
        sig = schema_to_python_sig(greet)
        assert sig == "async def greet(name: str) -> str"

    def test_optional_param(self) -> None:
        sig = schema_to_python_sig(search)
        assert "query: str" in sig
        assert "max_results" in sig

    def test_multiple_required(self) -> None:
        sig = schema_to_python_sig(add)
        assert "a: int" in sig
        assert "b: int" in sig

    def test_build_tool_signatures(self) -> None:
        result = build_tool_signatures({"greet": greet, "add": add})
        assert "async def greet" in result
        assert "async def add" in result
        assert '"""Greet someone by name."""' in result
        assert '"""Add two numbers."""' in result

    def test_build_tool_signatures_empty(self) -> None:
        assert build_tool_signatures({}) == "(no tools available)"


# ---------------------------------------------------------------------------
# PTCExecutor unit tests
# ---------------------------------------------------------------------------


class TestPTCExecutor:
    """Test PTCExecutor.run() directly."""

    def _make_agent(self, tools: list[Tool] | None = None) -> Agent:
        return Agent(name="test", tools=tools or [greet, add, search], ptc=True)

    async def test_simple_print(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('print("hello world")')
        assert result == "hello world"

    async def test_return_value(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("return 42")
        assert result == "42"

    async def test_print_and_return(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('print("line1")\nreturn 99')
        assert "line1" in result
        assert "99" in result

    async def test_tool_call(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('r = await default_api.greet(name="Alice")\nprint(r)')
        assert "Hello, Alice!" in result

    async def test_multiple_tools_sequential(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
r1 = await default_api.greet(name="Alice")
r2 = await default_api.add(a=3, b=4)
print(f"{r1} sum={r2}")
"""
        result = await executor.run(code)
        assert "Hello, Alice!" in result
        assert "sum=7" in result

    async def test_parallel_gather(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
results = await asyncio.gather(
    default_api.greet(name="A"),
    default_api.greet(name="B"),
    default_api.greet(name="C"),
)
for r in results:
    print(r)
"""
        result = await executor.run(code)
        assert "Hello, A!" in result
        assert "Hello, B!" in result
        assert "Hello, C!" in result

    async def test_loop_with_filter(self) -> None:
        """Core PTC use case: loop + filter to reduce context."""
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
data = json.loads(await default_api.search(query="test", max_results=5))
high_rank = [d for d in data if d["rank"] >= 3]
print(json.dumps(high_rank))
"""
        result = await executor.run(code)
        parsed = json.loads(result)
        assert len(parsed) == 2  # rank 3 and 4
        assert all(d["rank"] >= 3 for d in parsed)

    async def test_syntax_error(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("def foo(")
        assert "SyntaxError" in result

    async def test_runtime_error(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("x = 1 / 0")
        assert "ZeroDivisionError" in result

    async def test_tool_error_propagates(self) -> None:
        agent = Agent(name="test", tools=[failing_tool], ptc=True)
        executor = PTCExecutor(agent)
        # Error should appear in the output (caught by outer handler)
        result = await executor.run('await default_api.failing_tool(msg="boom")')
        assert "Intentional error: boom" in result

    async def test_tool_error_catchable_in_code(self) -> None:
        """User code can catch tool errors and continue."""
        agent = Agent(name="test", tools=[failing_tool, greet], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
try:
    await default_api.failing_tool(msg="oops")
except Exception as e:
    print(f"caught: {e}")
r = await default_api.greet(name="OK")
print(r)
"""
        result = await executor.run(code)
        assert "caught:" in result
        assert "Hello, OK!" in result

    async def test_timeout(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent, timeout=1)
        result = await executor.run("await asyncio.sleep(10)")
        assert "TimeoutError" in result

    async def test_stdlib_available(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
import math as m  # also available without import
print(math.sqrt(16))
print(json.dumps({"a": 1}))
print(re.sub(r"\\d+", "X", "abc123"))
"""
        result = await executor.run(code)
        assert "4.0" in result
        assert '{"a": 1}' in result
        assert "abcX" in result

    async def test_no_output(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("x = 1 + 1")
        assert result == "(no output)"


# ---------------------------------------------------------------------------
# Hook tests
# ---------------------------------------------------------------------------


class TestPTCHooks:
    async def test_hooks_fire_for_inner_calls(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        hook_calls: list[tuple[str, str]] = []

        async def capture_hook(**kwargs: Any) -> None:
            hook_calls.append((kwargs.get("tool_name", ""), "called"))

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_hook)
        agent.hook_manager.add(HookPoint.POST_TOOL_CALL, capture_hook)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Test")')

        # PRE + POST for the inner greet call
        tool_names = [name for name, _ in hook_calls]
        assert tool_names.count("greet") == 2

    async def test_hooks_correct_arguments(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        captured_args: list[dict[str, Any]] = []

        async def capture_pre(**kwargs: Any) -> None:
            if kwargs.get("arguments"):
                captured_args.append(kwargs["arguments"])

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_pre)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Alice")')

        assert len(captured_args) == 1
        assert captured_args[0] == {"name": "Alice"}

    async def test_hooks_fire_for_multiple_inner_calls(self) -> None:
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        pre_calls: list[str] = []

        async def capture_pre(**kwargs: Any) -> None:
            pre_calls.append(kwargs.get("tool_name", ""))

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_pre)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        assert "greet" in pre_calls
        assert "add" in pre_calls


# ---------------------------------------------------------------------------
# PTC-eligible tools helper
# ---------------------------------------------------------------------------


class TestPTCEligible:
    def test_excludes_framework_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "greet" in eligible
        assert PTC_TOOL_NAME not in eligible
        assert "retrieve_artifact" not in eligible

    def test_excludes_hitl_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet, add], hitl_tools=["greet"], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "greet" not in eligible
        assert "add" in eligible


# ---------------------------------------------------------------------------
# Integration tests: full agent.run()
# ---------------------------------------------------------------------------


class TestPTCIntegration:
    async def test_end_to_end_run(self) -> None:
        """MockProvider returns PTC tool call → agent runs PTC → final text."""
        tc = ToolCall(
            id="tc-1",
            name=PTC_TOOL_NAME,
            arguments=json.dumps({"code": 'r = await default_api.greet(name="World")\nprint(r)'}),
        )
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        resp_final = ModelResponse(
            content="I greeted World for you!",
            tool_calls=[],
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        provider = _multi_step_provider(resp_tool, resp_final)

        agent = Agent(name="bot", tools=[greet], ptc=True)
        output = await agent.run("Say hi to World", provider=provider)

        assert output.text == "I greeted World for you!"
        assert provider.complete.await_count == 2

    async def test_ptc_tool_result_contains_output(self) -> None:
        """The tool result from PTC tool should contain the code output."""
        tc = ToolCall(
            id="tc-1",
            name=PTC_TOOL_NAME,
            arguments=json.dumps({"code": 'r = await default_api.greet(name="Test")\nprint(r)'}),
        )
        # Track messages sent to provider
        call_count = 0
        captured_msgs: list[Any] = []

        async def fake_complete(messages: Any, **kwargs: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            captured_msgs.append(list(messages))
            if call_count == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[tc],
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            return ModelResponse(
                content="Done",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=fake_complete)

        agent = Agent(name="bot", tools=[greet], ptc=True)
        await agent.run("test", provider=provider)

        # The second call should have the tool result with "Hello, Test!"
        second_call_msgs = captured_msgs[1]
        tool_results = [m for m in second_call_msgs if getattr(m, "role", "") == "tool"]
        assert len(tool_results) == 1
        assert "Hello, Test!" in tool_results[0].content


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestPTCSerialization:
    def test_to_dict_includes_ptc(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True, ptc_timeout=120)
        data = agent.to_dict()
        assert data["ptc"] is True
        assert data["ptc_timeout"] == 120

    def test_to_dict_ptc_false(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        data = agent.to_dict()
        assert data["ptc"] is False

    def test_ptc_tool_not_in_serialized_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        data = agent.to_dict()
        if "tools" in data:
            tool_names = [t.get("name", t) if isinstance(t, dict) else t for t in data["tools"]]
            assert PTC_TOOL_NAME not in str(tool_names)

    def test_describe_includes_ptc(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        info = agent.describe()
        assert info["ptc"] is True


# ---------------------------------------------------------------------------
# Swarm propagation tests
# ---------------------------------------------------------------------------


class TestSwarmPTC:
    def test_swarm_ptc_propagation(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet])
        a2 = Agent(name="a2", tools=[add])
        assert a1.ptc is False
        assert a2.ptc is False

        Swarm(agents=[a1, a2], ptc=True)
        assert a1.ptc is True
        assert a2.ptc is True

    def test_swarm_ptc_none_no_change(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet], ptc=True)
        a2 = Agent(name="a2", tools=[add])

        Swarm(agents=[a1, a2])  # ptc=None by default
        assert a1.ptc is True  # unchanged
        assert a2.ptc is False  # unchanged


# ---------------------------------------------------------------------------
# PTC transparency tests — synthetic event emission
# ---------------------------------------------------------------------------


class TestPTCTransparency:
    """Verify that PTC emits per-tool ToolCallEvent/ToolResultEvent to the queue."""

    async def test_single_tool_emits_events(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Alice")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        assert call_events[0].agent_name == "test"
        assert '"Alice"' in call_events[0].arguments

        assert len(result_events) == 1
        assert result_events[0].tool_name == "greet"
        assert result_events[0].success is True
        assert "Hello, Alice!" in str(result_events[0].result)

    async def test_multiple_tools_emit_ordered_events(self) -> None:
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        tool_events = [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]

        # Expect: call(greet), result(greet), call(add), result(add)
        assert len(tool_events) == 4
        assert tool_events[0].tool_name == "greet"
        assert isinstance(tool_events[0], ToolCallEvent)
        assert tool_events[1].tool_name == "greet"
        assert isinstance(tool_events[1], ToolResultEvent)
        assert tool_events[2].tool_name == "add"
        assert isinstance(tool_events[2], ToolCallEvent)
        assert tool_events[3].tool_name == "add"
        assert isinstance(tool_events[3], ToolResultEvent)

    async def test_error_tool_emits_failure_event(self) -> None:
        agent = Agent(name="test", tools=[failing_tool], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.failing_tool(msg="boom")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert "boom" in (result_events[0].error or "")

    async def test_events_have_unique_call_ids(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.greet(name="B")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(call_events) == 2
        assert call_events[0].tool_call_id != call_events[1].tool_call_id

    async def test_call_and_result_share_call_id(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="X")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_ev = next(e for e in events if isinstance(e, ToolCallEvent))
        result_ev = next(e for e in events if isinstance(e, ToolResultEvent))
        assert call_ev.tool_call_id == result_ev.tool_call_id

    async def test_result_has_duration(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert result_events[0].duration_ms >= 0.0

    async def test_no_ptc_tool_name_in_events(self) -> None:
        """No event should reference the internal PTC tool name."""
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        for ev in events:
            if hasattr(ev, "tool_name"):
                assert ev.tool_name != PTC_TOOL_NAME


# ---------------------------------------------------------------------------
# Streaming + non-streaming integration helpers
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Minimal stream chunk for PTC integration tests."""

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
    """Minimal tool call delta for PTC integration tests."""

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


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
    """Create a mock provider returning pre-defined stream chunks."""
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


def _ptc_tool_call(code: str) -> list[_FakeStreamChunk]:
    """Build a stream round that calls the __exo_ptc__ tool with *code*."""
    return [
        _FakeStreamChunk(
            tool_call_deltas=[
                _FakeToolCallDelta(index=0, id="tc_ptc_1", name=PTC_TOOL_NAME),
            ],
        ),
        _FakeStreamChunk(
            tool_call_deltas=[
                _FakeToolCallDelta(index=0, arguments=json.dumps({"code": code})),
            ],
            finish_reason="tool_calls",
        ),
    ]


# ---------------------------------------------------------------------------
# PTC streaming integration tests
# ---------------------------------------------------------------------------


class TestPTCStreamingIntegration:
    """End-to-end: PTC events flow through run.stream()."""

    async def test_ptc_events_in_stream_detailed(self) -> None:
        """run.stream(detailed=True) yields inner ToolCallEvent + ToolResultEvent."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="Alice")')
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        assert call_events[0].agent_name == "bot"

        assert len(result_events) == 1
        assert result_events[0].tool_name == "greet"
        assert result_events[0].success is True
        assert "Hello, Alice!" in str(result_events[0].result)

    async def test_ptc_tool_call_without_detailed(self) -> None:
        """run.stream(detailed=False) yields ToolCallEvent but NOT ToolResultEvent."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="Bob")')
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=False)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        # ToolResultEvent requires detailed=True
        assert len(result_events) == 0

    async def test_ptc_outer_tool_suppressed(self) -> None:
        """No event should reference __exo_ptc__ in the stream."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="X")')
        round2 = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        for ev in events:
            if hasattr(ev, "tool_name"):
                assert ev.tool_name != PTC_TOOL_NAME

    async def test_ptc_multiple_inner_tools_stream(self) -> None:
        """Multiple PTC inner tool calls emit ordered events in stream."""
        agent = Agent(name="bot", tools=[greet, add], ptc=True)
        code = 'await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)'
        round1 = _ptc_tool_call(code)
        round2 = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 2
        assert call_events[0].tool_name == "greet"
        assert call_events[1].tool_name == "add"
        assert len(result_events) == 2
        assert result_events[0].tool_name == "greet"
        assert result_events[1].tool_name == "add"


# ---------------------------------------------------------------------------
# Non-streaming: queue drain prevents stale events
# ---------------------------------------------------------------------------


class TestPTCNonStreamingDrain:
    """Verify that run() (non-streaming) drains PTC events from the queue."""

    async def test_run_drains_ptc_event_queue(self) -> None:
        """After run(), the agent's _event_queue should be empty."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        provider = _multi_step_provider(
            ModelResponse(
                id="r1",
                model="test",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name=PTC_TOOL_NAME,
                        arguments=json.dumps({"code": 'await default_api.greet(name="Z")'}),
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                id="r2",
                model="test",
                content="Done",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )

        result = await run(agent, "greet Z", provider=provider)

        assert result.output == "Done"
        assert agent._event_queue.empty(), (
            "_event_queue should be empty after non-streaming run() — "
            "PTC events must be drained to prevent memory leaks"
        )

    async def test_run_then_stream_no_stale_events(self) -> None:
        """Stale PTC events from run() must not leak into a later stream()."""
        agent = Agent(name="bot", tools=[greet], ptc=True)

        # First: non-streaming run
        provider1 = _multi_step_provider(
            ModelResponse(
                id="r1",
                model="test",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name=PTC_TOOL_NAME,
                        arguments=json.dumps({"code": 'await default_api.greet(name="Old")'}),
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                id="r2",
                model="test",
                content="done1",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )
        await run(agent, "first", provider=provider1)

        # Second: streaming run (text-only, no tool calls)
        provider2 = _make_stream_provider([[_FakeStreamChunk(delta="Hello stream!")]])
        events = [ev async for ev in run.stream(agent, "second", provider=provider2, detailed=True)]

        # No stale ToolCallEvent/ToolResultEvent from the first run should appear
        tool_events = [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]
        assert len(tool_events) == 0, (
            f"Expected no tool events in text-only stream, got {len(tool_events)}"
        )


# ---------------------------------------------------------------------------
# Robustness tests (H1-H5, M6, M9, M11, M12, L13-L18)
# ---------------------------------------------------------------------------


class TestPTCBaseExceptionHandling:
    """PTC must catch SystemExit / KeyboardInterrupt / MaxToolCallsExceeded."""

    async def test_system_exit_is_caught(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("raise SystemExit(2)")
        assert "SystemExit" in result
        assert "blocked inside PTC" in result

    async def test_system_exit_with_captured_output(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('print("done work")\nraise SystemExit(1)')
        assert "done work" in result
        assert "SystemExit" in result

    async def test_raise_system_exit_with_code_is_caught(self) -> None:
        """``raise SystemExit(3)`` in user code is caught by the inner trap.

        ``import sys`` is now sandbox-blocked, so this tests the bare
        ``raise SystemExit(...)`` form which remains a valid escape attempt.
        """
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("raise SystemExit(3)")
        assert "SystemExit" in result
        assert "code=3" in result

    async def test_cancelled_error_propagates(self) -> None:
        """asyncio.CancelledError MUST propagate, not be swallowed."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Raise CancelledError directly from user code
        with pytest.raises(asyncio.CancelledError):
            await executor.run("raise asyncio.CancelledError()")


class TestPTCStderrCapture:
    """Verify stderr from library code inside tools is captured.

    Since the sandbox blocks ``import sys``, user code cannot write to
    stderr directly.  But libraries invoked *inside* tool execution (e.g.
    ``warnings.warn``, legacy C extensions) may still write to stderr —
    those writes must be captured and returned to the model, not leaked
    to the real terminal.
    """

    async def test_tool_stderr_is_captured(self) -> None:
        import sys as _sys

        @tool
        def noisy() -> str:
            """Write a warning to stderr then return a value."""
            _sys.stderr.write("library warning: about to compute\n")
            return "done"

        agent = Agent(name="t", tools=[noisy], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("r = await default_api.noisy()\nprint(r)")
        assert "library warning" in result
        assert "done" in result


class TestPTCHyphenatedNames:
    """Tools whose name is not a valid Python identifier are excluded from PTC."""

    def test_hyphen_name_excluded_from_ptc(self) -> None:
        @tool(name="get-data")
        def get_data(query: str) -> str:
            """Fetch data."""
            return f"data:{query}"

        agent = Agent(name="t", tools=[get_data, greet], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "get-data" not in eligible
        assert "greet" in eligible

    def test_hyphen_name_stays_as_direct_schema(self) -> None:
        @tool(name="get-data")
        def get_data(query: str) -> str:
            """Fetch data."""
            return f"data:{query}"

        agent = Agent(name="t", tools=[get_data, greet], ptc=True)
        schemas = agent.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        # Hyphenated tool stays as direct schema (so the LLM can still call it)
        assert "get-data" in names
        # greet is absorbed into PTC
        assert "greet" not in names
        assert PTC_TOOL_NAME in names


class TestPTCExcludeAttribute:
    """Tools with _ptc_exclude=True are never wrapped by PTC."""

    def test_ptc_exclude_attribute(self) -> None:
        @tool
        def special(x: int) -> str:
            """Do something special."""
            return str(x)

        special._ptc_exclude = True  # type: ignore[attr-defined]
        agent = Agent(name="t", tools=[special, greet], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "special" not in eligible
        assert "greet" in eligible

        schemas = agent.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "special" in names  # stays as direct schema


class TestPTCJsonDumpsDefault:
    """Tools returning non-JSON-serializable objects are stringified via default=str."""

    async def test_datetime_in_dict_return_serialized(self) -> None:
        import datetime as _dt

        @tool
        def get_info() -> dict:
            """Return a dict with datetime."""
            return {"when": _dt.datetime(2026, 4, 10, 12, 0, 0)}

        agent = Agent(name="t", tools=[get_info], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("r = await default_api.get_info()\nprint(r)")
        # datetime is stringified via default=str; no TypeError
        assert "2026-04-10" in result

    async def test_non_serialisable_falls_back_to_str(self) -> None:
        class Weird:
            def __str__(self) -> str:
                return "weird-object"

        @tool
        def get_weird() -> dict:
            """Return a dict with weird object."""
            return {"thing": Weird()}

        agent = Agent(name="t", tools=[get_weird], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("r = await default_api.get_weird()\nprint(r)")
        # Either json.dumps(default=str) or str(raw) handles this
        assert "weird-object" in result


class TestPTCOutputTruncation:
    async def test_output_is_capped(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_output_bytes=500)
        code = "print('x' * 5000)"
        result = await executor.run(code)
        assert "[truncated" in result
        assert len(result) <= 600  # cap + trailing marker

    async def test_output_under_cap_is_untouched(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_output_bytes=500)
        result = await executor.run("print('hello')")
        assert result == "hello"
        assert "[truncated" not in result


class TestPTCTracebackLineNumbers:
    async def test_traceback_line_numbers_match_user_code(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Line 1 of user code raises
        result = await executor.run("x = 1/0")
        # The rewritten traceback should say line 1, not line 2
        assert "line 1" in result or "ZeroDivisionError" in result
        # And definitely NOT "line 2" from the wrapper
        assert 'File "<ptc>", line 2' not in result

    async def test_traceback_multi_line_user_code(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Line 3 of user code raises
        code = "a = 1\nb = 2\nc = a / 0\n"
        result = await executor.run(code)
        assert "ZeroDivisionError" in result
        # Wrapper adds 1 line, so without rewrite the traceback would say line 4
        assert 'File "<ptc>", line 4' not in result

    async def test_syntax_error_line_number_adjusted(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Line 1 has the syntax error
        result = await executor.run("def foo(")
        assert "SyntaxError" in result


class TestPTCMaxToolCalls:
    async def test_max_tool_calls_enforced(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_tool_calls=3)
        code = """\
for _ in range(10):
    await default_api.greet(name="X")
"""
        result = await executor.run(code)
        assert "max_tool_calls" in result

    async def test_under_cap_runs_normally(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_tool_calls=5)
        code = """\
for _ in range(3):
    r = await default_api.greet(name="X")
print(r)
"""
        result = await executor.run(code)
        assert "Hello, X!" in result

    async def test_max_tool_calls_propagated_from_agent(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True, ptc_max_tool_calls=2)
        assert agent.ptc_max_tool_calls == 2
        assert agent.tools[PTC_TOOL_NAME]._max_tool_calls == 2  # type: ignore[attr-defined]


class TestPTCFullUuid:
    async def test_tool_call_id_is_full_uuid(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(call_events) == 1
        # Full uuid.uuid4().hex is 32 hex chars, prefix 'ptc_'
        assert len(call_events[0].tool_call_id) == 4 + 32
        assert call_events[0].tool_call_id.startswith("ptc_")


class TestPTCOrphanTaskCleanup:
    async def test_orphan_tasks_are_cancelled(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
async def spinner():
    while True:
        await asyncio.sleep(0.01)

asyncio.create_task(spinner())
print("spawned")
"""
        before = asyncio.all_tasks()
        result = await executor.run(code)
        assert "spawned" in result
        # After run, the orphan spinner should be cancelled
        await asyncio.sleep(0.05)
        after = asyncio.all_tasks()
        new_tasks = after - before
        # Any surviving new task should be done (cancelled) or filtered
        for task in new_tasks:
            assert task.done()


class TestPTCArrayItemTypes:
    def test_array_items_shown_in_signature(self) -> None:
        @tool
        def batch(ids: list[str]) -> str:
            """Process a list of ids."""
            return str(ids)

        sig = schema_to_python_sig(batch)
        # Should show list[str], not just list
        assert "list[str]" in sig

    def test_array_without_items_defaults_to_list(self) -> None:
        class ToolNoItems(Tool):
            name = "simple"
            description = "simple"
            parameters = {  # noqa: RUF012
                "type": "object",
                "properties": {
                    "vals": {"type": "array"},
                },
                "required": ["vals"],
            }

            async def execute(self, **kwargs: Any) -> str:
                return ""

        sig = schema_to_python_sig(ToolNoItems())
        assert "vals: list" in sig


class TestPTCDescriptionCache:
    def test_description_cached_when_tools_unchanged(self) -> None:
        agent = Agent(name="t", tools=[greet, add], ptc=True)
        ptc_tool: PTCTool = agent.tools[PTC_TOOL_NAME]  # type: ignore[assignment]
        d1 = ptc_tool.description
        d2 = ptc_tool.description
        # Same content and same object (cache hit)
        assert d1 is d2

    async def test_description_invalidated_on_add_tool(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        ptc_tool: PTCTool = agent.tools[PTC_TOOL_NAME]  # type: ignore[assignment]
        d1 = ptc_tool.description
        assert "async def greet" in d1
        assert "async def add" not in d1

        await agent.add_tool(add)
        d2 = ptc_tool.description
        assert "async def add" in d2
        assert d1 is not d2

    def test_description_contains_instructions_block(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "<instructions>" in desc
        assert "</instructions>" in desc
        assert "<recommended_usage>" in desc
        assert "</recommended_usage>" in desc

    def test_description_emphasises_minimal_code(self) -> None:
        """Key directive: write MINIMAL, PTC-only code."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        # Guidance about minimal code presence
        assert "MINIMAL" in desc
        assert "ONLY write PTC-related" in desc

    def test_code_param_description_points_to_examples(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        code_desc = ptc_tool.parameters["properties"]["code"]["description"]
        # The parameter description should contain examples and a reference
        # to the tool description (not duplicate the full ruleset).
        assert "EXAMPLE" in code_desc
        assert "default_api" in code_desc
        assert "MINIMAL" in code_desc
        assert "See the tool description" in code_desc

    def test_code_param_description_does_not_duplicate_rules(self) -> None:
        """The parameter description should NOT re-list the full instructions block."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        code_desc = ptc_tool.parameters["properties"]["code"]["description"]
        # The full rule block lives in the tool description — param desc stays concise.
        assert "<instructions>" not in code_desc
        assert "<recommended_usage>" not in code_desc
        # Compact: under ~1500 chars (vs the 2000+ of the rule block)
        assert len(code_desc) < 1500


class TestPTCAgentParameters:
    def test_max_output_bytes_parameter(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True, ptc_max_output_bytes=1024)
        assert agent.ptc_max_output_bytes == 1024
        assert agent.tools[PTC_TOOL_NAME]._max_output_bytes == 1024  # type: ignore[attr-defined]

    def test_default_max_output_bytes(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        assert agent.ptc_max_output_bytes == 200_000

    def test_default_max_tool_calls(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        assert agent.ptc_max_tool_calls == 200

    def test_serialisation_round_trip(self) -> None:
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_timeout=90,
            ptc_max_output_bytes=1024,
            ptc_max_tool_calls=50,
        )
        data = agent.to_dict()
        assert data["ptc_max_output_bytes"] == 1024
        assert data["ptc_max_tool_calls"] == 50

        restored = Agent.from_dict(data)
        assert restored.ptc_max_output_bytes == 1024
        assert restored.ptc_max_tool_calls == 50


# ---------------------------------------------------------------------------
# Sandbox tests — restricted builtins, blocked imports, AST pre-scan
# ---------------------------------------------------------------------------


class TestPTCSandboxImports:
    """Blocked stdlib imports must raise a clear ImportError."""

    async def test_import_os_blocked_by_ast_scan(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import os\nprint(os.getcwd())")
        assert "blocked" in result
        assert "os" in result
        assert "default_api" in result

    async def test_import_subprocess_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import subprocess")
        assert "blocked" in result
        assert "subprocess" in result

    async def test_import_sys_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import sys")
        assert "blocked" in result
        assert "sys" in result

    async def test_import_pathlib_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import pathlib")
        assert "blocked" in result

    async def test_import_shutil_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import shutil")
        assert "blocked" in result

    async def test_import_socket_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import socket")
        assert "blocked" in result

    async def test_import_urllib_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import urllib")
        assert "blocked" in result

    async def test_import_threading_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import threading")
        assert "blocked" in result

    async def test_import_multiprocessing_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import multiprocessing")
        assert "blocked" in result

    async def test_import_ctypes_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import ctypes")
        assert "blocked" in result

    async def test_import_inspect_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import inspect")
        assert "blocked" in result

    async def test_from_os_import_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("from os import path")
        assert "blocked" in result

    async def test_from_subprocess_import_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("from subprocess import run")
        assert "blocked" in result

    async def test_import_dotted_blocked(self) -> None:
        """Blocking is by top package name — `os.path` → `os`."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import os.path")
        assert "blocked" in result

    async def test_re_import_of_preloaded_allowed(self) -> None:
        """`import json as j` should work (no-op re-import)."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import json as j\nprint(j.dumps({'a': 1}))")
        assert '{"a": 1}' in result

    async def test_unknown_import_rejected(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import some_random_module")
        assert "not allowed" in result or "blocked" in result


class TestPTCSandboxBuiltins:
    """Dangerous builtins must raise PTCSandboxError with a clear message."""

    async def test_open_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("open('/etc/passwd')")
        assert "blocked" in result
        assert "default_api" in result

    async def test_eval_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("eval('1+1')")
        assert "blocked" in result

    async def test_exec_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("exec('x = 1')")
        assert "blocked" in result

    async def test_compile_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("compile('1+1', '<s>', 'eval')")
        assert "blocked" in result

    async def test_globals_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("globals()")
        assert "blocked" in result

    async def test_locals_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("locals()")
        assert "blocked" in result

    async def test_vars_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("vars()")
        assert "blocked" in result

    async def test_dir_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("dir()")
        assert "blocked" in result

    async def test_input_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("input('> ')")
        assert "blocked" in result

    async def test_blocked_error_is_catchable(self) -> None:
        """User code can catch the sandbox error and continue."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
try:
    open('/etc/passwd')
except Exception as e:
    print(f'caught: {e}')
print('continued')
"""
        result = await executor.run(code)
        assert "caught:" in result
        assert "continued" in result


class TestPTCSandboxAstScan:
    """Static AST pre-scan blocks dunder escape patterns."""

    async def test_dunder_class_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(().__class__)")
        assert "__class__" in result
        assert "blocked" in result

    async def test_dunder_bases_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(object.__bases__)")
        assert "__bases__" in result

    async def test_dunder_subclasses_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("object.__subclasses__()")
        assert "__subclasses__" in result

    async def test_dunder_mro_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(int.__mro__)")
        assert "__mro__" in result

    async def test_dunder_globals_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("f = lambda: 1\nprint(f.__globals__)")
        assert "__globals__" in result

    async def test_dunder_dict_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print({}.__dict__)")
        assert "__dict__" in result

    async def test_dunder_code_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("f = lambda: 1\nprint(f.__code__)")
        assert "__code__" in result

    async def test_chained_dunder_blocked(self) -> None:
        """The classic escape chain must be caught at the first hop."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(().__class__.__bases__[0].__subclasses__())")
        assert "blocked" in result

    async def test_dynamic_getattr_dunder_blocked_at_runtime(self) -> None:
        """Dynamic dunder access via getattr(x, '__class__') is runtime-blocked."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Build the dunder name at runtime so the AST scan cannot see it.
        code = "name = '__' + 'class' + '__'\nprint(getattr(object(), name))"
        result = await executor.run(code)
        # safer_getattr rejects underscore-prefixed names at runtime
        assert "blocked" in result or "invalid attribute" in result or "starts with" in result


class TestPTCSandboxSafeBuiltins:
    """The safe builtin subset must remain functional."""

    async def test_len_range_sum_sorted(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(sum(range(10)))\nprint(len(sorted([3,1,2])))")
        assert "45" in result
        assert "3" in result

    async def test_comprehensions_and_filters(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print([x * 2 for x in range(5) if x % 2 == 0])")
        assert "[0, 4, 8]" in result

    async def test_type_constructors(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run(
            "print(dict(a=1, b=2))\nprint(list((1,2,3)))\nprint(set([1,1,2]))"
        )
        assert "{'a': 1, 'b': 2}" in result
        assert "[1, 2, 3]" in result

    async def test_isinstance_still_works(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(isinstance(1, int))")
        assert "True" in result

    async def test_any_all_min_max(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(any([0, 1]))\nprint(all([1, 1]))\nprint(max([3, 1, 2]))")
        assert "True" in result
        assert "3" in result

    async def test_getattr_on_safe_name_works(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('d = {"a": 1}\nprint(getattr(d, "keys")())')
        assert "dict_keys" in result

    async def test_asyncio_gather_still_works(self) -> None:
        agent = Agent(name="t", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
results = await asyncio.gather(
    default_api.greet(name="A"),
    default_api.add(a=1, b=2),
)
for r in results:
    print(r)
"""
        result = await executor.run(code)
        assert "Hello, A!" in result
        assert "3" in result

    async def test_json_loads_still_works(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('d = json.loads(\'{"a": 1, "b": 2}\')\nprint(d["a"] + d["b"])')
        assert "3" in result

    async def test_math_re_still_work(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('print(math.sqrt(16))\nprint(re.sub(r"\\d", "X", "a1b2"))')
        assert "4.0" in result
        assert "aXbX" in result


class TestPTCSandboxMessages:
    """Sandbox errors must clearly direct the agent to default_api."""

    async def test_import_error_mentions_default_api(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import os")
        assert "default_api" in result

    async def test_builtin_error_mentions_default_api(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("open('x')")
        assert "default_api" in result

    async def test_ast_scan_error_mentions_default_api(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("x = object.__bases__")
        assert "default_api" in result
