---
name: exo:testing
description: "Use when writing tests for Exo agents, tools, swarms, or streaming — MockProvider patterns, async test configuration, asyncio_mode auto, FakeStreamChunk, test fixtures, tool testing, swarm testing, streaming event testing, ToolContext testing, RalphRunner streaming tests. Triggers on: test exo, mock provider, MockProvider, test agent, test swarm, test streaming, pytest exo, test tool, async test, test ToolContext, test ralph."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Testing — Test Patterns & MockProvider

## When To Use This Skill

Use this skill when the developer needs to:
- Write tests for Exo agents, tools, swarms, or streaming
- Create mock providers that return predetermined responses
- Test tool execution without real API calls
- Test streaming event sequences
- Understand test configuration (asyncio_mode, importlib mode)
- Test ToolContext injection and event emission
- Test RalphRunner streaming event sequences
- Write integration tests with real providers

## Decision Guide

1. **Testing a simple agent response?** → Use `MockProvider` with fixed content
2. **Testing a tool-calling flow?** → Use sequential `MockProvider` (returns tool call first, then text)
3. **Testing streaming?** → Use `_make_stream_provider` with `FakeStreamChunk` lists
4. **Testing a swarm?** → Use `_make_provider` with sequential `AgentOutput` responses
5. **Testing tool execution directly?** → Call `tool.execute(**kwargs)` directly
6. **Need to inspect what was sent to the LLM?** → Use `RecordingProvider` that captures messages
7. **Testing ToolContext?** → Create `asyncio.Queue`, construct `ToolContext(name, queue)`, call `emit()`, check queue
8. **Testing RalphRunner.stream()?** → Mock `stream_execute_fn` as async generator, verify event sequence

## Reference

### Test Configuration

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["packages/*/tests", "tests/integration"]
asyncio_mode = "auto"
addopts = "--import-mode=importlib"
```

**Key settings:**
- `asyncio_mode = "auto"` — All `async def test_*` functions run automatically without `@pytest.mark.asyncio`
- `--import-mode=importlib` — Allows loading tests from multiple `packages/*/tests/` dirs without naming collisions
- **Test file names must be unique** across all packages (importlib mode requirement)

### MockProvider Patterns

#### Pattern 1: Simple Fixed Response

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class _MockResponse:
    content: str = ""
    tool_calls: list = None
    usage: Any = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.usage is None:
            from exo.types import Usage
            self.usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)

class MockProvider:
    """Returns a fixed response for every call."""

    def __init__(self, content: str = "") -> None:
        self._content = content

    async def complete(self, messages: Any, **kwargs: Any) -> _MockResponse:
        return _MockResponse(content=self._content)
```

**Usage:**
```python
async def test_simple_response():
    agent = Agent(name="bot", memory=None, context=None)
    provider = MockProvider("Hello, world!")
    result = await run(agent, "Hi", provider=provider)
    assert result.output == "Hello, world!"
```

#### Pattern 2: Sequential Responses (Tool Calls)

```python
from unittest.mock import AsyncMock
from exo.types import AgentOutput, ToolCall, Usage

def _make_provider(responses: list[AgentOutput]) -> Any:
    """Returns responses in sequence — one per LLM call."""
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage or Usage(input_tokens=10, output_tokens=5, total_tokens=15)

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock
```

**Usage (tool call flow):**
```python
async def test_agent_uses_tool():
    @tool
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    agent = Agent(name="calc", tools=[add], memory=None, context=None)
    responses = [
        # Step 1: LLM requests tool call
        AgentOutput(
            text="",
            tool_calls=[ToolCall(id="tc1", name="add", arguments='{"a":3,"b":7}')],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        # Step 2: LLM returns final text after seeing tool result
        AgentOutput(
            text="The answer is 10.",
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        ),
    ]
    provider = _make_provider(responses)

    result = await run(agent, "What is 3+7?", provider=provider)
    assert result.output == "The answer is 10."
```

#### Pattern 3: Recording Provider (Inspection)

```python
class RecordingProvider:
    """Captures all calls for assertion."""

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
```

**Usage:**
```python
async def test_instructions_sent():
    agent = Agent(name="bot", instructions="Be helpful.", memory=None, context=None)
    provider = RecordingProvider([AgentOutput(text="OK")])

    await run(agent, "Hi", provider=provider)

    # Inspect what was sent to the LLM
    first_call = provider.calls[0]
    messages = first_call["messages"]
    assert any("Be helpful" in str(m) for m in messages)
```

#### Pattern 4: Stream Provider (For run.stream())

```python
class FakeStreamChunk:
    """Mirrors StreamChunk fields for testing."""

    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list = None,
        finish_reason: str | None = None,
        usage: Any = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()

class FakeToolCallDelta:
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

def _make_stream_provider(stream_rounds: list[list[FakeStreamChunk]]) -> Any:
    """Returns stream chunks in sequence — one round per LLM call."""
    call_count = 0

    async def stream(messages: Any, **kwargs: Any):
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock
```

**Usage:**
```python
from exo.types import TextEvent

async def test_stream_text_events():
    agent = Agent(name="bot", memory=None, context=None)
    chunks = [
        FakeStreamChunk(delta="Hello"),
        FakeStreamChunk(delta=" world"),
        FakeStreamChunk(delta="!", finish_reason="stop"),
    ]
    provider = _make_stream_provider([chunks])

    events = []
    async for ev in run.stream(agent, "Hi", provider=provider):
        events.append(ev)

    assert len(events) == 3
    assert all(isinstance(e, TextEvent) for e in events)
    assert events[0].text == "Hello"
```

### Testing Tools Directly

```python
@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

async def test_tool_execution():
    result = await greet.execute(name="Alice")
    assert result == "Hello, Alice!"

def test_tool_schema():
    schema = greet.to_schema()
    assert schema["function"]["name"] == "greet"
    assert "name" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["name"]
```

### Testing Tool Errors

```python
@tool
def risky(x: int) -> str:
    """Risky operation."""
    raise ValueError("Something went wrong!")

async def test_tool_error_captured():
    agent = Agent(name="bot", tools=[risky], memory=None, context=None)
    responses = [
        AgentOutput(
            text="",
            tool_calls=[ToolCall(id="tc1", name="risky", arguments='{"x":1}')],
        ),
        AgentOutput(text="The tool failed, trying another approach."),
    ]
    provider = _make_provider(responses)

    result = await run(agent, "Do the risky thing", provider=provider)
    # Tool error is fed back to LLM, which responds gracefully
    assert result.output == "The tool failed, trying another approach."
```

### Testing Swarms

```python
async def test_workflow_pipeline():
    a = Agent(name="a", memory=None, context=None)
    b = Agent(name="b", memory=None, context=None)
    swarm = Swarm(agents=[a, b], flow="a >> b")

    provider = _make_provider([
        AgentOutput(text="from_a"),
        AgentOutput(text="from_b"),
    ])

    result = await swarm.run("Hi", provider=provider)
    assert result.output == "from_b"

async def test_swarm_streaming():
    a = Agent(name="a", memory=None, context=None)
    swarm = Swarm(agents=[a])
    provider = _make_stream_provider([[
        FakeStreamChunk(delta="Hello"),
        FakeStreamChunk(delta="!", finish_reason="stop"),
    ]])

    events = [ev async for ev in swarm.stream("Hi", provider=provider)]
    text = "".join(e.text for e in events if isinstance(e, TextEvent))
    assert text == "Hello!"
```

### Testing Parallel Tool Calls

```python
async def test_parallel_tools():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    agent = Agent(name="bot", tools=[greet], memory=None, context=None)
    responses = [
        AgentOutput(
            text="",
            tool_calls=[
                ToolCall(id="tc1", name="greet", arguments='{"name":"Alice"}'),
                ToolCall(id="tc2", name="greet", arguments='{"name":"Bob"}'),
            ],
        ),
        AgentOutput(text="Greeted both!"),
    ]
    provider = _make_provider(responses)

    result = await run(agent, "Greet Alice and Bob", provider=provider)
    assert result.output == "Greeted both!"
```

### Testing Structured Output

```python
from pydantic import BaseModel

class Result(BaseModel):
    answer: str
    confidence: float

async def test_structured_output():
    agent = Agent(
        name="bot",
        output_type=Result,
        memory=None,
        context=None,
    )
    provider = MockProvider('{"answer": "42", "confidence": 0.95}')

    result = await run(agent, "What is the answer?", provider=provider)
    assert isinstance(result.output, Result)
    assert result.output.answer == "42"
```

### Testing Ephemeral Messages

```python
async def test_ephemeral_present_then_removed():
    """Ephemeral message visible in first call, gone in second."""
    calls: list[list[Any]] = []

    @tool
    def ping(msg: str) -> str:
        """Simple tool."""
        return "pong"

    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        calls.append(list(messages))
        call_count += 1
        if call_count == 1:
            class R1:
                content = ""
                tool_calls = [ToolCall(id="tc1", name="ping", arguments='{"msg":"hi"}')]
                usage = Usage()
            return R1()
        class R2:
            content = "Done"
            tool_calls = []
            usage = Usage()
        return R2()

    provider = AsyncMock()
    provider.complete = complete

    agent = Agent(name="bot", tools=[ping], memory=None, context=None)
    agent.inject_ephemeral("One-shot context")

    result = await run(agent, "Go", provider=provider)

    # First call: ephemeral present
    assert any(
        isinstance(m, UserMessage) and m.content == "One-shot context"
        for m in calls[0]
    )
    # Second call: ephemeral gone
    assert not any(
        isinstance(m, UserMessage) and m.content == "One-shot context"
        for m in calls[1]
    )
```

**Key testing points:**
- Use a RecordingProvider or capture `messages` in each `complete()` call
- First call should contain the ephemeral message; second call should not
- `inject_ephemeral()` also accepts `Message` objects (e.g., `SystemMessage`) — test with both
- No `MessageInjectedEvent` is emitted for ephemeral messages (unlike `inject_message()`)

### Testing Multi-Turn Conversations

```python
async def test_multi_turn():
    agent = Agent(name="bot", memory=None, context=None)
    p1 = _make_provider([AgentOutput(text="Count: 1")])
    r1 = await run(agent, "Start counting", provider=p1)

    p2 = _make_provider([AgentOutput(text="Count: 2")])
    r2 = await run(agent, "Next", messages=r1.messages, provider=p2)
    assert r2.output == "Count: 2"
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("model_string,expected_provider,expected_model", [
    ("openai:gpt-4o", "openai", "gpt-4o"),
    ("anthropic:claude-sonnet-4-20250514", "anthropic", "claude-sonnet-4-20250514"),
    ("gpt-4o", "openai", "gpt-4o"),  # Default provider
])
def test_model_parsing(model_string, expected_provider, expected_model):
    agent = Agent(name="bot", model=model_string, memory=None, context=None)
    assert agent.provider_name == expected_provider
    assert agent.model_name == expected_model
```

### Testing ToolContext

```python
import asyncio
from exo.tool_context import ToolContext
from exo.tool import _generate_schema, FunctionTool, tool
from exo.types import TextEvent

def test_tool_context_excluded_from_schema():
    """ToolContext params are hidden from LLM schema."""
    def fn(query: str, ctx: ToolContext) -> str:
        return query

    schema = _generate_schema(fn)
    assert "ctx" not in schema["properties"]
    assert schema["required"] == ["query"]

def test_function_tool_detects_context_param():
    """FunctionTool caches the ToolContext parameter name."""
    @tool
    async def research(query: str, ctx: ToolContext) -> str:
        """Research."""
        return query

    assert research._tool_context_param == "ctx"

def test_tool_context_emit():
    """emit() pushes events to the queue."""
    queue: asyncio.Queue = asyncio.Queue()
    ctx = ToolContext(agent_name="parent", queue=queue)
    event = TextEvent(text="hello", agent_name="inner")
    ctx.emit(event)
    assert queue.get_nowait() is event
```

### Testing RalphRunner Streaming

```python
from exo.eval.ralph import RalphRunner, RalphConfig, StopConditionConfig
from exo.types import RalphIterationEvent, RalphStopEvent, TextEvent

async def test_ralph_stream_events():
    """Verify event sequence from RalphRunner.stream()."""
    async def stream_execute(input: str):
        yield TextEvent(text="result", agent_name="inner")

    async def execute(input: str) -> str:
        return "result"

    cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=1))
    runner = RalphRunner(
        execute_fn=execute,
        scorers=[],
        stream_execute_fn=stream_execute,
        config=cfg,
    )

    events = [ev async for ev in runner.stream("input", name="test")]

    assert isinstance(events[0], RalphIterationEvent)
    assert events[0].status == "started"
    assert isinstance(events[1], TextEvent)
    assert isinstance(events[2], RalphIterationEvent)
    assert events[2].status == "completed"
    assert isinstance(events[3], RalphStopEvent)

async def test_ralph_stream_requires_fn():
    """stream() raises ValueError without stream_execute_fn."""
    import pytest
    runner = RalphRunner(execute_fn=lambda x: x, scorers=[])
    with pytest.raises(ValueError, match="stream_execute_fn required"):
        async for _ in runner.stream("input"):
            pass
```

### Testing RalphNode in Swarm

```python
from exo._internal.nested import RalphNode
from exo.types import RalphIterationEvent, RalphStopEvent, RunResult

async def test_ralph_node_stream():
    """RalphNode delegates to runner.stream()."""
    expected = [
        RalphIterationEvent(iteration=1, status="started", agent_name="test"),
        RalphStopEvent(stop_type="max_iterations", reason="done", iterations=1, agent_name="test"),
    ]

    class FakeRunner:
        async def stream(self, input, *, name="ralph"):
            for ev in expected:
                yield ev

    node = RalphNode(runner=FakeRunner(), name="test")
    events = [ev async for ev in node.stream("query")]
    assert events == expected
    assert node.is_group is True  # Swarm duck-typing
```

## Patterns

### Disable Auto-Features in Tests

Always set `memory=None` and `context=None` in tests to avoid side effects:

```python
agent = Agent(
    name="test-bot",
    memory=None,     # No auto-memory (avoids DB/embedding calls)
    context=None,    # No auto-context (avoids windowing side effects)
    tools=[my_tool],
)
```

### Test Fixture for Common Setup

```python
import pytest

@pytest.fixture
def agent():
    return Agent(name="test-bot", memory=None, context=None)

@pytest.fixture
def simple_provider():
    return MockProvider("OK")

async def test_basic(agent, simple_provider):
    result = await run(agent, "Hi", provider=simple_provider)
    assert result.output == "OK"
```

### Integration Test with Real Provider

```python
import os
import pytest

@pytest.fixture
def openai_provider():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    from exo.models.provider import get_provider
    return get_provider("openai:gpt-4o-mini", api_key=api_key)

@pytest.mark.integration
async def test_real_agent(openai_provider):
    agent = Agent(name="bot", model="openai:gpt-4o-mini", memory=None, context=None)
    result = await run(agent, "Say hello in one word.", provider=openai_provider)
    assert len(result.output) > 0
```

## Gotchas

- **Always set `memory=None, context=None` in tests** — prevents auto-creation of ChromaDB/SQLite stores and context windowing side effects
- **`asyncio_mode = "auto"`** — no `@pytest.mark.asyncio` needed on `async def test_*` functions
- **Test file names must be globally unique** — `test_agent.py` can only exist in one package
- **Never make real API calls in unit tests** — always use MockProvider
- **MockProvider is defined per-test-file** — there's no shared `conftest.py` provider. Each test file defines its own suited to its needs.
- **`AgentOutput.tool_calls` arguments must be JSON strings** — `'{"a":3,"b":7}'`, not dicts
- **Mock provider `complete()` must return an object with `.content`, `.tool_calls`, and `.usage` attributes** — not a dict
- **Stream provider needs both `.stream()` and `.complete`** — some code paths may call either
- **Use `_make_provider` for multi-step flows** — it handles sequential response cycling with `call_count`
- **Swarm tests need one provider** — all agents in a swarm share the same provider, so responses must cover all agents' calls in sequence
