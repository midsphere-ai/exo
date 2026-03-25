# Testing Guide

Exo uses pytest with pytest-asyncio for all testing. This guide covers conventions, patterns, and common pitfalls.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific package's tests
uv run pytest packages/exo-core/tests/
uv run pytest packages/exo-models/tests/

# Run a specific test file
uv run pytest packages/exo-core/tests/test_agent.py

# Run a specific test function
uv run pytest packages/exo-core/tests/test_agent.py::test_agent_calls_tool_and_returns_result

# Run with verbose output
uv run pytest -v

# Run with print statements visible
uv run pytest -s
```

## Async Test Support

The root `pyproject.toml` configures `asyncio_mode = "auto"`, which means:

- **All `async def test_*` functions run automatically** as async tests
- **No need for `@pytest.mark.asyncio`** decorator
- You can mix sync and async tests freely in the same file

```python
# Both of these just work:

def test_sync_creation():
    agent = Agent(name="test", model="openai:gpt-4o")
    assert agent.name == "test"

async def test_async_execution(mock_provider):
    agent = Agent(name="test", tools=[add_tool])
    result = await run(agent, "add 2 and 3", provider=mock_provider)
    assert "5" in result.output
```

## Test File Naming

Test file names must be **unique across all packages**. Pytest collects tests from multiple `tests/` directories using `--import-mode=importlib`, and duplicate names cause import conflicts.

```
# GOOD — unique names
packages/exo-core/tests/test_types.py
packages/exo-models/tests/test_model_types.py     # prefixed!
packages/exo-core/tests/test_config.py
packages/exo-models/tests/test_model_provider.py   # prefixed!

# BAD — collisions
packages/exo-core/tests/test_types.py
packages/exo-models/tests/test_types.py            # COLLISION!
```

Convention: prefix test files with the package's sub-namespace when there is potential for collision.

## Test Function Naming

Use the pattern `test_<what>_<scenario>`:

```python
# Clear, descriptive names
async def test_agent_calls_tool_and_returns_result(mock_provider): ...
async def test_agent_respects_max_steps(mock_provider): ...
def test_parse_model_string_with_provider(): ...
def test_parse_model_string_without_provider_defaults_to_openai(): ...
def test_registry_raises_on_duplicate(): ...
async def test_stream_yields_text_events(mock_provider): ...
```

## Mock Providers

**Never make real API calls in tests.** Always use mock providers for LLM calls.

### Basic Mock Provider

```python
import pytest
from exo.types import AssistantMessage, ToolCall, Usage

class MockProvider:
    """A provider that returns canned responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self._call_idx = 0

    async def complete(self, messages, *, tools=None, temperature=None, max_tokens=None):
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        # Default: return a simple text response
        return _make_response("I don't know.")

    async def stream(self, messages, *, tools=None, temperature=None, max_tokens=None):
        resp = await self.complete(messages, tools=tools)
        yield _make_chunk(resp.content)


def _make_response(text, tool_calls=None):
    """Helper to create a ModelResponse-like object."""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.content = text
    resp.tool_calls = tool_calls or []
    resp.usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    return resp
```

### Provider as Fixture

```python
@pytest.fixture
def mock_provider():
    """A simple mock provider returning 'Hello!'."""
    return MockProvider(responses=[_make_response("Hello!")])

@pytest.fixture
def tool_calling_provider():
    """A provider that calls a tool then responds."""
    return MockProvider(responses=[
        _make_response("", tool_calls=[
            ToolCall(id="tc-1", name="add", arguments='{"a": 2, "b": 3}'),
        ]),
        _make_response("The result is 5."),
    ])
```

### Using in Tests

```python
from exo import Agent, tool, run

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async def test_agent_calls_tool_and_returns_result(tool_calling_provider):
    agent = Agent(name="calc", model="openai:gpt-4o", tools=[add])
    result = await run(agent, "What is 2 + 3?", provider=tool_calling_provider)
    assert "5" in result.output
    assert result.steps >= 1
```

## Testing Patterns

### Test the Public API, Not Internals

```python
# GOOD — tests the public API
async def test_run_returns_result(mock_provider):
    agent = Agent(name="test")
    result = await run(agent, "Hello", provider=mock_provider)
    assert isinstance(result, RunResult)
    assert result.output == "Hello!"

# BAD — tests internal implementation details
async def test_call_runner_creates_run_state():
    state = RunState(agent_name="test")
    # ... testing internal state machine transitions
```

### Test Edge Cases

```python
def test_agent_rejects_duplicate_tools():
    """Duplicate tool names should raise AgentError."""
    t1 = FunctionTool(lambda: "a", name="dupe")
    t2 = FunctionTool(lambda: "b", name="dupe")
    with pytest.raises(AgentError, match="Duplicate tool name"):
        Agent(name="test", tools=[t1, t2])

def test_empty_swarm_raises():
    """Swarm with no agents should raise SwarmError."""
    with pytest.raises(SwarmError, match="at least one agent"):
        Swarm(agents=[])
```

### Test Error Handling

```python
async def test_tool_error_returned_as_tool_result(error_provider):
    """Tool errors should be captured as ToolResult(error=...), not propagated."""
    @tool
    def failing_tool(x: str) -> str:
        """Always fails."""
        raise ValueError("deliberate failure")

    agent = Agent(name="test", tools=[failing_tool])
    # The agent should not crash — it gets the error as a tool result
    result = await run(agent, "call the tool", provider=error_provider)
    assert result is not None
```

## Cross-Package Import Workaround

Pyright cannot resolve cross-namespace-package imports from editable installs. In test files that import across packages, use the `pyright: ignore` directive:

```python
# In packages/exo-core/tests/test_integration.py
from exo.models.types import ModelError  # pyright: ignore[reportMissingImports]
```

**Only use this in test files** that genuinely need cross-package imports. Source files should not need this directive because their dependencies are declared in `pyproject.toml`.

## Fixture Patterns

### Shared Fixtures

Put shared fixtures in `conftest.py` at the `tests/` directory level:

```python
# packages/exo-core/tests/conftest.py

import pytest
from exo import Agent

@pytest.fixture
def basic_agent():
    return Agent(name="test", model="openai:gpt-4o", instructions="Be helpful.")

@pytest.fixture
def mock_provider():
    return MockProvider()
```

### Parameterized Tests

```python
@pytest.mark.parametrize("model_string,expected_provider,expected_model", [
    ("openai:gpt-4o", "openai", "gpt-4o"),
    ("anthropic:claude-sonnet-4-20250514", "anthropic", "claude-sonnet-4-20250514"),
    ("gpt-4o", "openai", "gpt-4o"),  # defaults to openai
])
def test_parse_model_string(model_string, expected_provider, expected_model):
    provider, model = parse_model_string(model_string)
    assert provider == expected_provider
    assert model == expected_model
```

## Coverage Goals

- **Every module has a corresponding test file** (`test_<module>.py`)
- **~5-15 tests per file** covering the public API surface
- **No tests for private `_internal/` methods** unless they have complex logic worth testing directly
- Focus on behavior, not implementation details
