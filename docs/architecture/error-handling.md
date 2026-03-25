# Error Handling

Exo uses a structured exception hierarchy rooted at `ExoError`. This document describes the error types, their semantics, and how errors propagate through the system.

## Exception Hierarchy

```
Exception
  +-- ExoError                  Base for all Exo errors (exo.types)
  |     +-- AgentError              Agent-level failures (exo.agent)
  |     +-- ToolError               Tool execution failures (exo.tool)
  |     +-- CallRunnerError         Call runner failures (exo._internal.call_runner)
  |     +-- SwarmError              Swarm orchestration failures (exo.swarm)
  |     +-- RegistryError           Registry lookup/duplicate failures (exo.registry)
  |     +-- OutputParseError        LLM output parsing failures (exo._internal.output_parser)
  |     +-- ModelError              LLM provider failures (exo.models.types)
  +-- GraphError                    Graph/flow DSL failures (exo._internal.graph)
```

### Error Types in Detail

**`ExoError`** (`exo.types`)
The root exception for all Exo errors. Catching this catches everything Exo can raise (except `GraphError` which is intentionally separate as a low-level utility).

```python
class ExoError(Exception):
    """Base exception for all Exo errors."""
```

**`AgentError`** (`exo.agent`)
Raised for agent-level problems: duplicate tools, missing provider, retry exhaustion, context length overflow.

```python
# Missing provider
AgentError("Agent 'researcher' requires a provider for run()")

# Retry exhaustion
AgentError("Agent 'researcher' failed after 3 retries: ConnectionError(...)")

# Context length
AgentError("Context length exceeded on agent 'researcher': ...")
```

**`ToolError`** (`exo.tool`)
Raised when a tool's `execute()` method fails. The tool name is always included.

```python
class ToolError(ExoError):
    """Raised when a tool execution fails."""
```

`FunctionTool` catches all exceptions from the wrapped function and re-raises as `ToolError`:

```python
async def execute(self, **kwargs):
    try:
        if self._is_async:
            return await self._fn(**kwargs)
        return await asyncio.to_thread(self._fn, **kwargs)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Tool '{self.name}' failed: {exc}") from exc
```

**`CallRunnerError`** (`exo._internal.call_runner`)
Raised by the call runner for loop detection or agent execution failures.

```python
# Loop detection
CallRunnerError("Endless loop detected: same tool calls repeated 3 times (threshold=3)")

# Agent failure (wraps original exception)
CallRunnerError("Call runner failed for agent 'researcher': ...")
```

**`SwarmError`** (`exo.swarm`)
Raised for swarm configuration and execution problems: empty agent list, duplicate names, invalid flow DSL, cycle detection, max handoffs exceeded.

**`OutputParseError`** (`exo._internal.output_parser`)
Raised when LLM output cannot be parsed: invalid JSON in tool arguments, structured output validation failure.

```python
# Invalid JSON
OutputParseError("Invalid JSON in arguments for tool 'search': Expecting value: line 1 column 1")

# Structured output validation
OutputParseError("Structured output failed ResearchReport validation: ...")
```

**`ModelError`** (`exo.models.types`)
Raised by LLM providers for API errors, authentication failures, rate limits, etc.

**`RegistryError`** (`exo.registry`)
Raised when a registry operation fails: duplicate registration, missing lookup.

```python
# Duplicate
RegistryError("'search' is already registered in tool_registry")

# Missing
RegistryError("'unknown' not found in model_registry")
```

## Error Propagation Patterns

### Tool Errors Are Captured, Not Propagated

When a tool fails during agent execution, the error is **caught and returned as a `ToolResult` with an error field**. The error is NOT propagated up the call stack. This allows the LLM to see the error and decide what to do (retry, use a different tool, or report the failure to the user).

```python
# In agent.py, _execute_tools()
try:
    output = await tool.execute(**action.arguments)
    result = ToolResult(
        tool_call_id=action.tool_call_id,
        tool_name=action.tool_name,
        content=output if isinstance(output, str) else str(output),
    )
except (ToolError, Exception) as exc:
    result = ToolResult(
        tool_call_id=action.tool_call_id,
        tool_name=action.tool_name,
        error=str(exc),  # Error goes to LLM, not up the stack
    )
```

This is a deliberate design choice. The LLM is often better at recovering from tool failures than the framework would be.

### LLM Errors Use Retry with Exponential Backoff

Transient LLM errors (network timeouts, rate limits) are retried up to `max_retries` times with exponential backoff:

```python
for attempt in range(max_retries):
    try:
        response = await provider.complete(...)
        return parse_response(...)
    except Exception as exc:
        if _is_context_length_error(exc):
            raise AgentError(...) from exc  # No retry for context length
        last_error = exc
        if attempt < max_retries - 1:
            delay = 2 ** attempt  # 1s, 2s, 4s, ...
            await asyncio.sleep(delay)

raise AgentError(f"... failed after {max_retries} retries: {last_error}") from last_error
```

Context-length errors are detected and raised immediately (no retry) because adding more context will not fix the problem.

### Exception Chaining with `from`

All re-raised exceptions use `from e` to preserve the original traceback:

```python
try:
    result = await tool.execute(**args)
except Exception as e:
    raise ToolExecutionError(tool.name, e) from e  # preserves original traceback
```

This means `__cause__` is always set, and `traceback.print_exception()` shows the full chain.

### Call Runner Wraps All Exceptions

The call runner is the outermost try/except around agent execution. Any unhandled exception from `agent.run()` is caught and wrapped in `CallRunnerError`:

```python
try:
    output = await agent.run(...)
    state.succeed()
    return RunResult(...)
except CallRunnerError:
    node.fail("Endless loop detected")
    state.fail("Endless loop detected")
    raise  # re-raise as-is
except Exception as exc:
    node.fail(str(exc))
    state.fail(str(exc))
    raise CallRunnerError(f"Call runner failed for agent '{agent.name}': {exc}") from exc
```

## Best Practices for Error Handling

### In Tools

```python
from exo import tool
from exo.tool import ToolError

@tool
async def query_database(sql: str) -> str:
    """Execute a SQL query."""
    try:
        result = await db.execute(sql)
        return str(result)
    except DatabaseError as e:
        # Raise ToolError with context -- this becomes ToolResult(error=...)
        raise ToolError(f"Database query failed: {e}") from e
```

### In Hooks

```python
from exo.hooks import HookPoint

async def validate_tool_call(agent, tool_name, arguments):
    """Hook that validates tool calls before execution."""
    if tool_name == "delete_all" and not arguments.get("confirm"):
        # Raising in a hook aborts the operation
        raise ValueError("delete_all requires confirm=True")

agent = Agent(
    name="admin",
    hooks=[(HookPoint.PRE_TOOL_CALL, validate_tool_call)],
    ...
)
```

### In User Code

```python
from exo import run
from exo.types import ExoError
from exo.agent import AgentError

try:
    result = await run(agent, "Delete everything")
except AgentError as e:
    print(f"Agent failed: {e}")
    # e.__cause__ has the original exception
except ExoError as e:
    print(f"Framework error: {e}")
```

## Summary Table

| Error Type | Raised By | Propagated? | Recovery |
|-----------|-----------|-------------|----------|
| `ToolError` | `tool.execute()` | No -- captured as `ToolResult(error=...)` | LLM sees error and decides |
| `AgentError` | `agent.run()`, `_call_llm()` | Yes -- up to call_runner | Caller handles or run fails |
| `CallRunnerError` | `call_runner()` | Yes -- up to `run()` | Caller handles |
| `SwarmError` | `swarm.run()` | Yes -- up to `run()` | Caller handles |
| `OutputParseError` | `output_parser` | Yes -- causes LLM retry | Retry with same messages |
| `ModelError` | Provider SDK | Yes -- causes LLM retry | Exponential backoff retry |
| `RegistryError` | `Registry.get()` | Yes -- immediate | Fix registration |
