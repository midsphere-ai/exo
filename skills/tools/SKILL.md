---
name: exo:tools
description: "Use when creating tools for Exo agents — @tool decorator, FunctionTool, Tool ABC subclass, injected_tool_args, large_output, structured output (output_type), or MCP server integration. Triggers on: exo tool, @tool, FunctionTool, Tool ABC, injected_tool_args, large_output, output_type, add_mcp_server, structured output."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Tools — Advanced Tool Creation

## When To Use This Skill

Use this skill when the developer needs to:
- Create tools for an Exo agent using `@tool`, `FunctionTool`, or `Tool` ABC
- Configure schema-only injected arguments (`injected_tool_args`)
- Handle large tool output with workspace offloading (`large_output`)
- Define structured output schemas (`output_type`)
- Connect MCP servers to agents (`add_mcp_server`)
- Understand how tool schemas are generated from function signatures

## Decision Guide

Ask these questions to pick the right pattern:

1. **Simple function → tool?** Use `@tool` decorator (handles sync/async, schema generation)
2. **Need to customize name/description programmatically?** Use `FunctionTool(fn, name=..., description=...)`
3. **Need stateful tool with custom execute logic?** Subclass `Tool` ABC
4. **Tool returns huge results (>10KB)?** Set `large_output=True` — result offloaded to workspace, LLM gets a pointer
5. **Need the LLM to see a parameter it doesn't actually fill?** Use `injected_tool_args` on Agent
6. **Need validated structured output from the agent?** Set `output_type=MyPydanticModel` on Agent
7. **Need tools from an MCP server?** Use `agent.add_mcp_server(config)`

## Reference

### The @tool Decorator

Three forms — bare, empty parens, and with kwargs:

```python
from exo import tool

# Form 1: Bare decorator
@tool
async def search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query string.
    """
    return f"Results for: {query}"

# Form 2: Empty parens (equivalent to Form 1)
@tool()
def get_time() -> str:
    """Return the current UTC time."""
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat()

# Form 3: With overrides
@tool(name="web_search", description="Custom description", large_output=True)
async def search_web(query: str, max_results: int = 10) -> str:
    """Search the web.

    Args:
        query: Search query.
        max_results: Maximum number of results to return.
    """
    return "x" * 50000  # Large result → offloaded automatically
```

**Decorator parameters:**
- `name: str | None` — Override tool name (defaults to `fn.__name__`)
- `description: str | None` — Override description (defaults to first line of docstring)
- `large_output: bool` — When `True`, results exceeding `EXO_LARGE_OUTPUT_THRESHOLD` (default 10KB) are offloaded to workspace

**Sync vs async:** Both work. Sync functions are auto-wrapped via `asyncio.to_thread()`.

### FunctionTool (Programmatic)

```python
from exo import FunctionTool

def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: The math expression to evaluate.
    """
    return str(eval(expression))  # noqa: S307

calc_tool = FunctionTool(
    calculate,
    name="calculator",
    description="Evaluate mathematical expressions",
    large_output=False,
)

# Use it:
agent = Agent(name="math-bot", tools=[calc_tool])
```

### Tool ABC (Custom Subclass)

For tools that need custom state, initialization, or non-function-based logic:

```python
from exo import Tool
from typing import Any

class DatabaseQueryTool(Tool):
    name = "query_db"
    description = "Execute a read-only SQL query against the database"
    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQL query to execute (SELECT only)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return",
            },
        },
        "required": ["sql"],
    }

    def __init__(self, connection_string: str) -> None:
        self._conn_str = connection_string

    async def execute(self, **kwargs: Any) -> str:
        sql = kwargs["sql"]
        limit = kwargs.get("limit", 100)
        # ... execute query ...
        return f"Query returned {limit} rows"
```

**Return types for `execute()`:** `str | dict[str, Any] | list[ContentBlock]`

### Schema Generation Rules

The `@tool` decorator and `FunctionTool` auto-generate JSON Schema from the function signature:

**Type mapping:**
| Python Type | JSON Schema Type |
|-------------|-----------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list[T]` | `{"type": "array", "items": ...}` |
| `dict` | `{"type": "object"}` |
| `Optional[T]` / `T | None` | Unwraps to T's schema |
| `Any` / missing annotation | `"string"` (fallback) |

**Docstring parsing:** Google-style `Args:` section becomes parameter descriptions:
```python
@tool
def example(name: str, count: int = 5) -> str:
    """Do something.

    Args:
        name: The target name.
        count: How many times to repeat.
    """
```
Generates:
```json
{
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The target name."},
        "count": {"type": "integer", "description": "How many times to repeat."}
    },
    "required": ["name"]
}
```

**Rules:**
- Parameters with defaults are optional (not in `required`)
- `self`, `cls`, `*args`, `**kwargs` are skipped
- Multi-line `Args:` descriptions are joined with spaces

### injected_tool_args (Schema-Only Arguments)

Arguments that appear in the LLM's tool schema but are filled by the runtime, not the LLM:

```python
agent = Agent(
    name="assistant",
    tools=[my_tool],
    injected_tool_args={
        "user_id": "The ID of the current user",
        "session_token": "Authentication token for the current session",
    },
)
```

The LLM sees `user_id` and `session_token` in tool schemas but the runtime injects them. Useful for passing context that the LLM should be aware of but shouldn't generate.

**Validation:** Keys must be non-empty strings, values must be strings (descriptions).

### large_output (Workspace Offloading)

When a tool's result exceeds `EXO_LARGE_OUTPUT_THRESHOLD` bytes (default 10,240 = 10KB):

1. Result stored in agent's `Workspace` as an artifact
2. LLM receives a pointer: `[Result stored as artifact 'tool_result_search_a3f1b2c4'. Call retrieve_artifact('tool_result_search_a3f1b2c4') to access.]`
3. LLM can call `retrieve_artifact(id)` to fetch the full content

```python
@tool(large_output=True)
async def fetch_page(url: str) -> str:
    """Fetch a web page's full HTML content.

    Args:
        url: The URL to fetch.
    """
    # Returns potentially large content
    return await httpx.get(url).text
```

**Auto-registration:** When any `large_output=True` tool is registered, `retrieve_artifact` is auto-added to the agent's tool set.

**Threshold override:** Set `EXO_LARGE_OUTPUT_THRESHOLD` env var (in bytes).

### Structured Output (output_type)

Force the agent's final response to conform to a Pydantic model:

```python
from pydantic import BaseModel
from exo import Agent, run

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    key_topics: list[str]
    summary: str

agent = Agent(
    name="analyzer",
    model="openai:gpt-4o",
    instructions="Analyze the given text and return structured results.",
    output_type=AnalysisResult,
)

result = await run(agent, "Analyze: The product launch exceeded expectations...")
# result.output is a validated AnalysisResult instance
print(result.output.sentiment)     # "positive"
print(result.output.confidence)    # 0.95
print(result.output.key_topics)    # ["product launch", "expectations"]
```

### MCP Server Integration

Connect external MCP servers and use their tools:

```python
from exo.mcp import MCPServerConfig

config = MCPServerConfig(
    name="memory-server",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-memory"],
)

# Add at runtime (async)
await agent.add_mcp_server(config)
# All MCP tools are now registered on the agent
```

**What happens internally:**
1. Creates `MCPServerConnection(config)` and connects
2. Lists all tools from the server
3. Wraps each as an Exo `Tool` via `load_tools_from_connection()`
4. Registers each tool on the agent (duplicate names raise `AgentError`)

## Patterns

### Tool That Returns Multimodal Content

```python
from exo import tool
from exo.types import TextBlock, ImageURLBlock, ContentBlock

@tool
async def analyze_image(url: str) -> list[ContentBlock]:
    """Analyze an image and return findings with the image.

    Args:
        url: URL of the image to analyze.
    """
    return [
        TextBlock(text="This image shows a sunset over mountains."),
        ImageURLBlock(url=url, detail="high"),
    ]
```

### Tool Error Handling

```python
from exo.tool import ToolError

@tool
async def safe_divide(a: float, b: float) -> str:
    """Divide two numbers safely.

    Args:
        a: Numerator.
        b: Denominator.
    """
    if b == 0:
        raise ToolError("Cannot divide by zero")
    return str(a / b)
```

`ToolError` is caught by the runtime and fed back to the LLM as an error result. Other exceptions are wrapped in `ToolError` automatically.

### Runtime Tool Addition

```python
agent = Agent(name="bot", tools=[search])

# Add a tool dynamically (async-safe via _tools_lock)
await agent.add_tool(new_tool)

# Remove a tool
agent.remove_tool("old_tool_name")
```

## Gotchas

- **Tool names must be unique per agent** — duplicate names raise `AgentError` at registration
- **`retrieve_artifact` is auto-registered** — don't manually create a tool with this name
- **Omitting `Args:` docstring** means no parameter descriptions in the schema — the LLM still sees parameter names and types but no descriptions
- **Sync tools run in a thread** via `asyncio.to_thread()` — they block a thread pool thread, not the event loop
- **`ToolError` vs other exceptions:** `ToolError` passes through directly; all other exceptions are wrapped in `ToolError(f"Tool '{name}' failed: {exc}")`
- **Schema fallback:** Unannotated parameters default to `{"type": "string"}`
- **`large_output` requires exo-context** — without it installed, offloading silently falls back to returning the full content
