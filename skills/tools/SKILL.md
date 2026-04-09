---
name: exo:tools
description: "Use when creating tools for Exo agents — @tool decorator, FunctionTool, Tool ABC subclass, ToolContext for nested agent streaming, injected_tool_args, large_output, structured output (output_type), MCP server integration, tool_gate conditional injection, or tool offloading via CLI. Triggers on: exo tool, @tool, FunctionTool, Tool ABC, ToolContext, injected_tool_args, large_output, output_type, add_mcp_server, structured output, nested agent, inner agent events, tool offloading, exo tool call, exo tool list, exo tool schema, bash tool, tool_gate, conditional tool, gated tool, deferred tool."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Tools — Advanced Tool Creation

## When To Use This Skill

Use this skill when the developer needs to:
- Create tools for an Exo agent using `@tool`, `FunctionTool`, or `Tool` ABC
- Forward inner agent streaming events via `ToolContext`
- Configure schema-only injected arguments (`injected_tool_args`)
- Handle large tool output with workspace offloading (`large_output`)
- Define structured output schemas (`output_type`)
- Connect MCP servers to agents (`add_mcp_server`)
- Conditionally inject tools after a trigger fires (`tool_gate`) — KV-cache safe
- Understand how tool schemas are generated from function signatures
- Offload tool execution to CLI (`exo tool list/call/schema`) for token-efficient agent operation

## Decision Guide

Ask these questions to pick the right pattern:

1. **Simple function → tool?** Use `@tool` decorator (handles sync/async, schema generation)
2. **Need to customize name/description programmatically?** Use `FunctionTool(fn, name=..., description=...)`
3. **Need stateful tool with custom execute logic?** Subclass `Tool` ABC
4. **Tool runs an inner agent and you want its events in the parent stream?** Declare `ctx: ToolContext` — auto-injected, hidden from LLM
5. **Tool returns huge results (>10KB)?** Set `large_output=True` — result offloaded to workspace, LLM gets a pointer
6. **Need the LLM to fill a parameter the tool never receives?** Use `injected_tool_args` on Agent
7. **Need validated structured output from the agent?** Set `output_type=MyPydanticModel` on Agent
8. **Need tools from an MCP server?** Use `agent.add_mcp_server(config)`
9. **Need tools to appear only after a trigger tool fires?** Use `tool_gate` on Agent — KV-cache safe, append-only

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
- `ToolContext`-typed parameters are skipped (injected at runtime, not exposed to LLM)
- Multi-line `Args:` descriptions are joined with spaces

### ToolContext (Nested Agent Event Forwarding)

When a tool internally runs another agent, the inner agent's stream events are normally invisible. `ToolContext` makes them visible by providing an `emit()` method that pushes events to the parent agent's stream.

**Opt-in:** Declare a `ToolContext` parameter in your tool function. It's auto-injected at runtime and excluded from the LLM-visible schema.

```python
from exo import tool, Agent, run, ToolContext
from exo.types import TextEvent

@tool
async def deep_research(query: str, ctx: ToolContext) -> str:
    """Run a research agent and stream its progress.

    Args:
        query: The research topic.
    """
    inner = Agent(name="researcher", instructions="Research deeply.")
    parts = []
    async for event in run.stream(inner, query, provider=provider):
        ctx.emit(event)           # Forward event to parent's stream
        if isinstance(event, TextEvent):
            parts.append(event.text)
    return "".join(parts)

# Parent sees inner events in its stream
parent = Agent(name="orchestrator", tools=[deep_research])
async for event in run.stream(parent, "AI trends", provider=provider):
    print(f"[{event.agent_name}] {event.type}")
```

**How it works:**
1. `FunctionTool` detects the `ToolContext`-typed parameter at init time
2. Schema generation skips it — the LLM never sees it
3. `Agent._execute_tools()` injects a `ToolContext(agent_name, _event_queue)` into kwargs
4. After tool execution, `run.stream()` drains the queue and yields all buffered events
5. Inner events retain their original `agent_name` for distinguishing sources

**Parameter name is flexible:** `ctx`, `tool_ctx`, `context` — any name works as long as the type is `ToolContext`.

**Only works with `FunctionTool` / `@tool`:** Custom `Tool` ABC subclasses don't support auto-injection.

### ToolContext.require_approval() (On-Demand HITL)

Tools can self-gate with human approval using `ctx.require_approval()`. The tool decides when approval is needed based on its own logic — no agent-level configuration required.

**Setup:** Set `human_input_handler` on the Agent to provide the approval mechanism:

```python
from exo import Agent, tool, ToolContext, ConsoleHandler

@tool
async def run_command(command: str, ctx: ToolContext) -> str:
    """Execute a shell command.

    Args:
        command: The shell command to run.
    """
    # Gate sensitive operations
    if any(p in command for p in ["rm ", "sudo ", "DROP ", "chmod "]):
        await ctx.require_approval(f"Sensitive command: {command}\nApprove?")
    return subprocess.check_output(command, shell=True, text=True)

agent = Agent(
    name="ops",
    tools=[run_command],
    human_input_handler=ConsoleHandler(),  # provides the approval mechanism
)
```

**How it works:**
1. Tool calls `await ctx.require_approval(message)` at any point during execution
2. The handler (from `Agent.human_input_handler`) prompts the human with `message` and choices `["yes", "no"]`
3. If approved ("yes"/"y"), execution continues normally
4. If denied, raises `ToolError("Tool execution denied by human")` — the LLM sees the denial and can adjust

**Key behaviors:**
- `require_approval()` raises `ToolError` if no `human_input_handler` is set on the agent — fail-fast, never silently skip approval
- The tool controls when and why approval is needed — dynamic gating based on arguments, state, or any condition
- Works with any `HumanInputHandler` implementation (console, web UI, Slack, etc.)
- The `message` parameter is fully customizable — include relevant context for the human reviewer

### injected_tool_args (Schema-Only Arguments)

Arguments that appear in the LLM's tool schema and the LLM fills in, but are **stripped before the tool executes**. The tool function never receives them.

Use cases:
- **Chain-of-thought elicitation:** Force the LLM to reason before calling a tool
- **Confidence scoring:** Have the LLM declare confidence without the tool caring
- **Correlation IDs:** Let the LLM echo back trace IDs visible in logs but irrelevant to tool logic

```python
agent = Agent(
    name="assistant",
    tools=[search, write_file],
    injected_tool_args={
        "reasoning": "Explain why you are calling this tool",
        "confidence": "Your confidence level: low, medium, or high",
    },
)
```

**How it works (implementation details):**

1. `Agent.get_tool_schemas()` deep-copies each tool schema and adds injected fields as **optional string properties**. The underlying `Tool.parameters` is never mutated.
2. The LLM sees and fills these fields in its tool call JSON.
3. `Agent._execute_tools()` strips all injected arg keys from `kwargs` before calling `tool.execute()`.

```python
# What the LLM returns:
# {"reasoning": "Need to find the API docs", "query": "python requests library"}

# What the tool receives (reasoning stripped):
# {"query": "python requests library"}
```

**Collision safety:** If an injected arg name collides with a real tool parameter, the real parameter wins — the injected field is not added to that tool's schema.

**Works with all tool types:** `@tool`, `FunctionTool`, `Tool` ABC, `MCPToolWrapper` — any tool registered on the agent gets augmented schemas and stripped execution args.

**Validation:** Keys must be non-empty strings, values must be strings (used as the property description in the JSON schema).

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

### tool_gate (Conditional Tool Injection — KV-Cache Safe)

Declare tools that become available only after a specific trigger tool executes. Gated tools are **appended** to the tool list (never reordered), preserving the LLM's KV-cache prefix.

```python
from exo import Agent, tool

@tool
def search(query: str) -> str:
    """Search the database."""
    return f"results for {query}"

@tool
def write_record(data: str) -> str:
    """Write a record to the database."""
    return f"wrote {data}"

@tool
def delete_record(record_id: str) -> str:
    """Delete a record from the database."""
    return f"deleted {record_id}"

agent = Agent(
    name="db_agent",
    tools=[search],
    tool_gate={
        "search": [write_record, delete_record],
    },
)
```

**How it works:**
1. Gated tools (`write_record`, `delete_record`) are **not registered** at init — the LLM cannot see or call them
2. When the trigger tool (`search`) executes, a `POST_TOOL_CALL` hook fires and **appends** the gated tools
3. On the next LLM turn, the new tools appear in `get_tool_schemas()` and are callable
4. The gate is idempotent — calling the trigger again does not duplicate tools

**Why it's KV-cache safe:**
- LLM providers (Anthropic, OpenAI) cache based on prefix matching: `[system + tools + messages]`
- Gated tools are appended at the end — the existing prefix stays identical
- Before: `[sys, tools[A,B,C], msgs]` → After: `[sys, tools[A,B,C, D,E], msgs]` — prefix `[A,B,C]` is a cache hit

**Multiple independent gates:**

```python
agent = Agent(
    name="bot",
    tools=[read_tool, auth_tool],
    tool_gate={
        "read_tool": [analyze_tool],       # read unlocks analyze
        "auth_tool": [write_tool, admin],   # auth unlocks write + admin
    },
)
```

**Validation:**
- Trigger names must refer to already-registered tools — raises `AgentError` otherwise
- Gated tools must have unique names (standard duplicate-name rules apply)

## Tool Offloading via CLI

Tool offloading lets agents execute Exo tools via bash commands instead of LLM tool calling, keeping schemas out of the context window and reducing token usage.

### Commands

```bash
# List available tools from a Python module or file
exo tool list --from myapp.tools
exo tool list --from ./tools.py
exo tool list --from myapp.tools --json    # JSON output

# Call a tool with key=value args (types auto-coerced from schema)
exo tool call greet --from myapp.tools -a name=Alice -a greeting=Hey

# Call a tool with JSON args (preserves types natively)
exo tool call search --from myapp.tools -j '{"query": "python", "max_results": 5}'

# Get full JSON schema for a specific tool
exo tool schema search --from myapp.tools

# Inject args (like injected_tool_args — auto-filled, LLM doesn't specify)
exo tool call search --from myapp.tools -j '{"query": "test"}' -i api_key=sk-123 -i user_id=u42

# Raw JSON output (for programmatic consumption)
exo tool call greet --from myapp.tools -a name=X --raw
```

### Environment Variables

Set these once (e.g., in `.bashrc`, `.zshenv`, or agent bootstrap) to avoid repeating flags:

| Variable | Purpose | Example |
|---|---|---|
| `EXO_TOOL_SOURCE` | Default for `--from` — Python module or file path | `myapp.tools` or `./tools.py` |
| `EXO_TOOL_INJECT` | Default injected args — JSON object, merged into every call | `'{"user_id": "u42", "api_key": "sk-123"}'` |

With both set, agents just run bare commands:
```bash
export EXO_TOOL_SOURCE=myproject.tools
export EXO_TOOL_INJECT='{"user_id": "u42", "api_key": "sk-123"}'

exo tool list                                     # no --from needed
exo tool call search -j '{"query": "test"}'       # user_id + api_key auto-injected
exo tool schema search                            # no --from needed
```

`--from` overrides `EXO_TOOL_SOURCE` when provided. Explicit `--arg`/`--json`/`--inject` values override `EXO_TOOL_INJECT`.

### Injected Arguments (`--inject` / `-i` and `EXO_TOOL_INJECT`)

The CLI equivalent of `injected_tool_args`. These arguments are:
- Visible in the tool schema (LLM knows they exist via `exo tool schema`)
- Auto-filled at call time (LLM doesn't need to specify them)
- Overridable by higher-precedence sources if explicitly provided

**Precedence (lowest → highest):** `EXO_TOOL_INJECT` → `--inject` → `--json` → `--arg`

Two ways to configure:

```bash
# 1. Environment variable (set once, always active)
export EXO_TOOL_INJECT='{"api_key": "sk-123", "user_id": "u42"}'
exo tool call search -j '{"query": "test"}'

# 2. CLI flag (per-call)
exo tool call search -j '{"query": "test"}' -i api_key=sk-123 -i user_id=u42

# 3. Both (flag overrides env for overlapping keys)
export EXO_TOOL_INJECT='{"api_key": "sk-123", "user_id": "default"}'
exo tool call search -j '{"query": "test"}' -i user_id=override
```

### How It Works

The `--from` / `EXO_TOOL_SOURCE` accepts either:
- **Dotted module path:** `myapp.tools` — imported via `importlib.import_module()`
- **File path:** `./tools.py` or `/abs/path/tools.py` — imported via `importlib.util`

Tool discovery scans all module-level attributes for `Tool` instances, plus items in a conventional `tools` list/tuple.

### Type Coercion

When using `--arg KEY=VALUE` or `--inject KEY=VALUE`, string values are auto-coerced based on the tool's JSON Schema:
- `"integer"` → `int("42")` → `42`
- `"number"` → `float("3.14")` → `3.14`
- `"boolean"` → `"true"/"1"/"yes"` → `True`
- `"array"/"object"` → `json.loads(value)`

With `--json` and `EXO_TOOL_INJECT`, types are preserved natively from JSON.

### Agent Integration Pattern

An agent can offload tools to bash for token-efficient execution. With env vars pre-configured, instructions are minimal:

```python
# Bootstrap: set env vars before agent starts
import os
os.environ["EXO_TOOL_SOURCE"] = "myproject.tools"
os.environ["EXO_TOOL_INJECT"] = '{"api_key": "sk-123", "user_id": "u42"}'

agent = Agent(
    name="efficient-bot",
    instructions="""You have access to tools via the command line.
To list available tools: exo tool list
To call a tool: exo tool call <name> -j '{"arg": "val"}'
To see a tool's schema: exo tool schema <name>
Use the bash tool to execute these commands instead of calling tools directly.""",
    tools=[bash_tool],  # Only needs a bash/shell tool
)
```

### Source: `packages/exo-cli/src/exo_cli/tool_commands.py`

## Gotchas

- **Tool names must be unique per agent** — duplicate names raise `AgentError` at registration
- **`retrieve_artifact` is auto-registered** — don't manually create a tool with this name
- **Omitting `Args:` docstring** means no parameter descriptions in the schema — the LLM still sees parameter names and types but no descriptions
- **Sync tools run in a thread** via `asyncio.to_thread()` — they block a thread pool thread, not the event loop
- **`ToolError` vs other exceptions:** `ToolError` passes through directly; all other exceptions are wrapped in `ToolError(f"Tool '{name}' failed: {exc}")`
- **Schema fallback:** Unannotated parameters default to `{"type": "string"}`
- **`large_output` requires exo-context** — without it installed, offloading silently falls back to returning the full content
- **`ToolContext` only works with `@tool` / `FunctionTool`** — `Tool` ABC subclasses and MCP tools don't get auto-injection
- **`ToolContext.emit()` is non-blocking** — events are buffered and drained after tool execution, not yielded in real-time
- **Sync tools can't use `ToolContext`** — `emit()` calls `asyncio.Queue.put_nowait()` which is unsafe from a thread. Use async tools for inner agent streaming.
- **`require_approval()` needs `human_input_handler`** — raises `ToolError` if the agent has no handler set. This is intentional: if a tool requires approval, silently skipping it is a security risk.
- **`require_approval()` needs async tools** — it's an `await` call, so the tool must be `async def`
- **Tool offloading: `--from` must be importable** — the module's dependencies must be installed in the current Python environment. If using a file path, it's loaded in isolation.
- **Tool offloading: ToolContext not available** — CLI-invoked tools don't have an agent context, so `ToolContext`-dependent tools will fail. Use tools without `ToolContext` for offloading.
- **Tool offloading: async tools work fine** — `asyncio.run()` drives async tool execution from the CLI.
- **`tool_gate` triggers must be registered tools** — passing a trigger name that isn't in `tools=` raises `AgentError` at init
- **`tool_gate` is one-shot** — once a gate opens, the tools stay registered for the rest of the run. There is no "re-lock" mechanism.
- **`tool_gate` and PTC** — gated tools added at runtime are not automatically PTC-eligible. If `ptc=True`, gated tools appear as LLM-callable schemas (not PTC functions).
