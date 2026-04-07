---
name: exo:ptc
description: "Use when enabling Programmatic Tool Calling on Exo agents — ptc=True, ptc_timeout, PTCExecutor, PTCTool, batch tool calls in code, reduce LLM round-trips, context-efficient multi-tool workflows, PTC-eligible tools, HITL exclusion, Swarm PTC propagation, transparent streaming events. Triggers on: ptc, programmatic tool calling, ptc=True, ptc_timeout, PTCExecutor, PTCTool, batch tools, reduce round-trips, code execution tool."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo PTC — Programmatic Tool Calling

## When To Use This Skill

Use this skill when the developer needs to:
- Reduce LLM round-trips by letting the model write Python code that calls multiple tools in one step
- Keep intermediate tool results out of the LLM context window (token-efficient)
- Enable batch processing, loops, filtering, or aggregation over tool results
- Turn on PTC with a single flag (`ptc=True`) on an Agent or Swarm
- Understand which tools are PTC-eligible vs direct
- Ensure HITL tools and framework tools remain as direct calls
- Configure PTC timeout or understand PTC hooks integration

## Decision Guide

1. **Does the agent call multiple tools per task?** → Consider `ptc=True`
2. **Are tool results large but only summaries matter?** → `ptc=True` keeps raw data out of LLM context
3. **Does the agent loop over similar calls (e.g., 20 regions)?** → `ptc=True` collapses N round-trips to 1
4. **Does the agent have HITL tools requiring human approval?** → Safe: HITL tools stay direct, PTC never bypasses approval
5. **Multiple providers?** → PTC is provider-agnostic — works with OpenAI, Anthropic, Gemini, any model
6. **Need to customize PTC timeout?** → Set `ptc_timeout` (default 60 seconds)
7. **Swarm of agents that should all use PTC?** → Set `ptc=True` on the Swarm

## Reference

### Enabling PTC

```python
from exo import Agent

agent = Agent(
    name="analyst",
    model="openai:gpt-4o",
    instructions="Analyze data efficiently. Use code to batch operations.",
    tools=[search, query_db, get_expenses],
    ptc=True,           # Registers PTC tool automatically
    ptc_timeout=60,     # Timeout for code execution (seconds, default 60)
)
```

**Agent parameters:**
- `ptc: bool = False` — When `True`, registers the internal PTC tool (`__exo_ptc__`) and hides PTC-eligible tools from the schema list (they become functions inside the PTC tool instead)
- `ptc_timeout: int = 60` — Maximum seconds for a single PTC invocation. Long-running code is terminated with a `TimeoutError` message.

### How It Works

```
Without PTC (ptc=False):
  LLM → tool_use: search("A") → result → LLM → tool_use: search("B") → result → LLM → ...
  N tools = N round-trips, all results in context

With PTC (ptc=True):
  LLM → tool_use: __exo_ptc__(code="""
      results = {}
      for region in ["A", "B", "C", ...]:
          data = json.loads(await default_api.search(query=region))
          results[region] = sum(d['revenue'] for d in data)
      top = sorted(results.items(), key=lambda x: x[1])[-3:]
      print(f"Top 3: {top}")
  """) → execute all tools inside code → return "Top 3: ..." → LLM
  1 round-trip, only the summary in context
```

### Stream Transparency

PTC is **fully transparent** to the event stream. The internal `__exo_ptc__` tool never appears in events — instead, individual `ToolCallEvent`/`ToolResultEvent` are emitted for each inner tool call:

```python
# With ptc=True, calling add() then subtract() inside PTC code:
# The stream shows:
#   ToolCallEvent(tool_name="add", ...)
#   ToolResultEvent(tool_name="add", result="5", ...)
#   ToolCallEvent(tool_name="subtract", ...)
#   ToolResultEvent(tool_name="subtract", result="6", ...)
#
# NOT:
#   ToolCallEvent(tool_name="__exo_ptc__", ...)
#   ToolResultEvent(tool_name="__exo_ptc__", ...)
```

This means UI consumers never need to know PTC exists. The event stream looks identical to non-PTC mode (except with fewer LLM steps).

### Tool Classification

When `ptc=True`, tools are split into two groups:

| Category | In schema list? | Available in PTC code? |
|----------|----------------|----------------------|
| User tools (`@tool`, `FunctionTool`, `Tool` subclass) | NO | YES — as `await default_api.tool_name(...)` |
| MCP tools (`add_mcp_server`) | NO | YES — as `await default_api.mcp__server__tool(...)` |
| `__exo_ptc__` (internal) | YES | NO (itself) |
| Framework tools (`retrieve_artifact`, `spawn_self`, `activate_skill`) | YES | NO |
| Context tools (`_is_context_tool`) | YES | NO |
| HITL tools (`hitl_tools`) | YES | NO — approval flow preserved |
| Handoff targets | YES | NO |

The LLM sees the PTC tool plus any direct tools. Inside the code, tools are accessed via the `default_api` namespace to avoid collisions with Python builtins and keywords.

### PTC Tool Internals

Auto-registered when `ptc=True` as `__exo_ptc__`. The LLM calls it with Python code:

```python
# What the LLM sends:
__exo_ptc__(code="""
members = json.loads(await default_api.get_team_members(department="engineering"))
over_budget = []
for m in members:
    expenses = json.loads(await default_api.get_expenses(employee_id=m['id'], quarter='Q3'))
    travel = sum(e['amount'] for e in expenses if e['category'] == 'travel')
    if travel > m['budget']:
        over_budget.append(f"{m['name']}: ${travel:,.0f}")
print("Over budget:\\n" + "\\n".join(over_budget))
""")
```

**Code execution environment:**
- Tools accessed via `default_api` namespace — use `await default_api.tool_name(...)` to call them
- `print()` captures to stdout — this is the primary output mechanism
- Return values are also captured (combined with stdout)
- Standard library modules pre-loaded: `json`, `math`, `re`, `asyncio`, `collections`, `itertools`, `datetime`
- `asyncio.gather()` works for parallel tool calls: `await asyncio.gather(default_api.a(), default_api.b())`
- The `default_api` namespace prevents tool names from colliding with Python builtins (`map`, `list`, `type`, `id`, etc.), keywords (`return`, `class`), or stdlib modules (`json`, `math`, `re`)

### PTCExecutor

The executor runs the code in-process:

1. Builds namespace with tool wrapper functions + stdlib modules
2. Wraps code in `async def __ptc_main__():\n    {code}`
3. Compiles (catches `SyntaxError` → returns error string)
4. Executes with `asyncio.wait_for(timeout=ptc_timeout)`
5. Returns captured stdout + `repr(return_value)`

**Error handling:**
- `SyntaxError` → returns `"SyntaxError: ..."` as tool result
- Runtime exceptions → returns traceback as tool result
- Tool execution errors → propagate as exceptions within the code (catchable with `try/except`)
- Timeout → returns `"TimeoutError: execution exceeded Ns"`

### Hooks Integration

`PRE_TOOL_CALL` and `POST_TOOL_CALL` hooks fire for **each inner tool call** inside PTC code:

```python
async def log_tool_call(**kwargs):
    print(f"Tool: {kwargs['tool_name']}, Args: {kwargs['arguments']}")

agent = Agent(
    name="bot",
    tools=[search, query_db],
    ptc=True,
    hooks=[(HookPoint.PRE_TOOL_CALL, log_tool_call)],
)
# When PTC runs and calls default_api.search() + default_api.query_db() inside the code,
# log_tool_call fires twice — once per inner tool call
```

Hook arguments for inner PTC calls:
- `agent`: the parent agent
- `tool_name`: the actual tool name (e.g., `"search"`, not `"__exo_ptc__"`)
- `arguments`: the kwargs dict passed to the tool
- `result` (POST only): `ToolResult` with the inner tool's call ID

### ToolContext Support

FunctionTools that declare a `ToolContext` parameter get it injected inside PTC, just like in normal execution:

```python
@tool
def emit_progress(status: str, ctx: ToolContext) -> str:
    """Report progress."""
    ctx.emit(StatusEvent(status=status, agent_name=ctx.agent_name))
    return f"Reported: {status}"

agent = Agent(name="bot", tools=[emit_progress, search], ptc=True)
# Inside PTC code, await default_api.emit_progress(status="step 1") works — ToolContext injected
```

### Swarm Propagation

```python
from exo import Agent, Swarm

a1 = Agent(name="a1", tools=[search])
a2 = Agent(name="a2", tools=[query_db])

swarm = Swarm(agents=[a1, a2], ptc=True)
# a1.ptc == True, a2.ptc == True — propagated to all members
```

- `ptc=None` (default): no propagation, each agent keeps its own setting
- `ptc=True`: all member agents get `ptc=True`
- `ptc=False`: all member agents get `ptc=False`

### Serialization

PTC settings survive `to_dict()` / `from_dict()`:

```python
data = agent.to_dict()
# data["ptc"] == True
# data["ptc_timeout"] == 60
# "__exo_ptc__" NOT in data["tools"] (auto-registered, not serialized)

restored = Agent.from_dict(data)
# restored.ptc == True, PTC tool re-registered
```

## Patterns

### Batch Data Processing

```python
agent = Agent(
    name="expense_auditor",
    model="anthropic:claude-sonnet-4-5-20250929",
    instructions=(
        "You audit employee expenses. When checking multiple employees, "
        "write code that fetches all records, filters violations, and "
        "returns only the summary. Do not return raw expense data."
    ),
    tools=[get_team_members, get_expenses, get_budget],
    ptc=True,
)

result = await run(agent, "Check Q3 travel budget compliance for engineering")
# LLM writes code that:
# 1. Fetches team members
# 2. Loops over each, fetching expenses
# 3. Filters to travel category, sums amounts
# 4. Compares against budget
# 5. Prints only violators
# → 1 LLM round-trip instead of 15+
```

### Parallel Search with Aggregation

```python
agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions=(
        "Research topics by searching multiple queries in parallel. "
        "Use asyncio.gather() for concurrent searches. "
        "Aggregate and deduplicate results before returning."
    ),
    tools=[web_search],
    ptc=True,
    ptc_timeout=120,  # Allow more time for multiple searches
)
```

The LLM writes:
```python
queries = ["topic A latest", "topic A research", "topic A comparison"]
results = await asyncio.gather(*[default_api.web_search(query=q) for q in queries])
all_items = []
for r in results:
    all_items.extend(json.loads(r))
unique = {item['url']: item for item in all_items}
print(json.dumps(list(unique.values())[:10]))
```

### PTC with HITL Approval Tools

```python
@tool
def execute_trade(ticker: str, amount: float) -> str:
    """Execute a stock trade (requires human approval)."""
    return f"Executed: {ticker} ${amount}"

agent = Agent(
    name="trader",
    tools=[get_portfolio, get_market_data, execute_trade],
    hitl_tools=["execute_trade"],  # Requires human approval
    ptc=True,
)
# get_portfolio and get_market_data → inside PTC code as default_api.get_portfolio() etc. (batch-friendly)
# execute_trade → stays as direct tool (approval flow preserved)
# LLM can batch-read portfolio data in code, then call execute_trade directly
```

### PTC with MCP Servers

```python
agent = Agent(
    name="ops",
    model="openai:gpt-4o",
    tools=[local_analyzer],
    ptc=True,
)
await agent.add_mcp_server(MCPServerConfig(
    name="github",
    command="github-mcp-server",
    transport="stdio",
))
# MCP tools are now available via default_api.mcp__github__search_repos() etc.
# alongside default_api.local_analyzer()
```

### Dynamic Tool Addition with PTC

```python
agent = Agent(name="bot", tools=[search], ptc=True)

# Later, at runtime:
await agent.add_tool(query_db)
# PTC tool description now includes query_db
# Schema cache auto-invalidated

agent.remove_tool("search")
# PTC tool description no longer includes search
```

## Gotchas

- **PTC is provider-agnostic** — it works by injecting a standard tool, not by using any provider-specific API (like Anthropic's `code_execution_20260120`). Any LLM that supports tool calling can use PTC.
- **Code runs in-process** — there is no sandbox. The code executes in the same Python runtime as the agent. This is fast but means the LLM-generated code has access to the process environment. For untrusted models, consider additional guardrails.
- **`asyncio.wait_for` timeout doesn't interrupt sync loops** — `ptc_timeout` catches `await asyncio.sleep(forever)` but not `while True: pass`. This is acceptable because the LLM writes the code and `max_steps` provides an outer guard.
- **Tool results inside PTC are strings** — tools return strings (or JSON-serialized dicts/lists). The code must `json.loads()` to work with structured data.
- **Tools live in `default_api` namespace** — this prevents collisions with Python builtins (`map`, `list`, `filter`, `type`, `id`), keywords (`return`, `class`, `for`), and stdlib modules (`json`, `math`, `re`). Always use `await default_api.tool_name(...)`.
- **HITL tools are excluded from PTC** — they stay as direct schemas so the human approval flow is never bypassed. The LLM sees both the PTC tool and the HITL tools as separate callable tools.
- **Handoff tools are excluded from PTC** — handoffs must be direct tool calls to trigger agent transitions correctly.
- **`__exo_ptc__` name is reserved** — if a user tool is already named `__exo_ptc__`, setting `ptc=True` raises `AgentError`. This name uses dunder convention to avoid collisions.
- **Dynamic tools are reflected automatically** — `add_tool()` / `remove_tool()` invalidate the schema cache, and PTCTool rebuilds its description from live `agent.tools` on next `get_tool_schemas()` call.
- **Hooks fire per inner call** — `PRE_TOOL_CALL` and `POST_TOOL_CALL` fire for each `await tool_name()` inside the code.
- **Guide the LLM via instructions** — be explicit: "Write code to batch-process all items and return only the summary." Without guidance, the model may still use individual tool calls if PTC-eligible tools happen to also be in the schema (they're not, but the model may not realize PTC is available without clear instructions).
- **Stream is fully transparent** — `run.stream()` emits individual `ToolCallEvent`/`ToolResultEvent` per inner tool call. The `__exo_ptc__` tool never appears in the event stream. UI consumers never need to know PTC exists.
- **`output_type` works with PTC** — structured output validation happens after the tool loop ends, so PTC doesn't interfere.
