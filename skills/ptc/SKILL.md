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
    ptc=True,                        # Registers PTC tool automatically
    ptc_timeout=60,                  # Timeout for code execution (seconds, default 60)
    ptc_max_output_bytes=200_000,    # Truncate captured stdout+stderr above this
    ptc_max_tool_calls=200,          # Cap inner tool calls per PTC invocation
)
```

**Agent parameters:**
- `ptc: bool = False` — When `True`, registers the internal PTC tool (`__exo_ptc__`) and hides PTC-eligible tools from the schema list (they become functions inside the PTC tool instead).
- `ptc_timeout: int = 60` — Maximum seconds for a single PTC invocation. Long-running code is terminated with a `TimeoutError` message.
- `ptc_max_output_bytes: int = 200_000` — Maximum bytes of captured stdout+stderr returned to the model. Larger outputs are truncated with a `[truncated N chars]` suffix.
- `ptc_max_tool_calls: int = 200` — Maximum number of inner tool calls the user code can make per PTC invocation. Exceeding raises `MaxToolCallsExceeded`.

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

### Sandbox

PTC code runs inside a restricted Python namespace — the goal is to push the agent toward registered tools and away from the lazy "just use Python" escape hatches. Four layers of defense, in order:

1. **Restricted `__builtins__` whitelist.** Only a curated subset of builtins is available: type constructors (`str`, `int`, `list`, `dict`, `set`, `tuple`, `bool`, …), sequence/iteration (`len`, `range`, `enumerate`, `zip`, `sorted`, `reversed`, `map`, `filter`, `iter`, `next`, `slice`), numerics (`abs`, `round`, `min`, `max`, `sum`, `pow`, `divmod`), logic (`any`, `all`), string/format (`chr`, `ord`, `hex`, `oct`, `bin`, `ascii`, `repr`, `format`), type checks (`isinstance`, `issubclass`, `callable`, `type`, `object`), and the full set of safe exceptions. `print` is available and captured via stdout redirect.

2. **Blocked builtins.** `open`, `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`, `vars`, `dir`, `breakpoint`, `input`, `exit`, `quit`, `help`, `memoryview`, `__build_class__` are replaced with stubs that raise `PTCSandboxError` pointing at `default_api`. The errors are catchable so user code can `try/except` and fall back.

3. **Blocked imports.** A custom `__import__` hook rejects the expanded dangerous stdlib blocklist with `ImportError`:
   - **Filesystem/process:** `os`, `sys`, `subprocess`, `shutil`, `pathlib`, `tempfile`, `glob`, `fnmatch`, `fcntl`, `resource`
   - **I/O bypass:** `io`, `builtins`, `ctypes`, `mmap`
   - **Network:** `socket`, `urllib`, `http`, `httplib`, `ssl`, `ftplib`, `smtplib`, `poplib`, `imaplib`, `telnetlib`, `nntplib`
   - **Code/import/exec:** `importlib`, `pkgutil`, `runpy`, `code`, `codeop`
   - **Introspection/gc:** `inspect`, `gc`, `traceback`
   - **Concurrency:** `threading`, `multiprocessing`, `concurrent` (asyncio is pre-imported and allowed)
   - **System-level:** `signal`, `pty`, `tty`, `select`, `termios`, `pwd`, `grp`

   Only re-imports of already-pre-loaded modules are allowed: `json`, `math`, `re`, `asyncio`, `collections`, `itertools`, `datetime`. `import json as j` works as a no-op.

4. **AST pre-scan for dunder escape hatches.** Before compile, the code is statically walked and attribute access on `__class__`, `__bases__`, `__base__`, `__subclasses__`, `__mro__`, `__globals__`, `__builtins__`, `__import__`, `__getattribute__`, `__setattr__`, `__delattr__`, `__dict__`, `__code__`, `__closure__`, `__func__`, `__self__`, `__loader__`, `__spec__`, `__init_subclass__`, `__new__`, `__reduce__`, `__reduce_ex__` is rejected with a clear error. This blocks the classic escape chain `().__class__.__bases__[0].__subclasses__()`.

5. **Runtime `safer_getattr` guard.** Dynamic attribute access via `getattr(obj, name)` goes through `RestrictedPython.Guards.safer_getattr`, which rejects any attribute name starting with `_`. This catches the dynamic form of the same escapes (`getattr(x, "__class__")`).

**Philosophy:** the sandbox closes the *lazy* escape paths. Agents that try to read a file via `open("/etc/passwd")`, shell out via `subprocess`, or walk the class hierarchy via `__class__.__bases__` get a catchable error telling them to use `default_api` instead. A determined attacker with deep Python introspection can still find edge-case bypasses — for those, route PTC through a subprocess sandbox (future `exo-sandbox` backend work).

**Escape hatch for specific tools**: set `tool._ptc_exclude = True` on a tool to keep it as a direct schema (outside PTC's namespace). Useful for tools that need unrestricted access to the host environment.

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

### PTC with Tool-Level require_approval()

Tools that use `ToolContext.require_approval()` for on-demand HITL can also participate in PTC — but only if they are NOT in `hitl_tools`. The `require_approval()` call blocks inside the PTC code execution when triggered:

```python
@tool
async def execute_trade(ticker: str, amount: float, ctx: ToolContext) -> str:
    """Execute a stock trade."""
    if amount > 10000:
        await ctx.require_approval(f"Large trade: {ticker} ${amount}. Approve?")
    return f"Executed: {ticker} ${amount}"

agent = Agent(
    name="trader",
    tools=[get_portfolio, get_market_data, execute_trade],
    human_input_handler=ConsoleHandler(),
    ptc=True,
    # execute_trade is PTC-eligible (not in hitl_tools) but self-gates for large amounts
)
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
- **Code runs in a restricted in-process namespace** — `open`, `eval`, `exec`, `compile`, `os`, `sys`, `subprocess`, `pathlib`, `socket`, `threading`, `ctypes`, `inspect`, and most of the dangerous stdlib are blocked with `ImportError` / `PTCSandboxError`. The errors are catchable and the messages direct the agent at `default_api`. See the **Sandbox** section above for the full rules. The sandbox closes lazy escape paths; true isolation against determined attackers requires a subprocess backend (not yet implemented).
- **`asyncio.wait_for` timeout doesn't interrupt sync loops** — `ptc_timeout` catches `await asyncio.sleep(forever)` but not `while True: pass`. This is a fundamental limitation of in-process execution; a subprocess backend is the long-term fix.
- **No in-process memory cap** — a pathological `[0] * 10**10` can OOM the agent process. Same subprocess-backend answer.
- **Output is capped** — captured stdout+stderr is truncated to `ptc_max_output_bytes` (default 200 KiB). Long outputs get a `\n...[truncated N chars]` suffix.
- **Tool-call count is capped** — user code can make at most `ptc_max_tool_calls` (default 200) inner tool calls per invocation. Exceeding raises `MaxToolCallsExceeded`.
- **Code size is capped** — incoming code strings are rejected above `ptc_max_code_bytes` (default 100 KiB) with a clear error.
- **Orphan asyncio tasks are cancelled** — if user code does `asyncio.create_task(...)` without awaiting, PTC cancels the task after `__ptc_main__` completes so it cannot leak output or run forever.
- **Traceback line numbers match your code** — PTC rewrites `<ptc>` line numbers in tracebacks to remove the 2-line internal wrapper offset, so errors point at the agent's own code.
- **Tool names must be valid Python identifiers** — tools with hyphens (e.g., `get-data`) are automatically excluded from PTC and remain available as direct tool schemas. A warning is logged once per excluded tool.
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
