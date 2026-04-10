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
    ptc_max_code_bytes=100_000,      # Cap incoming user code size
    ptc_extra_args={                 # Schema fields added DIRECTLY to __exo_ptc__
        "intent": "Brief description of what this PTC invocation will do.",
    },
)
```

**Agent parameters:**
- `ptc: bool = False` — When `True`, registers the internal PTC tool (`__exo_ptc__`) and hides PTC-eligible tools from the schema list (they become functions inside the PTC tool instead).
- `ptc_timeout: int = 60` — Maximum seconds for a single PTC invocation. Long-running code is terminated with a `TimeoutError` message.
- `ptc_max_output_bytes: int = 200_000` — Maximum bytes of captured stdout+stderr returned to the model. Larger outputs are truncated with a `[truncated N chars]` suffix.
- `ptc_max_tool_calls: int = 200` — Maximum number of inner tool calls the user code can make per PTC invocation. Exceeding raises `MaxToolCallsExceeded`.
- `ptc_max_code_bytes: int = 100_000` — Maximum size of the incoming `code` string. Oversized code is rejected with a clear error before any execution.
- `ptc_extra_args: dict[str, str] | None = None` — Extra schema fields added to the **outer** `__exo_ptc__` tool call (not propagated to inner tools). The LLM fills them alongside `code`, and they become accessible inside the executing Python code as the `ptc_args` dict. Use this to surface metadata (intent, phase ID, confidence, …) or structured inputs the PTC code should act on. See the **PTC Extra Args** section below for details.

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

### Rich Tool Signatures

PTC renders each inner tool as a full Python `async def` block inside the `__exo_ptc__` description. **All** the metadata from the original `@tool` decorator survives into PTC mode — nothing gets dropped or truncated.

What the LLM sees for an inner tool:

```python
@tool(name="plan")
def plan_tool(
    action: Annotated[Literal["update", "advance", "get"], "The plan action to perform."],
    current_phase_id: Annotated[int | None, "ID of the current phase (auto-incrementing int starting from 1)."] = None,
    goal: Annotated[str | None, "The overall goal of the plan."] = None,
    phases: Annotated[list | None, "List of phase dicts. Each dict has 'id' (int), 'title' (str), ..."] = None,
) -> dict:
    """Create, update, and advance the structured task plan.

    <instructions>
    - This tool helps plan tasks and break down complex work into manageable phases.
    - MUST `update` when user makes new requests.
    - Phase count scales with complexity: simple (2), typical (4-6), complex (10+).
    </instructions>

    <recommended_usage>
    - Use `get` to retrieve the current plan.
    - Use `update` at the start of a new task.
    - Use `advance` when the current phase is complete.
    </recommended_usage>
    """
```

Is rendered into the `__exo_ptc__` description as:

```
  async def plan(
      action: Literal['update', 'advance', 'get'],
      current_phase_id: int | None = None,
      goal: str | None = None,
      phases: list | None = None,
  ) -> str
    """Create, update, and advance the structured task plan.

    <instructions>
    - This tool helps plan tasks and break down complex work into manageable phases.
    - MUST `update` when user makes new requests.
    - Phase count scales with complexity: simple (2), typical (4-6), complex (10+).
    </instructions>

    <recommended_usage>
    - Use `get` to retrieve the current plan.
    - Use `update` at the start of a new task.
    - Use `advance` when the current phase is complete.
    </recommended_usage>
    """

    Args:
        action: The plan action to perform.
        current_phase_id: ID of the current phase (auto-incrementing int starting from 1).
        goal: The overall goal of the plan.
        phases: List of phase dicts. Each dict has 'id' (int), 'title' (str), ...
```

What's preserved:

- **`Literal["a", "b", "c"]`** — enum/Literal constraints render as actual `Literal[...]` types, not collapsed to `str`. The LLM sees the allowed values directly.
- **`Annotated[T, "description"]`** — per-parameter descriptions surface in a Google-style `Args:` section under the docstring.
- **Full multi-line docstring** — untruncated, including `<instructions>`, `<recommended_usage>`, XML blocks, `Returns:`, `Raises:`, `Notes:`, `Examples:` — everything the author wrote. The only thing stripped is the `Args:` section inside the docstring (its contents are rendered separately to avoid duplication).
- **Default values** — `x: int = 42`, `mode: Literal['a', 'b'] = 'a'` — visible directly in the signature.
- **Required vs optional ordering** — required params come first, then optionals, then injected args. Stable alphabetical sort within each group.
- **`list[str]` / `list[int]`** item types when the JSON Schema provides `items.type`.
- **Long signatures auto-wrap** — single-line when ≤ 100 chars, multi-line (one param per line) when longer.
- **`injected_tool_args`** — appear as optional trailing params in the signature AND in the `Args:` section with a `[injected]` marker. Inside the PTC wrapper they are stripped from kwargs before the tool function is called (matching non-PTC dispatch semantics).

> **Footgun**: `from __future__ import annotations` turns type hints into strings. If `Annotated` / `Literal` aren't imported at the **module level** where your `@tool` function is defined (e.g., imported inside a test function body), `get_type_hints()` silently fails and all metadata drops. You'll get a loud warning log — watch for `Tool schema generation could not resolve type hints for ...` in your logs.

### PTC Extra Args

Sometimes you want to surface structured inputs directly on the outer `__exo_ptc__` tool call — metadata the LLM should provide alongside the code, or values the PTC code should read. `ptc_extra_args` adds optional string fields to the `__exo_ptc__` schema itself:

```python
agent = Agent(
    name="planner",
    tools=[...],
    ptc=True,
    ptc_extra_args={
        "intent": "Brief description of what the PTC code is trying to do.",
        "phase_id": "The current plan phase ID this invocation belongs to.",
        "confidence": "Your confidence level: 'low', 'medium', or 'high'.",
    },
)
```

**What changes:**

1. **Outer schema**: `__exo_ptc__.parameters.properties` now includes `code` (required) + `intent` + `phase_id` + `confidence` (all optional strings).
2. **Description**: a `<ptc_extra_args>` block appears in the `__exo_ptc__` description explaining the fields and referencing the `ptc_args` dict.
3. **LLM call**: the LLM invokes `__exo_ptc__(code="...", intent="batch search Q3 data", confidence="high")`.
4. **PTC code namespace**: a `ptc_args` dict is pre-populated with the values the LLM filled:

```python
# Inside the code= payload the LLM writes:
print(ptc_args['intent'])                    # "batch search Q3 data"
threshold = 0.9 if ptc_args.get('confidence') == 'high' else 0.5
regions = json.loads(await default_api.get_regions(phase=ptc_args['phase_id']))
for r in regions:
    data = await default_api.search(query=ptc_args['intent'], region=r)
    ...
```

**Semantics:**

- **Unknown keys are filtered.** If the LLM hallucinates a field not in `ptc_extra_args`, it never reaches `ptc_args` — no silent contamination.
- **Per-run isolation.** Each PTC invocation gets a fresh `ptc_args` dict. Mutations never bleed across runs.
- **NOT propagated to inner tools.** `ptc_args['intent']` does NOT auto-append to `default_api.foo(...)` calls. If you want to forward values to an inner tool, pass them explicitly.
- **NOT the same as `injected_tool_args`.** Those are agent-level schema decorations visible on *every* tool schema (inner and outer), stripped before dispatch so the tool function never sees them. `ptc_extra_args` are specific to `__exo_ptc__`, **exposed** rather than stripped, and reachable via the `ptc_args` dict inside user code.
- **Propagation.** `ptc_extra_args` is inherited by `spawn_self` children and by the ephemeral planner agent built inside `planning_enabled=True` flows, and survives `Agent.to_dict()` / `from_dict()` round-trips. Also forwardable via the YAML `Agent` loader.

**When to use which:**

| Need | Use |
|---|---|
| Metadata the LLM should be aware of on every tool call (observability hint) | `injected_tool_args` |
| Structured input the PTC **code** should actually read and use | `ptc_extra_args` |
| Values auto-injected from runtime context (e.g., user_id from a request handler) | Neither — use a `ToolContext` parameter on the tool function |

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
- **Full tool metadata survives into PTC** — `Literal`, `Annotated[T, "description"]`, full multi-line docstrings, default values, and injected_tool_args all render into the PTC description. See the **Rich Tool Signatures** section above. Nothing gets dropped or truncated.
- **`from __future__ import annotations` + missing typing imports** — turns type hints into strings that `get_type_hints()` tries to resolve in the function's module globals. If `Annotated` / `Literal` aren't imported at the module level where a `@tool` function is defined, resolution fails silently and **all** rich type metadata drops (every param falls back to `str` with no description or enum). Exo logs a loud warning (`Tool schema generation could not resolve type hints for ...`) so the footgun is visible. Fix: add the typing imports at module level.
- **`ptc_extra_args` are exposed, not stripped** — unlike `injected_tool_args` (stripped before tool dispatch), values the LLM fills for `ptc_extra_args` end up in a `ptc_args` dict inside the PTC code namespace. Use them when the code needs to **read** structured input the LLM provides alongside `code`. Unknown keys the LLM hallucinates are silently filtered so only declared fields reach `ptc_args`.
- **LLM sometimes emits direct `default_api.tool_name(...)` calls** — some models misread the PTC description and generate a tool_call with a dotted name instead of wrapping it in `__exo_ptc__(code="...")`. The runner normalizes these at parse time: if the stripped name matches a registered tool, the call is rewritten to the bare name and dispatched normally (events and dispatch both see the clean name). A warning is logged so the user can investigate why the model bypassed PTC.
- **Swarm/spawn_self/planner propagate PTC correctly** — `Swarm(ptc=True)` uses `Agent._apply_ptc_setting` to register `__exo_ptc__` on each member and invalidate their schema caches. `spawn_self` children inherit the parent's `ptc`/`ptc_timeout`/`ptc_max_output_bytes`/`ptc_max_tool_calls`/`ptc_extra_args` and exclude the parent's `__exo_ptc__` from their tool list. The ephemeral planner agent (planning pre-pass) does the same.
- **Handoffs invalidate the PTC schema cache** — `add_handoff` / `_register_handoff` invalidate `_cached_tool_schemas`. The PTC filter uses handoff names to exclude matching tools, so a runtime handoff must trigger a cache rebuild.
- **`HumanInputTool` is excluded from PTC** — has `_ptc_exclude=True` so the interactive prompt stays as a direct schema and isn't buffered inside a PTC code block.
- **`_DelegateTool` (Swarm team mode) is excluded from PTC** — has `_ptc_exclude=True` so routing-to-worker calls stay as direct schemas and the cache is invalidated correctly when delegates are injected.
- **Task-loop control tools are excluded from PTC** — `_QueueTool` (`steer_agent`, `abort_agent`) has `_ptc_exclude=True` so control-flow signals fire immediately and aren't delayed by PTC code execution.
