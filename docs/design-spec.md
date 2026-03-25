# Exo Design Specification

## 1. Target API

Exo's API draws from the best of **OpenAI Agents SDK** (clean Runner pattern, first-class handoffs), **Google ADK** (explicit workflow primitives), and **Pydantic AI** (dependency injection, model string convention). It rejects CrewAI's role-playing metaphor and excessive configuration.

### 1.1 Agent Definition

```python
from exo import Agent

# Minimal — just a model and instructions
agent = Agent(
    name="assistant",
    model="openai:gpt-4o",
    instructions="You are a helpful assistant.",
)

# Full-featured
agent = Agent(
    name="researcher",
    model="anthropic:claude-sonnet-4-20250514",
    instructions="You research topics thoroughly and cite sources.",
    tools=[search_web, read_url],
    handoffs=[writer_agent],
    hooks=[my_hook],
    output_type=ResearchReport,      # Pydantic model for structured output
    max_steps=20,
    temperature=0.7,
)
```

**Constructor parameters** (all keyword, only `name` required):

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Unique identifier |
| `model` | `str` | `"openai:gpt-4o"` | `"provider:model_name"` format |
| `instructions` | `str \| Callable` | `""` | System prompt; callable receives `RunContext` |
| `tools` | `list[Tool]` | `[]` | `@tool`-decorated functions or `Tool` instances |
| `handoffs` | `list[Agent]` | `[]` | Agents this agent can delegate to |
| `hooks` | `list[Hook]` | `[]` | Lifecycle hooks |
| `output_type` | `type[BaseModel] \| None` | `None` | Structured output schema |
| `max_steps` | `int` | `10` | Max LLM↔tool round-trips |
| `temperature` | `float` | `1.0` | LLM temperature |
| `max_tokens` | `int \| None` | `None` | Max output tokens per LLM call |
| `memory` | `Memory \| None` | `None` | Memory backend |
| `context` | `Context \| None` | `None` | Context engine for prompt building, state, RAG |

### 1.2 Tool Registration

**Primary: `@tool` decorator** (module-level, like OpenAI Agents SDK):

```python
from exo import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: The city name.
        unit: Temperature unit — "celsius" or "fahrenheit".
    """
    return f"22°C and sunny in {city}"

@tool
async def search_web(query: str) -> list[str]:
    """Search the web and return relevant URLs."""
    async with httpx.AsyncClient() as client:
        ...
```

- Sync functions are automatically wrapped in `asyncio.to_thread()`
- Schema is generated from type hints + docstring
- `Args:` section in docstring becomes parameter descriptions
- `@tool` with no arguments is the standard form; `@tool(name="override")` for customization
- Uses `@overload` so pyright knows `@tool` returns `FunctionTool` and `@tool()` returns a decorator

**Secondary: `Tool` base class** (for complex tools):

```python
from exo import Tool

class DatabaseQuery(Tool):
    name = "query_database"
    description = "Execute a SQL query against the database."

    async def execute(self, sql: str, params: dict | None = None) -> str:
        ...
```

### 1.3 Running Agents

```python
from exo import Agent, run

agent = Agent(name="assistant", model="openai:gpt-4o", instructions="Be helpful.")

# Async (primary)
result = await run(agent, "What's the weather in Tokyo?")
print(result.output)          # final text output
print(result.messages)        # full conversation history
print(result.usage)           # token usage

# Sync wrapper
result = run.sync(agent, "What's the weather in Tokyo?")

# Streaming
async for event in run.stream(agent, "Tell me a story"):
    if event.type == "text":
        print(event.text, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[calling {event.tool_name}...]")

# Multi-turn conversation
result = await run(agent, "What's the weather?")
result = await run(agent, "What about tomorrow?", messages=result.messages)
```

### 1.4 Multi-Agent Orchestration

**Workflow** (sequential/parallel DAG — like Google ADK's SequentialAgent):

```python
from exo import Agent, Swarm, run

researcher = Agent(name="researcher", ...)
writer = Agent(name="writer", ...)
editor = Agent(name="editor", ...)

# Sequential pipeline — output of each feeds into the next
pipeline = Swarm(
    agents=[researcher, writer, editor],
    flow="researcher >> writer >> editor",
)
result = await run(pipeline, "Write an article about quantum computing")
```

**Handoff** (agent-driven delegation — like OpenAI Agents SDK):

```python
billing = Agent(name="billing", instructions="Handle billing questions.")
support = Agent(name="support", instructions="Handle support requests.")

triage = Agent(
    name="triage",
    instructions="Route to the right department.",
    handoffs=[billing, support],
)
result = await run(triage, "I need a refund")
# triage calls transfer_to_billing → billing handles it
```

**Team** (leader + workers — like Google ADK with sub_agents):

```python
lead = Agent(name="lead", instructions="Coordinate the research team.")
analyst = Agent(name="analyst", instructions="Analyze data.")
writer = Agent(name="writer", instructions="Write reports.")

team = Swarm(
    agents=[lead, analyst, writer],
    mode="team",              # lead is first agent, others are workers
)
result = await run(team, "Research and report on market trends")
```

### 1.5 Context Engine

The context engine (rewritten clean from AWorld's AMNI system) is Exo's "digital brain" for agent execution. It manages runtime state, composable prompt building, hierarchical task decomposition, knowledge retrieval (RAG), workspace artifacts, token tracking, and checkpoint/restore. Unlike simple context managers in other frameworks, Exo's context is an **active system** that optimizes what goes into the LLM context window.

**Core design principles:**
- Context isn't just a container — it actively manages execution state, memory, and LLM interaction
- Hierarchical state with parent inheritance enables efficient multi-agent isolation
- Composable prompt building via neurons (modular prompt components)
- Event-driven processor pipeline for pluggable LLM intervention
- Workspace as external memory with RAG integration

#### 1.5.1 Context Creation & Configuration

```python
from exo.context import Context, ContextConfig

# Configure context automation level
config = ContextConfig(
    mode="copilot",               # "pilot" | "copilot" | "navigator"
    history_rounds=20,            # max conversation rounds to keep
    summary_threshold=30,         # trigger summarization after N rounds
    offload_threshold=4000,       # offload tool results > N tokens
    enable_retrieval=True,        # enable RAG for knowledge search
)

# Create context for a task
ctx = Context(task_id="task-1", config=config)
```

**`ContextConfig` fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `mode` | `Literal["pilot", "copilot", "navigator"]` | `"copilot"` | Automation level (see below) |
| `history_rounds` | `int` | `20` | Max conversation rounds in context |
| `summary_threshold` | `int` | `30` | Rounds before triggering summarization |
| `offload_threshold` | `int` | `4000` | Token count to trigger tool result offloading |
| `enable_retrieval` | `bool` | `False` | Enable RAG for knowledge search |
| `neuron_names` | `list[str]` | `[]` | Which prompt neurons to activate |

**Automation levels:**
- **`"pilot"`** — Manual. No automatic summarization or offloading. Basic prompt augmentation only.
- **`"copilot"`** — Auto-summarize long conversations, offload large tool results, history windowing.
- **`"navigator"`** — Full automation. Copilot features plus automated knowledge recall, context optimization, and cognitive processing.

#### 1.5.2 Hierarchical State (Parent-Child Contexts)

Context supports hierarchical state with parent inheritance — child contexts can read parent state but writes are isolated. This enables efficient multi-agent isolation without deep copying.

```python
# Fork a child context for a subtask
child_ctx = ctx.fork(task_id="subtask-1")

# Child reads parent state transparently
parent_val = child_ctx.state.get("shared_key")  # falls through to parent

# Child writes only affect local state
child_ctx.state.set("progress", "researching")  # local only

# Merge child results back into parent (with net token calculation)
ctx.merge(child_ctx)
```

**`ContextState`** — hierarchical key-value store:
- `get(key, default)` — Search local state first, then parent chain
- `set(key, value)` — Write to local state only
- `local_dict()` — Get only local mutations (excluding inherited state)
- `to_dict()` — Get full flattened view (local + parent)

#### 1.5.3 Composable Prompt Building (Neurons)

Neurons are modular prompt components that build rich system prompts from context. Each neuron produces a prompt fragment; they compose in priority order.

```python
from exo.context import PromptBuilder

builder = PromptBuilder(ctx)
builder.add("system", "You are a helpful assistant.")
builder.add("task")           # task_id, task_input, origin_user_input
builder.add("history")        # conversation history (windowed)
builder.add("knowledge", query="quantum computing")  # RAG retrieval
builder.add("workspace")      # workspace file listing
builder.add("todo")           # current task plan/checklist
builder.add("skills")         # available agent skills
prompt = await builder.build()
```

**Built-in neurons** (priority-ordered):

| Neuron | Priority | Provides |
|---|---|---|
| `task` | 1 | Task ID, input, output, subtask plan |
| `todo` | 2 | Task planning checklist |
| `history` | 10 | Conversation history (windowed) |
| `knowledge` | 20 | RAG-retrieved knowledge chunks |
| `workspace` | 30 | Workspace file listing |
| `skills` | 40 | Active agent skills/capabilities |
| `facts` | 50 | Long-term facts from memory |
| `entities` | 60 | Extracted named entities |
| `system` | 100 | Dynamic variables (date, time, platform) |

Custom neurons:

```python
from exo.context import Neuron, neuron_registry

@neuron_registry.register("custom_domain")
class DomainNeuron(Neuron):
    priority = 25

    async def format(self, ctx: Context) -> str:
        data = ctx.state.get("domain_data")
        return f"Domain context:\n{data}"
```

#### 1.5.4 Context Processors (Event-Driven Pipeline)

Processors intervene at specific points in the LLM execution cycle. They transform context before/after LLM calls and tool execution.

```python
from exo.context import ContextProcessor

class ToolResultOffloader(ContextProcessor):
    """Offload large tool results to workspace to save context window."""

    event = "post_tool_call"

    async def process(self, ctx: Context, payload: dict) -> None:
        result = payload["tool_result"]
        if token_count(result) > ctx.config.offload_threshold:
            artifact_id = await ctx.workspace.write_artifact(result)
            payload["tool_result"] = f"[Result stored as artifact: {artifact_id}]"

class PromptAugmenter(ContextProcessor):
    """Augment system prompt with context-aware neurons."""

    event = "pre_llm_call"

    async def process(self, ctx: Context, payload: dict) -> None:
        builder = PromptBuilder(ctx)
        for neuron_name in ctx.config.neuron_names:
            builder.add(neuron_name)
        augmented = await builder.build()
        payload["system_prompt"] = augmented

# Register processors
ctx.add_processor(ToolResultOffloader())
ctx.add_processor(PromptAugmenter())
```

**Processor events:**
- `pre_llm_call` — Before LLM call (augment prompt, summarize history)
- `post_llm_call` — After LLM response (extract entities, save memory)
- `pre_tool_call` — Before tool execution
- `post_tool_call` — After tool execution (offload large results)

#### 1.5.5 Knowledge & RAG

Context integrates with a retrieval pipeline for semantic search over indexed artifacts.

```python
# Add knowledge artifacts
await ctx.knowledge.add("report.pdf", content=pdf_text)
await ctx.knowledge.add("research.md", content=markdown_text)

# Semantic search
results = await ctx.knowledge.search("quantum entanglement", top_k=5)
for chunk in results:
    print(chunk.content, chunk.score)

# Retrieve specific artifact
content = await ctx.knowledge.get("report.pdf")
# Range query for large documents
section = await ctx.knowledge.get_range("report.pdf", start=0, end=50)
```

**RAG pipeline:**
```
Artifact → Chunker → ChunkStore → Index (Vector + FullText) → Search → Reranker → Results
```

#### 1.5.6 Workspace (Artifact Storage)

Workspace provides persistent artifact storage during execution — external memory that agents can read/write. Large tool results are offloaded here to keep the LLM context window clean.

```python
# Write artifacts to workspace
await ctx.workspace.write("output.md", content=report)
await ctx.workspace.write("data.csv", content=csv_data)

# List workspace files
files = await ctx.workspace.list()

# Read artifacts
content = await ctx.workspace.read("output.md")
```

#### 1.5.7 Token Tracking

Context tracks token usage at step granularity per agent — enabling cost analysis, budget enforcement, and trajectory replay.

```python
# Token usage is tracked automatically during execution
print(ctx.token_usage)
# {"prompt_tokens": 1200, "completion_tokens": 450, "total_tokens": 1650}

# Per-agent, per-step tracking
trajectory = ctx.get_trajectory("agent-1")
for step in trajectory.steps:
    print(f"Step {step.step}: {step.prompt_tokens} prompt, {step.output_tokens} output")
```

#### 1.5.8 Checkpoint & Restore

Save and restore complete execution state for long-running tasks.

```python
# Save checkpoint
checkpoint = await ctx.snapshot()

# Restore from checkpoint
restored_ctx = await Context.restore(checkpoint)

# Checkpoints are versioned per session
checkpoints = await ctx.list_checkpoints()
```

#### 1.5.9 Context Tools

Special tools that let agents manipulate their own context during execution.

```python
from exo.context.tools import planning_tool, knowledge_tool

agent = Agent(
    name="researcher",
    tools=[search_web, planning_tool, knowledge_tool],
    context=ctx,
)

# Agent can now:
# - planning_tool: add_todo("Step 1: Search papers"), get_todo()
# - knowledge_tool: get_knowledge("report.pdf"), grep_knowledge("report.pdf", "quantum")
```

#### 1.5.10 Integration with Agent

```python
from exo import Agent, run
from exo.context import Context, ContextConfig

ctx = Context(
    task_id="research-task",
    config=ContextConfig(mode="copilot"),
)

agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="You research topics thoroughly.",
    tools=[search_web],
    context=ctx,                   # attach context to agent
)

result = await run(agent, "Research quantum computing")
# Context automatically manages:
# - Prompt augmentation via neurons
# - History windowing and summarization
# - Tool result offloading
# - Token tracking
# - Checkpoint saving
```

**Constructor parameters for `Context`:**

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `task_id` | `str` | required | Unique task identifier |
| `config` | `ContextConfig` | `ContextConfig()` | Automation and threshold settings |
| `parent` | `Context \| None` | `None` | Parent context for hierarchical tasks |

### 1.6 Full Import Surface

```python
# Core — everything most users need
from exo import Agent, Swarm, Tool, tool, run
from exo.types import (
    UserMessage, AssistantMessage, SystemMessage,
    ToolCall, ToolResult, Message,
    RunResult, StreamEvent, TextEvent, ToolCallEvent,
)
from exo.config import AgentConfig, ModelConfig
from exo.registry import Registry, agent_registry, tool_registry
from exo.events import EventBus
from exo.hooks import Hook, HookPoint

# Models
from exo.models import ModelResponse, StreamChunk, ModelError, FinishReason
from exo.models import ModelProvider, get_provider

# Context engine
from exo.context import Context, ContextConfig, PromptBuilder, ContextProcessor

# Memory
from exo.memory import ShortTermMemory, LongTermMemory

# MCP
from exo.mcp import mcp_tools, MCPClient

# Sandbox
from exo.sandbox import Sandbox, LocalSandbox

# Trace
from exo.trace import traced, trace_span, TraceConfig

# Eval
from exo.eval import Evaluator, Scorer, EvalResult

# A2A
from exo.a2a import A2AServer, A2AClient
```

---

## 2. Architectural Patterns

### 2.1 Execution Flow

```
User code                    exo internals
─────────                    ─────────────────
run(agent, input)
  │
  ├─► wrap in Swarm if bare Agent
  ├─► create RunState
  ├─► call_runner(swarm, input, state)
  │     │
  │     ├─► for each agent in execution order:
  │     │     ├─► context processors: pre_llm_call (summarize, augment)
  │     │     ├─► message_builder: build LLM messages (via PromptBuilder if context attached)
  │     │     ├─► hooks: PRE_LLM_CALL
  │     │     ├─► provider.complete(messages, tools)
  │     │     ├─► hooks: POST_LLM_CALL
  │     │     ├─► output_parser: parse response
  │     │     │
  │     │     ├─► if tool_calls:
  │     │     │     ├─► hooks: PRE_TOOL_CALL
  │     │     │     ├─► execute tools (parallel if multiple)
  │     │     │     ├─► hooks: POST_TOOL_CALL
  │     │     │     ├─► context processors: post_tool_call (offload large results)
  │     │     │     └─► loop back to message_builder
  │     │     │
  │     │     ├─► if handoff:
  │     │     │     └─► switch to target agent, continue
  │     │     │
  │     │     └─► if text response (no tool calls):
  │     │           └─► agent done, pass to next in swarm
  │     │
  │     └─► return RunResult
  │
  └─► return RunResult
```

### 2.2 Module Dependency Graph

```
exo-core (zero external deps except pydantic)
  ├── types.py          → no internal deps
  ├── config.py         → types
  ├── registry.py       → types (ExoError)
  ├── events.py         → no internal deps
  ├── hooks.py          → types
  ├── tool.py           → types, registry
  ├── agent.py          → types, config, tool, hooks
  ├── swarm.py          → agent, _internal/graph
  ├── runner.py         → agent, swarm, _internal/call_runner
  └── _internal/
      ├── message_builder.py → types
      ├── output_parser.py   → types
      ├── call_runner.py     → agent, state
      ├── state.py           → types
      └── graph.py           → no internal deps

exo-models  → exo-core (types, config)
  ├── types.py          → exo-core types (ToolCall, Usage, ExoError)
  ├── provider.py       → types (ModelResponse, StreamChunk)
  ├── openai.py         → provider, types
  └── anthropic.py      → provider, types
exo-context → exo-core (types, hooks, events), exo-memory (storage backends)
  ├── context.py        → types, config
  ├── config.py         → no internal deps
  ├── prompt_builder.py → context, types
  ├── processor.py      → context, events
  ├── state.py          → types
  ├── workspace.py      → no internal deps
  └── checkpoint.py     → context, state
exo-memory  → exo-core (types)
exo-mcp     → exo-core (tool)
exo-sandbox → exo-core (tool)
exo-trace   → exo-core (hooks)
exo-eval    → exo-core (types)
exo-a2a     → exo-core (agent, runner)
exo-cli     → exo-core, exo-models
exo-server  → exo-core, exo-models
exo-train   → exo-core, exo-models
```

### 2.3 Key Simplifications Over AWorld

| AWorld Pattern | Exo Pattern | Why |
|---|---|---|
| `Message[DataType]` with 15 fields, stringly-typed `category`/`topic` routing | Typed message classes (`UserMessage`, `AssistantMessage`, etc.) as simple Pydantic models | Type safety, no routing bugs |
| `ConfigDict(dict)` + `BaseConfig` + Pydantic models mixed | Pydantic v2 models only | One config system, not three |
| `Factory[T]` + `AgentManager` + `ToolsManager` (3 layers) | Single `Registry[T]` class | One pattern, no subclass chain |
| `BaseTool` returning gym 5-tuple `(obs, reward, term, trunc, info)` | `Tool.execute(**kwargs) -> str \| dict` | Tools return results, not gym observations |
| `BaseTool` + `AsyncBaseTool` duplication | Single async `Tool` class, sync functions auto-wrapped | Async-first, no duplication |
| `ToolActionExecutor` + `ActionFactory` double dispatch | Direct `tool.execute()` call | Remove indirection |
| `LLMAgent.__init__` with 20+ params | `Agent.__init__` with ~10 params, rest in config | Clean constructor |
| `async_messages_transform()` (140 lines of tool_call reordering) | `message_builder.build()` (~40 lines, correct by construction) | Build messages right the first time |
| `_agent_result()` with GroupMessage routing | Direct return of tool calls or text | No message wrapping inside agent |
| 6 runner entry points (`run`, `run_task`, `sync_run`, `streaming_run`, `streaming_run_task`, `streamed_run_task`) | 3 entry points: `run()`, `run.sync()`, `run.stream()` | Clear API |
| `Swarm` with 3 builder classes (~400 lines) | `Swarm` with `flow=` DSL string or `mode=` enum | Declarative, not builder chain |
| AMNI `ApplicationContext` with 7 lazy services, 12 neuron types, factory chains | `Context` + `PromptBuilder` + `ContextProcessor` (3 composable classes) | Simpler composition, same power |
| `AmniConfigFactory` with PILOT/COPILOT/NAVIGATOR class hierarchy | `ContextConfig(mode="copilot")` — single Pydantic model | Config, not class hierarchy |
| Neurons as factory-registered classes with separate binding layer | `PromptBuilder.add(type, **kwargs)` — method calls, not class instantiation | Direct API, no factory |

---

## 3. Code Quality Standards

### 3.1 Style Rules

- **Line length:** 100 characters (ruff enforced)
- **Quotes:** Double quotes for strings
- **Imports:** Sorted by ruff (`isort` rules), stdlib → third-party → local
- **Type hints:** Required on all public functions and class attributes. Use `X | Y` union syntax (not `Union[X, Y]`). Use `list`, `dict`, `tuple` lowercase (not `List`, `Dict`, `Tuple`).
- **Docstrings:** Google style. Required on all public classes and functions. Not required on private/internal helpers or test functions.

```python
# Good
def build_messages(
    instructions: str,
    history: list[Message],
    tool_results: list[ToolResult] | None = None,
) -> list[Message]:
    """Build the message list for an LLM call.

    Constructs a correctly ordered message sequence from system instructions,
    conversation history, and any pending tool results.

    Args:
        instructions: The system prompt.
        history: Previous conversation messages.
        tool_results: Results from tool calls to include.

    Returns:
        Ordered list of messages ready for the LLM provider.
    """
```

### 3.2 Pydantic Model Conventions

```python
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for an Agent."""

    model_config = {"frozen": True}  # immutable after creation

    name: str
    model: str = "openai:gpt-4o"
    instructions: str = ""
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = None
    max_steps: int = Field(default=10, ge=1)
```

Rules:
- Use `model_config = {"frozen": True}` for config/data classes (immutable)
- Use `Field()` for validation constraints, defaults with metadata
- Plain defaults for simple values
- No `@validator` — use `@field_validator` (Pydantic v2)
- All models inherit `BaseModel` directly — no `BaseConfig` intermediary

### 3.3 Async Patterns

```python
# Good — async-first, sync wrapper at boundary
async def run(agent: Agent, input: str) -> RunResult:
    ...

# The ONLY place we bridge sync→async
def _sync_run(agent: Agent, input: str) -> RunResult:
    return asyncio.run(run(agent, input))

# Attach as attribute for clean API
run.sync = _sync_run
```

Rules:
- All internal functions are `async def`
- Sync functions wrapped via `asyncio.to_thread()` when called from async context
- Only ONE sync entry point: `run.sync()` — everything else is async
- Use `asyncio.TaskGroup` for parallel tool execution (Python 3.11+)
- Never use `loop.run_until_complete()` — always `asyncio.run()`

### 3.4 Error Handling

```python
# Good — specific exceptions with context
class ToolExecutionError(ExoError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, cause: Exception):
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' failed: {cause}")

# Good — catch specific, re-raise with context
try:
    result = await tool.execute(**args)
except Exception as e:
    raise ToolExecutionError(tool.name, e) from e
```

Rules:
- All Exo exceptions inherit from `ExoError` (defined in `exo/types.py`)
- Include the thing that failed (tool name, agent name, provider) in the message
- Use `from e` for exception chaining
- Never silently swallow exceptions — log or re-raise
- Tool execution errors are caught and returned as `ToolResult(error=...)`, not propagated

### 3.5 Testing Patterns

```python
import pytest
from exo import Agent, tool, run

# Test file naming: test_<module>.py
# Test function naming: test_<what>_<scenario>

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@pytest.fixture
def mock_provider():
    """A provider that returns canned responses."""
    ...

async def test_agent_calls_tool_and_returns_result(mock_provider):
    agent = Agent(name="calc", tools=[add])
    result = await run(agent, "What is 2 + 3?", _provider=mock_provider)
    assert "5" in result.output

async def test_agent_respects_max_steps(mock_provider):
    agent = Agent(name="looper", tools=[add], max_steps=2)
    result = await run(agent, "Keep adding", _provider=mock_provider)
    assert result.steps <= 2
```

Rules:
- Every module has a corresponding `test_<module>.py`
- Tests use `async def` (pytest-asyncio with `asyncio_mode = "auto"`)
- Mock LLM providers — never make real API calls in tests
- Test the public API surface, not internal methods
- Use `@pytest.fixture` for shared setup
- Aim for ~1 test file per source file, ~5-15 tests per file
- Test file names must be unique across all packages (e.g., `test_model_types.py` not `test_types.py` if exo-core already has one). Pytest uses `--import-mode=importlib` to handle multiple `tests/` packages.
- For cross-namespace-package imports in test files, add `# pyright: ignore[reportMissingImports]` on the import line (pyright can't resolve `.pth`-based editable installs)

### 3.6 File Size Guidelines

- **Max ~200 lines** per source file (not counting tests)
- If a file grows beyond 200 lines, split into `_internal/` submodules
- `__init__.py` files are for re-exports only — no logic
- Test files can be longer (up to ~300 lines)

---

## 4. Structural Conventions

### 4.1 Package Layout

```
packages/exo-core/
├── pyproject.toml
├── src/
│   └── exo/
│       ├── __init__.py          # re-exports: Agent, Swarm, Tool, tool, run
│       ├── types.py             # Message types, RunResult, StreamEvent
│       ├── config.py            # AgentConfig, ModelConfig, TaskConfig
│       ├── registry.py          # Registry[T], agent_registry, tool_registry
│       ├── events.py            # EventBus
│       ├── hooks.py             # Hook, HookPoint, HookManager
│       ├── tool.py              # Tool base, @tool decorator, FunctionTool
│       ├── agent.py             # Agent class
│       ├── swarm.py             # Swarm class
│       ├── runner.py            # run(), run.sync(), run.stream()
│       └── _internal/           # not part of public API
│           ├── __init__.py
│           ├── message_builder.py
│           ├── output_parser.py
│           ├── call_runner.py
│           ├── state.py
│           └── graph.py
└── tests/
    ├── __init__.py
    ├── test_types.py
    ├── test_config.py
    ├── test_registry.py
    ├── test_events.py
    ├── test_hooks.py
    ├── test_tool.py
    ├── test_agent.py
    ├── test_swarm.py
    └── test_runner.py
```

```

```
packages/exo-context/
├── pyproject.toml
├── src/
│   └── exo/
│       └── context/
│           ├── __init__.py          # re-exports: Context, ContextConfig, PromptBuilder, etc.
│           ├── context.py           # Context class — runtime execution context
│           ├── config.py            # ContextConfig — mode, thresholds, automation settings
│           ├── state.py             # ContextState — hierarchical key-value state with parent inheritance
│           ├── prompt_builder.py    # PromptBuilder — composable prompt components
│           ├── neuron.py            # Neuron ABC, neuron_registry, built-in neurons
│           ├── processor.py         # ContextProcessor ABC, processor pipeline
│           ├── workspace.py         # Workspace — artifact storage during execution
│           ├── checkpoint.py        # Checkpoint — save/restore execution state
│           ├── token_tracker.py     # TokenTracker — per-agent, per-step token tracking
│           ├── tools.py             # Context tools (planning, knowledge, file, skill)
│           └── _internal/
│               ├── __init__.py
│               ├── knowledge.py     # KnowledgeStore — RAG indexing and retrieval
│               ├── retriever.py     # Retriever pipeline (chunking, indexing, reranking)
│               └── summarizer.py    # Context summarization and compression
└── tests/
    ├── __init__.py
    ├── test_context.py
    ├── test_context_config.py
    ├── test_context_state.py
    ├── test_prompt_builder.py
    ├── test_neuron.py
    ├── test_context_processor.py
    ├── test_workspace.py
    ├── test_checkpoint.py
    ├── test_token_tracker.py
    └── test_context_tools.py
```

```
packages/exo-models/
├── pyproject.toml
├── src/
│   └── exo/
│       └── models/
│           ├── __init__.py          # re-exports: ModelError, ModelResponse, StreamChunk, etc.
│           ├── types.py             # ModelError, FinishReason, ModelResponse, ToolCallDelta, StreamChunk
│           ├── provider.py          # Model ABC, model_registry (future)
│           ├── openai.py            # OpenAI provider (future)
│           └── anthropic.py         # Anthropic provider (future)
└── tests/
    ├── __init__.py
    └── test_model_types.py
```

### 4.2 `__init__.py` Pattern

```python
"""Exo Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Public API — these are what users import from `exo`
from exo.agent import Agent
from exo.runner import run
from exo.swarm import Swarm
from exo.tool import Tool, tool
```

Rules:
- `__init__.py` is the **public API surface** — only export what users need
- Use `__all__` if the export list is ambiguous
- Never put logic in `__init__.py`
- Subpackages (`exo.models`, `exo.memory`, etc.) follow the same pattern

### 4.3 Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Modules | `snake_case.py` | `message_builder.py` |
| Classes | `PascalCase` | `Agent`, `ModelProvider`, `RunResult` |
| Functions | `snake_case` | `run()`, `get_provider()`, `build_messages()` |
| Constants | `UPPER_SNAKE` | `DEFAULT_MAX_STEPS`, `HookPoint.PRE_LLM_CALL` |
| Type aliases | `PascalCase` | `Message = UserMessage \| AssistantMessage \| ...` |
| Private/internal | `_prefix` | `_internal/`, `_sync_run()`, `_parse_tool_calls()` |
| Test functions | `test_<what>_<scenario>` | `test_agent_calls_tool_and_returns_result` |

### 4.4 Model String Convention

Provider and model are specified as a single string: `"provider:model_name"`.

```python
"openai:gpt-4o"
"openai:gpt-4o-mini"
"anthropic:claude-sonnet-4-20250514"
"anthropic:claude-haiku-3-20240307"
```

Parsed by `parse_model_string()` in `exo.config` (and by `get_provider()` in `exo-models`). If no prefix, defaults to `"openai"`.

### 4.5 Dependency Rules

1. **`exo-core` has ZERO heavy dependencies** — only `pydantic`. No `openai`, no `anthropic`, no `httpx`.
2. Provider SDKs live in `exo-models` only.
3. Optional heavy deps (chromadb, kubernetes, etc.) are declared as extras.
4. Internal packages depend on `exo-core` but NOT on each other (no `exo-models` → `exo-memory` dep).
5. `_internal/` modules are never imported from outside their package.

---

## 5. Anti-Patterns (Things We Explicitly Avoid)

### From AWorld (things we're fixing):

| Anti-Pattern | What We Do Instead |
|---|---|
| `ConfigDict(dict)` with attribute access magic | Pydantic models with proper typing |
| Stringly-typed message routing (`category="tool"`, `topic="GROUP_RESULTS"`) | Typed message classes and match statements |
| Gym-style 5-tuple returns from tools | `Tool.execute() -> str \| dict` |
| `async_messages_transform()` — 140-line message reordering | Build messages correctly from the start in `message_builder` |
| Memory coupled inside agent's policy method | Memory as a hook or explicit middleware |
| 6 different runner entry points | 3: `run()`, `run.sync()`, `run.stream()` |
| Sync/async class duplication (`BaseTool` + `AsyncBaseTool`) | Single async class, sync auto-wrapped |
| Auto-hoisting kwargs between config levels | Explicit config construction |

### General anti-patterns to avoid:

- **God classes.** No class >200 lines. Split into composable functions.
- **Stringly-typed dispatch.** Use enums, typed unions, or protocols.
- **Deep inheritance.** Max 2 levels. Prefer composition.
- **Catch-all dicts.** No `headers: dict`, `info: dict`, `metadata: dict` bags. Define the fields.
- **Implicit singletons.** No module-level mutable state except registries. Pass dependencies explicitly.
- **Optional everything.** If a field is always present at runtime, don't make it `Optional`.
- **Magic `__init__`.** Constructors should be predictable — no env var reading, no config merging, no side effects.
