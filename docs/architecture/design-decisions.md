# Design Decisions

This document explains the key architectural choices in Exo and why they differ from AWorld and other frameworks.

## Key Simplifications from AWorld

Exo is a ground-up rewrite that simplifies AWorld's patterns while preserving its capabilities. The table below summarizes the major changes:

| AWorld Pattern | Exo Pattern | Rationale |
|---|---|---|
| `Message[DataType]` with 15 fields, stringly-typed `category`/`topic` routing | Typed message classes (`UserMessage`, `AssistantMessage`, `ToolResult`, etc.) as simple Pydantic models | Type safety eliminates routing bugs at compile time |
| `ConfigDict(dict)` + `BaseConfig` + Pydantic models mixed | Pydantic v2 models only (`AgentConfig`, `ModelConfig`, etc.) | One config system instead of three |
| `Factory[T]` + `AgentManager` + `ToolsManager` (3 layers) | Single `Registry[T]` class (~88 lines) | One generic pattern replaces three specialized managers |
| `BaseTool` returning gym 5-tuple `(obs, reward, term, trunc, info)` | `Tool.execute(**kwargs) -> str | dict` | Tools return results, not gym observations |
| `BaseTool` + `AsyncBaseTool` duplication | Single async `Tool` class, sync functions auto-wrapped via `asyncio.to_thread()` | Async-first, no class duplication |
| `ToolActionExecutor` + `ActionFactory` double dispatch | Direct `tool.execute()` call | Remove indirection |
| `LLMAgent.__init__` with 20+ params | `Agent.__init__` with ~10 params, rest in config objects | Clean, predictable constructor |
| `async_messages_transform()` (140 lines of tool_call reordering) | `message_builder.build()` (~40 lines, correct by construction) | Build messages right the first time instead of reordering after |
| `_agent_result()` with GroupMessage routing | Direct return of tool calls or text | No internal message wrapping |
| 6 runner entry points | 3 entry points: `run()`, `run.sync()`, `run.stream()` | Clear, memorable API |
| `Swarm` with 3 builder classes (~400 lines) | `Swarm` with `flow=` DSL string or `mode=` enum | Declarative instead of builder chain |
| AMNI `ApplicationContext` with 7 lazy services, 12 neuron types, factory chains | `Context` + `PromptBuilder` + `ContextProcessor` (3 composable classes) | Simpler composition, same power |
| `AmniConfigFactory` with PILOT/COPILOT/NAVIGATOR class hierarchy | `ContextConfig(mode="copilot")` -- single Pydantic model | Config values instead of class hierarchy |
| Neurons as factory-registered classes with separate binding layer | `PromptBuilder.add(type, **kwargs)` -- method calls | Direct API instead of factory |

## Why a Single Agent Class

AWorld had five agent types: `LLMAgent`, `TaskLLMAgent`, `LoopLLMAgent`, `ParallelLLMAgent`, and `SerialLLMAgent`. Exo collapses these into one `Agent` class because:

1. **The differences were in orchestration, not capability.** A "parallel" agent is really two agents run in parallel -- that is orchestration (Swarm), not an agent property.

2. **Inheritance hierarchies resist composition.** With five agent types, you could not combine "task" behavior with "parallel" behavior without a new subclass.

3. **Configuration replaces subclasses.** The `Agent` class takes `max_steps`, `output_type`, `handoffs`, etc. as constructor parameters. Workflow behavior comes from `Swarm(mode=...)`.

```python
# AWorld: choose the right subclass
from aworld.agents import LLMAgent, TaskLLMAgent, ParallelLLMAgent

# Exo: one class, compose behavior
from exo import Agent, Swarm

agent = Agent(name="a", ...)
pipeline = Swarm(agents=[a, b, c], flow="a >> b >> c")
```

## Why Flow DSL Instead of Builder Pattern

AWorld used builder classes (`SwarmBuilder`, `AgentBuilder`, `SandboxBuilder`) totaling ~400 lines. Exo uses a string-based flow DSL:

```python
# Builder pattern (AWorld)
builder = SwarmBuilder()
builder.add_agent(a)
builder.add_agent(b)
builder.add_edge(a, b)
swarm = builder.build()

# Flow DSL (Exo)
swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")
```

The DSL is:
- **Readable** -- `"a >> b >> c"` is immediately clear
- **Supports parallel groups** -- `"a >> (b | c) >> d"` forks and joins
- **Parsed into a graph** -- Under the hood, `parse_flow_dsl()` produces a `Graph` that gets topologically sorted, so cycles are detected at construction time
- **Serializable** -- A string is trivially saved/loaded from config files

## Why Async-First with Sync Bridge

All internal functions are `async def`. The single sync entry point `run.sync()` calls `asyncio.run()`:

```python
# The ONLY place we bridge sync -> async
def _sync(agent, input, **kwargs):
    return asyncio.run(run(agent, input, **kwargs))

run.sync = _sync
```

This choice:
- **Eliminates sync/async duplication** -- AWorld had `BaseTool` and `AsyncBaseTool`, `run()` and `sync_run()`, etc. Exo has one of each.
- **Enables parallel tool execution** -- `asyncio.TaskGroup` runs multiple tool calls concurrently with zero threading complexity.
- **Wraps sync functions automatically** -- `@tool` detects sync functions and wraps them via `asyncio.to_thread()` so users do not need to think about it.

## Why Typed Messages Instead of Generic Message

AWorld's `Message[DataType]` had 15 fields and used string-based `category`/`topic` routing. This made it easy to route messages to the wrong handler. Exo uses a discriminated union:

```python
Message = UserMessage | AssistantMessage | SystemMessage | ToolResult
```

Each type is a frozen Pydantic model with only the fields that type needs:

```python
class UserMessage(BaseModel):
    model_config = {"frozen": True}
    role: Literal["user"] = "user"
    content: str

class ToolResult(BaseModel):
    model_config = {"frozen": True}
    role: Literal["tool"] = "tool"
    tool_call_id: str
    tool_name: str
    content: str = ""
    error: str | None = None
```

Benefits:
- **Pattern matching** -- `isinstance(msg, ToolResult)` is type-safe and exhaustive
- **No invalid states** -- A `UserMessage` cannot accidentally have `tool_calls`
- **Immutable** -- Frozen models prevent accidental mutation of conversation history

## Anti-Patterns Avoided

These are patterns Exo explicitly rejects:

| Anti-Pattern | What Exo Does Instead |
|---|---|
| God classes (>200 lines) | Split into composable functions; max ~200 lines per source file |
| Stringly-typed dispatch (`category="tool"`) | Enums (`HookPoint`), typed unions (`Message`), protocols |
| Deep inheritance (3+ levels) | Max 2 levels; prefer composition via tools, hooks, processors |
| Catch-all dicts (`headers: dict`, `info: dict`) | Define the fields explicitly as Pydantic models |
| Implicit singletons | No module-level mutable state except registries; pass dependencies explicitly |
| Optional everything | If a field is always present at runtime, it is not `Optional` |
| Magic `__init__` | Constructors are predictable -- no env var reading, no config merging, no side effects |
| Memory coupled inside agent | Memory as a hook or explicit middleware, not embedded in the policy method |
| Auto-hoisting kwargs between config levels | Explicit config construction with typed models |

## Influences from Other Frameworks

| Framework | What Exo Borrowed | What Exo Rejected |
|-----------|----------------------|----------------------|
| **OpenAI Agents SDK** | Clean Runner pattern, first-class handoffs, `@tool` decorator | -- |
| **Google ADK** | Explicit workflow primitives (`SequentialAgent` -> `Swarm(mode="workflow")`) | -- |
| **Pydantic AI** | Dependency injection, `"provider:model_name"` string convention | -- |
| **CrewAI** | -- | Role-playing metaphor, excessive configuration |
| **LangChain** | -- | Deep abstraction layers, runtime type confusion |
