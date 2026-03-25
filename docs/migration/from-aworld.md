# Migrating from AWorld to Exo

This guide provides detailed, side-by-side examples for migrating every major AWorld pattern to its Exo equivalent.

## Package Mapping

| AWorld | Exo | Notes |
|--------|---------|-------|
| `aworld` (monolith) | `exo` (meta-package) | Split into 13 focused packages |
| `aworld.agents` | `exo.agent` | Single `Agent` class replaces 5 agent types |
| `aworld.core.tool` | `exo.tool` | `@tool` decorator, `FunctionTool`, `Tool` ABC |
| `aworld.runner` | `exo.runner` | `run()`, `run.sync()`, `run.stream()` |
| `aworld.models` | `exo.models` | `get_provider("openai:gpt-4o")` factory |
| `aworld.core.context.amni` | `exo.context` | Clean rewrite -- neurons, processors, workspace |
| `aworld.memory` | `exo.memory` | Short/long-term, SQLite/Postgres backends |
| `aworld.mcp_client` | `exo.mcp` | MCP client + `@mcp_server` decorator |
| `aworld.sandbox` | `exo.sandbox` | Local + Kubernetes sandboxes |
| `aworld.trace` | `exo.trace` | OpenTelemetry-based tracing |
| `aworld.evaluations` | `exo.eval` | Scorers, reflection, evaluator |
| `aworld.ralph_loop` | `exo.ralph` | Ralph loop -- state, detectors, runner |
| `aworld.experimental.a2a` | `exo.a2a` | Agent-to-Agent protocol |

## 1. Agent Definition

### Before (AWorld)

```python
from aworld.agents import LLMAgent
from aworld.config.conf import AgentConfig, TaskConfig

agent_config = AgentConfig(
    name="my-agent",
    llm_provider="openai",
    llm_model_id="gpt-4o",
    max_steps=10,
    temperature=0.7,
)
task_config = TaskConfig(name="my-task", description="Do something")

agent = LLMAgent(agent_config=agent_config, task_config=task_config)
```

### After (Exo)

```python
from exo import Agent

agent = Agent(
    name="my-agent",
    model="openai:gpt-4o",        # provider:model in one string
    instructions="Do something",    # instructions replace task_config description
    max_steps=10,
    temperature=0.7,
)
```

### Key Changes

- **Single Agent class** replaces `LLMAgent`, `TaskLLMAgent`, `LoopLLMAgent`, `ParallelLLMAgent`, `SerialLLMAgent`
- **Model string format** -- `"provider:model_name"` replaces separate `llm_provider` and `llm_model_id` fields
- **No TaskConfig** -- Instructions and behavior are Agent constructor parameters
- **All params are keyword-only** -- `Agent(name="x")`, not `Agent("x")`

## 2. Configuration

### Before (AWorld)

AWorld had three config systems used in different places:

```python
# ConfigDict (dict subclass with attribute access)
from aworld.config.conf import ConfigDict
config = ConfigDict({"name": "agent", "llm_provider": "openai"})
print(config.name)  # attribute access on a dict

# BaseConfig (custom base class)
from aworld.config.conf import BaseConfig
class MyConfig(BaseConfig):
    name: str
    value: int = 0

# Pydantic models (some places)
from pydantic import BaseModel
class AgentConfig(BaseModel):
    name: str
```

### After (Exo)

Exo uses Pydantic v2 models exclusively:

```python
from exo.config import AgentConfig, ModelConfig, TaskConfig, RunConfig

# All configs are frozen Pydantic models
agent_config = AgentConfig(
    name="researcher",
    model="openai:gpt-4o",
    instructions="Research topics thoroughly.",
    temperature=0.7,
    max_steps=20,
)

model_config = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="sk-...",
    timeout=30.0,
)
```

### Key Changes

- **One config system** -- Pydantic v2 `BaseModel` everywhere
- **Frozen models** -- `model_config = {"frozen": True}` prevents accidental mutation
- **Field validation** -- `temperature: float = Field(default=1.0, ge=0.0, le=2.0)` with constraints
- **No ConfigDict** -- Use proper Pydantic models with typed fields

## 3. Tool Registration

### Before (AWorld)

```python
from aworld.tools.function_tools import FunctionTool
from aworld.core.tool.base import BaseTool, AsyncBaseTool

# Function tool
tool = FunctionTool(name="search", func=search_fn, description="Search the web")

# Class-based tool (returns gym-style 5-tuple)
class MyTool(BaseTool):
    def execute(self, action):
        return observation, reward, terminated, truncated, info

# Async version was a separate class
class MyAsyncTool(AsyncBaseTool):
    async def execute(self, action):
        return observation, reward, terminated, truncated, info
```

### After (Exo)

```python
from exo import tool, Tool

# Decorator (preferred) -- sync or async
@tool
def search(query: str) -> str:
    """Search the web and return results."""
    return "results"

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.text

# Class-based (for complex tools)
class DatabaseQuery(Tool):
    name = "query_database"
    description = "Execute a SQL query."
    parameters = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The SQL query"},
        },
        "required": ["sql"],
    }

    async def execute(self, **kwargs) -> str:
        sql = kwargs["sql"]
        return await db.execute(sql)
```

### Key Changes

- **`@tool` decorator** is the primary registration method -- schema is auto-generated from type hints + docstring
- **No sync/async split** -- Single `Tool` ABC, sync functions auto-wrapped via `asyncio.to_thread()`
- **Simple return type** -- `execute(**kwargs) -> str | dict` instead of gym 5-tuple
- **Google-style docstring** -- `Args:` section becomes parameter descriptions in the schema
- **No ActionFactory** -- Direct `tool.execute(**args)` call

## 4. Running Agents

### Before (AWorld)

```python
from aworld.runner import create_runner

# 6 different entry points
runner = create_runner(agent_config=config, task_config=task_config)
result = await runner.run(task)
result = runner.sync_run(task)
async for event in runner.streaming_run(task): ...
result = await runner.run_task(task)
async for event in runner.streaming_run_task(task): ...
result = await runner.streamed_run_task(task)
```

### After (Exo)

```python
from exo import Agent, run

agent = Agent(name="assistant", model="openai:gpt-4o", instructions="Be helpful.")

# 3 entry points
result = await run(agent, "What is 2+2?")                    # async
result = run.sync(agent, "What is 2+2?")                      # sync
async for event in run.stream(agent, "Tell me a story"):       # streaming
    if event.type == "text":
        print(event.text, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[calling {event.tool_name}...]")

# Multi-turn conversation
result = await run(agent, "What's the weather?")
result = await run(agent, "What about tomorrow?", messages=result.messages)
```

### Key Changes

- **3 entry points** instead of 6: `run()`, `run.sync()`, `run.stream()`
- **No runner factory** -- Call `run()` directly with an agent
- **Direct agent argument** -- `run(agent, "input")` instead of creating a runner from config
- **Multi-turn via messages** -- Pass `messages=result.messages` to continue a conversation

## 5. Multi-Agent / Swarm

### Before (AWorld)

```python
from aworld.agents import SwarmComposerAgent
from aworld.agents.swarm_builder import SwarmBuilder

# Builder pattern
builder = SwarmBuilder()
builder.add_agent(researcher)
builder.add_agent(writer)
builder.add_agent(editor)
builder.add_edge(researcher, writer)
builder.add_edge(writer, editor)
swarm = builder.build()
```

### After (Exo)

```python
from exo import Agent, Swarm, run

researcher = Agent(name="researcher", ...)
writer = Agent(name="writer", ...)
editor = Agent(name="editor", ...)

# Sequential pipeline with flow DSL
pipeline = Swarm(
    agents=[researcher, writer, editor],
    flow="researcher >> writer >> editor",
)
result = await run(pipeline, "Write an article about quantum computing")

# Handoff (agent-driven delegation)
triage = Agent(name="triage", handoffs=[billing, support], ...)
swarm = Swarm(agents=[triage, billing, support], mode="handoff")
result = await run(swarm, "I need a refund")

# Team (lead + workers)
team = Swarm(agents=[lead, analyst, writer], mode="team")
result = await run(team, "Research and report on market trends")

# Parallel groups in flow
parallel_pipeline = Swarm(
    agents=[fetcher, analyzer_a, analyzer_b, summarizer],
    flow="fetcher >> (analyzer_a | analyzer_b) >> summarizer",
)
```

### Key Changes

- **Flow DSL** replaces builder pattern -- `"a >> b >> c"` is readable and serializable
- **3 modes** -- `"workflow"` (sequential), `"handoff"` (agent-driven), `"team"` (lead-worker)
- **Parallel groups** -- `"(a | b)"` in flow DSL runs agents in parallel
- **No SwarmBuilder** -- Declarative construction via constructor

## 6. Context Engine

### Before (AWorld)

```python
from aworld.core.context.amni.contexts import AmniContext
from aworld.core.context.amni.config import AmniConfig, AmniConfigFactory
from aworld.core.context.amni.prompt.prompts import PromptService
from aworld.core.context.amni.prompt.neurons.neuron_factory import NeuronFactory

# Factory-based config
config = AmniConfigFactory.create("copilot")
context = AmniContext(config=config)

# Neuron factory registration
NeuronFactory.register("task", TaskNeuron)
NeuronFactory.register("history", HistoryNeuron)

# Service-based prompt building
prompt_service = PromptService(context)
prompt = await prompt_service.build_prompt(neuron_names=["task", "history"])
```

### After (Exo)

```python
from exo.context import Context, ContextConfig, PromptBuilder

# Config-based (no factory)
config = ContextConfig(
    mode="copilot",            # "pilot" | "copilot" | "navigator"
    history_rounds=20,
    summary_threshold=30,
    offload_threshold=4000,
)
ctx = Context(task_id="task-1", config=config)

# Direct prompt building (no factory)
builder = PromptBuilder(ctx)
builder.add("system", "You are a helpful assistant.")
builder.add("task")
builder.add("history")
builder.add("knowledge", query="quantum computing")
prompt = await builder.build()

# Hierarchical state
child_ctx = ctx.fork(task_id="subtask-1")
child_ctx.state.set("progress", "researching")  # local only
ctx.merge(child_ctx)  # merge results back
```

### Key Changes

- **`ContextConfig(mode="copilot")`** replaces `AmniConfigFactory.create("copilot")` -- config values instead of class hierarchy
- **`PromptBuilder.add(type, **kwargs)`** replaces `NeuronFactory.register()` + `PromptService.build_prompt()` -- direct API instead of factory
- **`Context.fork()` / `Context.merge()`** for hierarchical state -- parent-child contexts with isolated writes
- **`ContextProcessor`** replaces scattered hooks -- event-driven pipeline at `pre_llm_call`, `post_tool_call`, etc.

## 7. Memory

### Before (AWorld)

```python
from aworld.memory.main import Memory
from aworld.memory.longterm.default import DefaultLongTermMemory
from aworld.memory.embeddings.factory import EmbeddingFactory

memory = Memory(
    long_term=DefaultLongTermMemory(
        embedding=EmbeddingFactory.create("openai"),
        db_path="memory.db",
    )
)
```

### After (Exo)

```python
from exo.memory import ShortTermMemory, LongTermMemory

agent = Agent(
    name="assistant",
    memory=LongTermMemory(backend="sqlite", path="memory.db"),
    ...
)
```

### Key Changes

- **Simpler API** -- Memory configured directly on the Agent constructor
- **Backend selection** via string, not factory chain
- **Two types** -- `ShortTermMemory` (session-scoped) and `LongTermMemory` (persistent)

## 8. Import Mapping Reference

| AWorld Import | Exo Import |
|--------------|----------------|
| `from aworld.agents import LLMAgent` | `from exo import Agent` |
| `from aworld.agents import TaskLLMAgent` | `from exo import Agent` |
| `from aworld.agents import ParallelLLMAgent` | `from exo import Swarm` (with parallel flow) |
| `from aworld.agents import SerialLLMAgent` | `from exo import Swarm` (with sequential flow) |
| `from aworld.agents import SwarmComposerAgent` | `from exo import Swarm` |
| `from aworld.config.conf import AgentConfig` | `from exo.config import AgentConfig` |
| `from aworld.config.conf import TaskConfig` | `from exo.config import TaskConfig` |
| `from aworld.config.conf import ConfigDict` | Use Pydantic models directly |
| `from aworld.core.tool.base import BaseTool` | `from exo import Tool` |
| `from aworld.core.tool.base import AsyncBaseTool` | `from exo import Tool` (same class) |
| `from aworld.tools.function_tools import FunctionTool` | `from exo import tool` (decorator) |
| `from aworld.runner import create_runner` | `from exo import run` |
| `from aworld.models import llm` | `from exo.models import get_provider` |
| `from aworld.core.context.amni.contexts import AmniContext` | `from exo.context import Context` |
| `from aworld.core.context.amni.config import AmniConfig` | `from exo.context import ContextConfig` |
| `from aworld.memory.main import Memory` | `from exo.memory import LongTermMemory` |
| `from aworld.trace import traced` | `from exo.trace import traced` |
| `from aworld.evaluations import Evaluator` | `from exo.eval import Evaluator` |

## Migration Checklist

Use this checklist to track your migration progress:

- [ ] **Install Exo packages** -- `pip install exo` or individual packages
- [ ] **Update imports** -- Replace `aworld.*` imports with `exo.*` equivalents (see table above)
- [ ] **Migrate agent definitions** -- Replace agent subclasses with single `Agent` class
- [ ] **Migrate tool definitions** -- Replace `BaseTool`/`AsyncBaseTool`/`FunctionTool` with `@tool` decorator or `Tool` ABC
- [ ] **Migrate configuration** -- Replace `ConfigDict`/`BaseConfig` with Pydantic v2 models
- [ ] **Migrate runner calls** -- Replace `create_runner().run()` with `run()`, `run.sync()`, `run.stream()`
- [ ] **Migrate multi-agent** -- Replace `SwarmBuilder`/`SwarmComposerAgent` with `Swarm(flow=..., mode=...)`
- [ ] **Migrate context engine** -- Replace `AmniContext`/`AmniConfig` with `Context`/`ContextConfig`
- [ ] **Migrate memory** -- Replace factory-based memory with direct `ShortTermMemory`/`LongTermMemory`
- [ ] **Migrate model configuration** -- Replace separate provider/model fields with `"provider:model"` strings
- [ ] **Update tests** -- Replace mock patterns and async test setup
- [ ] **Run type checker** -- `uv run pyright packages/` to verify type safety
- [ ] **Run tests** -- `uv run pytest` to verify behavior

## Common Gotchas

1. **`run.sync()` cannot be called from an async context.** If you are already in an async function, use `await run()` instead. `asyncio.run()` raises an error if an event loop is already running.

2. **Tool return types changed.** AWorld tools returned a gym-style 5-tuple `(obs, reward, term, trunc, info)`. Exo tools return `str | dict`. You need to update the return statements.

3. **Message types are immutable.** Exo messages use `model_config = {"frozen": True}`. You cannot modify a message after creation -- create a new one instead.

4. **No automatic config hoisting.** AWorld sometimes hoisted kwargs between config levels. Exo requires explicit config construction.

5. **Model string format.** Use `"openai:gpt-4o"` not `"gpt-4o"` (though bare model names default to OpenAI).
