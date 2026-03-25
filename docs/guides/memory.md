# Memory

The `exo-memory` package provides typed, protocol-based memory management for agents. It separates memory into short-term (conversation-scoped) and long-term (persistent, extractable) layers, with a typed hierarchy for different message roles and a status lifecycle for memory curation.

## Basic Usage

```python
from exo.memory import (
    MemoryItem, MemoryMetadata, MemoryStatus,
    HumanMemory, AIMemory, SystemMemory, ToolMemory,
    ShortTermMemory,
)

# Create typed memories
system_msg = SystemMemory(content="You are a helpful assistant.")
user_msg = HumanMemory(
    content="What is Python?",
    metadata=MemoryMetadata(user_id="u-1", session_id="s-1"),
)
ai_msg = AIMemory(content="Python is a programming language.", tool_calls=[])

# Short-term memory with session scope
stm = ShortTermMemory(scope="session")
stm.add(system_msg)
stm.add(user_msg)
stm.add(ai_msg)

# Retrieve recent conversation
messages = stm.get_messages(
    metadata=MemoryMetadata(session_id="s-1"),
    max_rounds=10,
)
```

## Memory Type Hierarchy

All memory types extend `MemoryItem`, which is a Pydantic `BaseModel`:

| Type | Fields | Use Case |
|------|--------|----------|
| `MemoryItem` | `id`, `content`, `memory_type`, `status`, `metadata`, `created_at`, `updated_at` | Base type |
| `SystemMemory` | (base fields) | System instructions |
| `HumanMemory` | (base fields) | User messages |
| `AIMemory` | `tool_calls: list[dict]` | Assistant responses with optional tool calls |
| `ToolMemory` | `tool_call_id`, `tool_name`, `is_error` | Tool execution results |

```python
# AI message with tool calls
ai_msg = AIMemory(
    content="Let me search for that.",
    tool_calls=[{
        "id": "call_123",
        "function": {"name": "search", "arguments": '{"query": "Python"}'},
    }],
)

# Tool result
tool_msg = ToolMemory(
    content="Python is a high-level programming language...",
    tool_call_id="call_123",
    tool_name="search",
    is_error=False,
)
```

## Memory Metadata

`MemoryMetadata` carries routing information for scoping and filtering:

```python
from exo.memory import MemoryMetadata

metadata = MemoryMetadata(
    user_id="user-alice",
    session_id="session-42",
    task_id="task-research",
    agent_id="assistant",
)
```

All fields are optional. They are used by the memory store and short-term memory for filtering.

## Status Lifecycle

Memory items follow a lifecycle controlled by `MemoryStatus`:

| Status | Description |
|--------|-------------|
| `DRAFT` | Newly created, pending review |
| `ACCEPTED` | Confirmed and usable |
| `DISCARD` | Marked for removal |

```python
item = HumanMemory(content="Hello")
print(item.status)  # MemoryStatus.ACCEPTED (default)

# Transition to a new status
item.transition(MemoryStatus.DISCARD)
print(item.status)  # MemoryStatus.DISCARD
```

## Short-Term Memory

`ShortTermMemory` manages recent conversation messages with scoping and windowing:

```python
from exo.memory import ShortTermMemory, MemoryMetadata

stm = ShortTermMemory(scope="session")

# Add messages
stm.add(system_msg)
stm.add(user_msg)
stm.add(ai_msg)

# Retrieve with windowing
messages = stm.get_messages(
    metadata=MemoryMetadata(session_id="s-1"),
    max_rounds=5,   # last 5 conversation rounds
)
```

The `scope` parameter controls filtering:

| Scope | Filters By |
|-------|-----------|
| `"user"` | `metadata.user_id` |
| `"session"` | `metadata.session_id` |
| `"task"` | `metadata.task_id` |

Short-term memory also maintains tool call integrity -- when windowing truncates messages, it ensures tool call/result pairs are kept together.

## Long-Term Memory

`LongTermMemory` stores persistent knowledge extracted from conversations:

```python
from exo.memory import LongTermMemory, ExtractionType

ltm = LongTermMemory()

# Store extracted knowledge
ltm.add(
    content="User prefers Python over JavaScript",
    extraction_type=ExtractionType.USER_PROFILE,
    metadata=MemoryMetadata(user_id="u-1"),
)
```

Extraction types:

| Type | Description |
|------|-------------|
| `USER_PROFILE` | User preferences and characteristics |
| `AGENT_EXPERIENCE` | Patterns learned from agent execution |
| `FACTS` | Factual knowledge extracted from conversations |

Long-term memory includes deduplication -- adding a memory that is similar to an existing one updates rather than duplicates.

## Memory Orchestrator

The `MemoryOrchestrator` coordinates extraction from conversations into long-term memory:

```python
from exo.memory import MemoryOrchestrator, OrchestratorConfig

orchestrator = MemoryOrchestrator(
    config=OrchestratorConfig(
        batch_size=5,
        max_concurrent=3,
    ),
)

# Submit conversation for extraction
orchestrator.submit(conversation_messages, extractor=my_extractor)

# Process all pending extractions
results = await orchestrator.process_all()
```

## Memory Summarization

When conversations grow long, use summarization to compress older messages:

```python
from exo.memory import SummaryConfig, SummaryTemplate, check_trigger, generate_summary

config = SummaryConfig(
    threshold=20,          # trigger after 20 messages
    template=SummaryTemplate.CONVERSATION,
)

# Check if summarization should trigger
trigger = check_trigger(messages, config)
if trigger.should_trigger:
    summary = await generate_summary(
        messages=trigger.messages_to_summarize,
        summarizer=my_summarizer,
        template=SummaryTemplate.CONVERSATION,
    )
```

Summary templates:

| Template | Description |
|----------|-------------|
| `CONVERSATION` | General conversation summary |
| `FACTS` | Extract factual assertions |
| `PROFILES` | Summarize user/agent profiles |

## Auto-Persistence

`MemoryPersistence` automatically saves LLM responses and tool results to a memory store using hooks. Instead of manually adding memory items after each call, attach a `MemoryPersistence` instance to your agent and it handles persistence for both `run()` and `run.stream()`.

```python
from exo.agent import Agent
from exo.memory import ShortTermMemory, MemoryPersistence, MemoryMetadata, HumanMemory
from exo.runner import run

agent = Agent(name="assistant", model="openai:gpt-4o")

store = ShortTermMemory(scope="session")
meta = MemoryMetadata(user_id="u-1", session_id="s-1")

persistence = MemoryPersistence(store, metadata=meta)
persistence.attach(agent)

# Save the user input manually (MemoryPersistence handles AI + tool memories)
await store.add(HumanMemory(content="Hello!", metadata=meta))

# Run the agent -- AI responses and tool results are auto-saved
result = await run(agent, "Hello!")

# The store now contains: HumanMemory, AIMemory, (and ToolMemory for each tool call)
items = await store.search(metadata=meta, limit=100)

# Detach when done
persistence.detach(agent)
```

**How it works:**

- `attach()` registers `POST_LLM_CALL` and `POST_TOOL_CALL` hooks on the agent
- Each LLM response is saved as an `AIMemory` (including any `tool_calls`)
- Each tool result is saved as a `ToolMemory` (with `tool_call_id`, `tool_name`, and `is_error`)
- `detach()` removes the hooks cleanly
- The caller is responsible for saving `HumanMemory` (user input) before calling `run()`

This is the same mechanism used internally by the [distributed worker](../distributed/workers.md#memory-hydration) for automatic memory hydration.

## Event Integration

Wrap any `MemoryStore` with `MemoryEventEmitter` to emit events on memory operations:

```python
from exo.memory import MemoryEventEmitter
from exo.events import EventBus

bus = EventBus()
store = SQLiteMemoryStore(":memory:")
emitting_store = MemoryEventEmitter(store=store, bus=bus)

# Events are emitted on add, search, clear
bus.subscribe("memory.added", lambda e: print(f"Added: {e}"))

await emitting_store.add(user_msg)
# prints: Added: ...
```

Events: `memory.added`, `memory.searched`, `memory.cleared`.

## Configuration

See [Memory Backends](memory-backends.md) for storage backend configuration (SQLite, Postgres, Vector).

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `MemoryItem` | `exo.memory` | Base memory type (Pydantic model) |
| `SystemMemory` | `exo.memory` | System instruction memory |
| `HumanMemory` | `exo.memory` | User message memory |
| `AIMemory` | `exo.memory` | Assistant response with `tool_calls` |
| `ToolMemory` | `exo.memory` | Tool result with `tool_call_id`, `tool_name`, `is_error` |
| `MemoryMetadata` | `exo.memory` | Routing metadata: `user_id`, `session_id`, `task_id`, `agent_id` |
| `MemoryStatus` | `exo.memory` | Enum: `DRAFT`, `ACCEPTED`, `DISCARD` |
| `MemoryStore` | `exo.memory` | Protocol: `add`, `get`, `search`, `clear` |
| `ShortTermMemory` | `exo.memory` | Scoped, windowed conversation memory |
| `LongTermMemory` | `exo.memory` | Persistent extracted knowledge with deduplication |
| `ExtractionType` | `exo.memory` | Enum: `USER_PROFILE`, `AGENT_EXPERIENCE`, `FACTS` |
| `MemoryOrchestrator` | `exo.memory` | Coordinates LLM extraction into long-term memory |
| `SummaryConfig` | `exo.memory` | Configuration for summarization triggers |
| `SummaryTemplate` | `exo.memory` | Enum: `CONVERSATION`, `FACTS`, `PROFILES` |
| `check_trigger` | `exo.memory` | Check if summarization should trigger |
| `generate_summary` | `exo.memory` | Generate a summary from messages |
| `MemoryPersistence` | `exo.memory` | Hook-based auto-persistence for LLM responses and tool results |
| `MemoryEventEmitter` | `exo.memory` | Event-emitting wrapper for any `MemoryStore` |
