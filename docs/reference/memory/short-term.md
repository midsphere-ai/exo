# exo.memory.short_term

Short-term memory: conversation context with scope filtering and windowing.

## Module Path

```python
from exo.memory.short_term import ShortTermMemory
```

---

## ShortTermMemory

In-memory conversation store with scope-based filtering and windowing. Implements the `MemoryStore` protocol.

### Constructor

```python
ShortTermMemory(
    *,
    scope: str = "task",
    max_rounds: int = 0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scope` | `str` | `"task"` | Filtering scope: `"user"`, `"session"`, or `"task"` |
| `max_rounds` | `int` | `0` | Maximum conversation rounds to keep (0 = unlimited) |

**Raises:** `MemoryError` if scope is not one of `"user"`, `"session"`, `"task"`.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `scope` | `str` | Current filtering scope |
| `max_rounds` | `int` | Maximum rounds setting |

### Methods (MemoryStore Protocol)

#### add()

```python
async def add(self, item: MemoryItem) -> None
```

Persist a memory item to the in-memory store.

#### get()

```python
async def get(self, item_id: str) -> MemoryItem | None
```

Retrieve a memory item by ID. Returns `None` if not found.

#### search()

```python
async def search(
    self,
    *,
    query: str = "",
    metadata: MemoryMetadata | None = None,
    memory_type: str | None = None,
    status: MemoryStatus | None = None,
    limit: int = 10,
) -> list[MemoryItem]
```

Search memory with optional filters, scope awareness, and windowing.

**Processing order:**
1. Scope-based and optional filters (metadata, type, status)
2. Keyword filter (case-insensitive substring match on content)
3. Conversation windowing (if `max_rounds > 0`)
4. Tool call integrity filtering (removes dangling tool calls/results)
5. Limit applied

#### clear()

```python
async def clear(
    self,
    *,
    metadata: MemoryMetadata | None = None,
) -> int
```

Remove memory items matching the filter. Returns count removed.

### Scope-Based Filtering

The `scope` parameter controls how metadata matching works:

| Scope | Matches On |
|---|---|
| `"user"` | `user_id`, `agent_id` |
| `"session"` | `user_id`, `session_id`, `agent_id` |
| `"task"` | `user_id`, `session_id`, `task_id`, `agent_id` |

### Conversation Windowing

When `max_rounds > 0`, only the last N conversation rounds are returned. A round starts with a `human` message. System messages before the window cutoff are always retained.

### Tool Call Integrity

The search method removes trailing incomplete tool call pairs:
- AI messages with `tool_calls` that have no following tool results
- Tool results with no matching AI message

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of stored items |
| `__repr__` | `ShortTermMemory(scope='task', max_rounds=10, items=5)` |

### Example

```python
import asyncio
from exo.memory import (
    ShortTermMemory, HumanMemory, AIMemory,
    SystemMemory, MemoryMetadata,
)

async def main():
    memory = ShortTermMemory(scope="session", max_rounds=5)
    meta = MemoryMetadata(user_id="u1", session_id="s1")

    # Add conversation
    await memory.add(SystemMemory(content="You are helpful.", metadata=meta))
    await memory.add(HumanMemory(content="Hello!", metadata=meta))
    await memory.add(AIMemory(content="Hi there!", metadata=meta))
    await memory.add(HumanMemory(content="What is Python?", metadata=meta))
    await memory.add(AIMemory(content="Python is a language.", metadata=meta))

    # Search with scope filtering
    results = await memory.search(metadata=meta)
    print(len(results))  # 5

    # Keyword search
    results = await memory.search(query="Python", metadata=meta)
    print(len(results))  # 2 (question + answer about Python)

    # Clear by session
    removed = await memory.clear(metadata=meta)
    print(removed)  # 5

asyncio.run(main())
```
