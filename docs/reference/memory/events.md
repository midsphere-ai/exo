# exo.memory.events

Memory event integration for async processing pipelines.

## Module Path

```python
from exo.memory.events import (
    MemoryEventEmitter,
    MEMORY_ADDED,
    MEMORY_SEARCHED,
    MEMORY_CLEARED,
)
```

---

## Event Constants

| Constant | Value | Description |
|---|---|---|
| `MEMORY_ADDED` | `"memory:added"` | Emitted after `add(item)` |
| `MEMORY_SEARCHED` | `"memory:searched"` | Emitted after `search(...)` with results |
| `MEMORY_CLEARED` | `"memory:cleared"` | Emitted after `clear(...)` with count |

---

## MemoryEventEmitter

Wraps a `MemoryStore` to emit events on operations via an `EventBus`.

### Constructor

```python
MemoryEventEmitter(store: Any, bus: EventBus | None = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `store` | `Any` | *(required)* | Any MemoryStore-compatible backend |
| `bus` | `EventBus \| None` | `None` | EventBus for emitting events (creates new one if `None`) |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `store` | `Any` | The wrapped memory store |
| `bus` | `EventBus` | The event bus |

### Methods

#### add()

```python
async def add(self, item: MemoryItem) -> None
```

Add item and emit `memory:added` event with `item=item`.

#### get()

```python
async def get(self, item_id: str) -> MemoryItem | None
```

Retrieve item by ID. No event emitted.

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

Search and emit `memory:searched` event with `results=results, query=query`.

#### clear()

```python
async def clear(
    self,
    *,
    metadata: MemoryMetadata | None = None,
) -> int
```

Clear and emit `memory:cleared` event with `count=count, metadata=metadata`.

### Example

```python
import asyncio
from exo.events import EventBus
from exo.memory import (
    ShortTermMemory, HumanMemory, MemoryMetadata,
    MemoryEventEmitter, MEMORY_ADDED, MEMORY_SEARCHED,
)

async def main():
    store = ShortTermMemory()
    bus = EventBus()
    emitter = MemoryEventEmitter(store, bus)

    # Register event handlers
    async def on_added(**kwargs):
        item = kwargs["item"]
        print(f"Memory added: [{item.memory_type}] {item.content[:50]}")

    async def on_searched(**kwargs):
        results = kwargs["results"]
        print(f"Search returned {len(results)} results")

    bus.on(MEMORY_ADDED, on_added)
    bus.on(MEMORY_SEARCHED, on_searched)

    # Operations trigger events
    meta = MemoryMetadata(user_id="u1")
    await emitter.add(HumanMemory(content="Hello!", metadata=meta))
    # Output: Memory added: [human] Hello!

    results = await emitter.search(query="Hello")
    # Output: Search returned 1 results

asyncio.run(main())
```
