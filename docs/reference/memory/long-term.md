# exo.memory.long_term

Persistent knowledge across sessions with async LLM extraction.

## Module Path

```python
from exo.memory.long_term import (
    LongTermMemory,
    ExtractionType,
    Extractor,
    ExtractionTask,
    TaskStatus,
    MemoryOrchestrator,
    OrchestratorConfig,
)
```

---

## ExtractionType

Types of knowledge that can be extracted from conversations.

```python
class ExtractionType(StrEnum):
    USER_PROFILE = "user_profile"
    AGENT_EXPERIENCE = "agent_experience"
    FACTS = "facts"
```

---

## TaskStatus

Lifecycle states for extraction tasks.

```python
class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

---

## Extractor (Protocol)

Protocol for LLM-powered knowledge extraction.

```python
@runtime_checkable
class Extractor(Protocol):
    async def extract(self, prompt: str) -> str: ...
```

Implement this protocol by wrapping your LLM provider.

---

## ExtractionTask

A single knowledge extraction task.

**Decorator:** `@dataclass(slots=True)`

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `extraction_type` | `ExtractionType` | *(required)* | What kind of knowledge to extract |
| `source_items` | `list[MemoryItem]` | *(required)* | Memory items to extract from |
| `task_id` | `str` | Auto UUID | Unique identifier |
| `status` | `TaskStatus` | `PENDING` | Current lifecycle status |
| `result` | `str \| None` | `None` | Extracted text (set on completion) |
| `error` | `str \| None` | `None` | Error message (set on failure) |
| `created_at` | `str` | ISO-8601 now | Creation timestamp |
| `completed_at` | `str \| None` | `None` | Completion timestamp |

### Methods

| Method | Description |
|---|---|
| `start()` | Mark task as `RUNNING` |
| `complete(result)` | Mark as `COMPLETED` with result text |
| `fail(error)` | Mark as `FAILED` with error message |

---

## LongTermMemory

Persistent memory store for knowledge that spans sessions. Implements the `MemoryStore` protocol with content-based deduplication.

### Constructor

```python
LongTermMemory(*, namespace: str = "default")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `namespace` | `str` | `"default"` | Isolation namespace for multi-tenant usage |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `namespace` | `str` | The isolation namespace |

### Methods (MemoryStore Protocol)

#### add()

```python
async def add(self, item: MemoryItem) -> None
```

Persist a memory item, **deduplicating by content**. Items with identical `content` and `memory_type` are silently skipped.

#### get()

```python
async def get(self, item_id: str) -> MemoryItem | None
```

Retrieve by ID.

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

Search with filters. Results sorted by creation time (newest first).

#### clear()

```python
async def clear(*, metadata: MemoryMetadata | None = None) -> int
```

Remove items matching filter.

---

## OrchestratorConfig

Configuration for the memory orchestrator.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
OrchestratorConfig(
    extraction_types: tuple[ExtractionType, ...] = (
        ExtractionType.USER_PROFILE,
        ExtractionType.AGENT_EXPERIENCE,
        ExtractionType.FACTS,
    ),
    prompts: dict[ExtractionType, str] = {},
    min_items: int = 3,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `extraction_types` | `tuple[ExtractionType, ...]` | All three types | Which knowledge types to extract |
| `prompts` | `dict[ExtractionType, str]` | `{}` | Custom prompt overrides per type |
| `min_items` | `int` | `3` | Minimum items to trigger extraction |

### Methods

#### get_prompt()

```python
def get_prompt(self, extraction_type: ExtractionType) -> str
```

Get the prompt for an extraction type, falling back to built-in defaults.

---

## MemoryOrchestrator

Orchestrates async LLM extraction of structured knowledge from conversations.

### Constructor

```python
MemoryOrchestrator(
    store: LongTermMemory,
    *,
    config: OrchestratorConfig | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `store` | `LongTermMemory` | *(required)* | Long-term memory store for results |
| `config` | `OrchestratorConfig \| None` | `None` | Configuration (defaults to `OrchestratorConfig()`) |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `store` | `LongTermMemory` | The backing store |
| `config` | `OrchestratorConfig` | Orchestrator configuration |

### Methods

#### submit()

```python
def submit(
    self,
    items: Sequence[MemoryItem],
    *,
    extraction_type: ExtractionType | None = None,
    metadata: MemoryMetadata | None = None,
) -> list[ExtractionTask]
```

Submit extraction tasks. Creates one task per configured extraction type (or a single task if `extraction_type` is specified).

#### process()

```python
async def process(
    self,
    task_id: str,
    extractor: Any,
    *,
    metadata: MemoryMetadata | None = None,
) -> ExtractionTask
```

Process a single extraction task. Runs the extractor, stores the result, updates the task status.

**Raises:** `KeyError` if no task with the given ID exists.

#### process_all()

```python
async def process_all(
    self,
    extractor: Any,
    *,
    metadata: MemoryMetadata | None = None,
) -> list[ExtractionTask]
```

Process all pending tasks.

#### get_task()

```python
def get_task(self, task_id: str) -> ExtractionTask | None
```

#### list_tasks()

```python
def list_tasks(self, *, status: TaskStatus | None = None) -> list[ExtractionTask]
```

### Example

```python
import asyncio
from exo.memory import (
    LongTermMemory, MemoryOrchestrator, OrchestratorConfig,
    ExtractionType, HumanMemory, AIMemory,
)

class MyExtractor:
    async def extract(self, prompt: str) -> str:
        return "Extracted: user prefers Python"

async def main():
    store = LongTermMemory(namespace="user-1")
    orchestrator = MemoryOrchestrator(
        store,
        config=OrchestratorConfig(
            extraction_types=(ExtractionType.USER_PROFILE,),
        ),
    )

    # Submit items for extraction
    items = [
        HumanMemory(content="I love Python and data science"),
        AIMemory(content="Great! Python is excellent for data science."),
    ]
    tasks = orchestrator.submit(items)

    # Process with extractor
    results = await orchestrator.process_all(MyExtractor())
    for task in results:
        print(f"Status: {task.status}, Result: {task.result}")

    # Check long-term memory
    stored = await store.search(memory_type="user_profile")
    print(stored)

asyncio.run(main())
```
