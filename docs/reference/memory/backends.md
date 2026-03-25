# exo.memory.backends

Persistent backend implementations: SQLite, PostgreSQL, and Vector (embedding-based).

## Module Paths

```python
from exo.memory.backends.sqlite import SQLiteMemoryStore
from exo.memory.backends.postgres import PostgresMemoryStore
from exo.memory.backends.vector import VectorMemoryStore, Embeddings, OpenAIEmbeddings
```

---

## SQLiteMemoryStore

SQLite-backed persistent memory store with JSON metadata indexes, soft deletes, and optimistic concurrency versioning.

### Constructor

```python
SQLiteMemoryStore(db_path: str = ":memory:")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` | `":memory:"` | SQLite database path (use `":memory:"` for in-memory) |

### Lifecycle

```python
# Context manager (recommended)
async with SQLiteMemoryStore("/path/to/db.sqlite") as store:
    await store.add(item)

# Manual lifecycle
store = SQLiteMemoryStore("/path/to/db.sqlite")
await store.init()   # create tables
# ... use store ...
await store.close()  # close connection
```

### Methods (MemoryStore Protocol)

#### add()

```python
async def add(self, item: MemoryItem) -> None
```

Persist a memory item. Uses upsert -- bumps version on conflict.

#### get()

```python
async def get(self, item_id: str) -> MemoryItem | None
```

Retrieve a non-deleted memory item by ID.

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

Search with SQL LIKE for query, `json_extract()` for metadata fields. Results ordered by `created_at DESC`.

#### clear()

```python
async def clear(*, metadata: MemoryMetadata | None = None) -> int
```

Soft-delete (sets `deleted = 1`). Returns count of affected rows.

### Extra Methods

#### count()

```python
async def count(self, *, include_deleted: bool = False) -> int
```

Return the number of stored items.

### Schema

The SQLite table has these columns:

| Column | Type | Description |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | Memory item ID |
| `content` | `TEXT NOT NULL` | Content |
| `memory_type` | `TEXT NOT NULL` | Item type discriminator |
| `status` | `TEXT NOT NULL` | Lifecycle status |
| `metadata` | `TEXT NOT NULL` | JSON-encoded metadata |
| `extra_json` | `TEXT NOT NULL` | JSON-encoded subclass fields |
| `created_at` | `TEXT NOT NULL` | ISO-8601 timestamp |
| `updated_at` | `TEXT NOT NULL` | ISO-8601 timestamp |
| `deleted` | `INTEGER NOT NULL` | Soft-delete flag (0/1) |
| `version` | `INTEGER NOT NULL` | Optimistic concurrency version |

**Requires:** `aiosqlite` package.

---

## PostgresMemoryStore

PostgreSQL-backed persistent memory store using asyncpg. Uses JSONB for metadata with GIN-style indexes.

### Constructor

```python
PostgresMemoryStore(dsn: str = "postgresql://localhost/exo")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dsn` | `str` | `"postgresql://localhost/exo"` | PostgreSQL connection DSN |

### Lifecycle

```python
# Context manager (recommended)
async with PostgresMemoryStore("postgresql://user:pass@host/db") as store:
    await store.add(item)

# Manual lifecycle
store = PostgresMemoryStore("postgresql://user:pass@host/db")
await store.init()   # create pool + tables
# ... use store ...
await store.close()  # close pool
```

### Methods

Same `MemoryStore` protocol as SQLite. Differences:

- Uses `ILIKE` for case-insensitive query search
- Uses `metadata->>'field'` JSONB operators for metadata filtering
- Uses parameterized queries (`$1`, `$2`, ...) instead of `?`
- Includes additional JSONB indexes for `user_id` and `session_id`

### Extra Methods

#### count()

```python
async def count(self, *, include_deleted: bool = False) -> int
```

**Requires:** `asyncpg` package.

---

## Embeddings (ABC)

Abstract base class for embedding providers.

### Constructor

No parameters (abstract).

### Abstract Methods

#### embed()

```python
def embed(self, text: str) -> list[float]
```

Generate an embedding vector synchronously.

#### aembed()

```python
async def aembed(self, text: str) -> list[float]
```

Generate an embedding vector asynchronously.

#### dimension (property)

```python
@property
def dimension(self) -> int
```

Return the embedding dimension.

---

## OpenAIEmbeddings

OpenAI-compatible embedding provider. Works with any API that follows the OpenAI embeddings format.

### Constructor

```python
OpenAIEmbeddings(
    *,
    model: str = "text-embedding-3-small",
    dimension: int = 1536,
    api_key: str | None = None,
    base_url: str | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"text-embedding-3-small"` | Embedding model name |
| `dimension` | `int` | `1536` | Embedding vector dimension |
| `api_key` | `str \| None` | `None` | OpenAI API key |
| `base_url` | `str \| None` | `None` | Custom API base URL |

**Requires:** `openai` package.

---

## VectorMemoryStore

In-memory vector store backed by an `Embeddings` provider. Implements the `MemoryStore` protocol with semantic (cosine similarity) search.

### Constructor

```python
VectorMemoryStore(embeddings: Embeddings)
```

| Parameter | Type | Description |
|---|---|---|
| `embeddings` | `Embeddings` | The embedding provider |

### Properties

| Property | Type | Description |
|---|---|---|
| `embeddings` | `Embeddings` | The underlying embeddings provider |

### Methods (MemoryStore Protocol)

#### add()

```python
async def add(self, item: MemoryItem) -> None
```

Persist a memory item and compute its embedding vector.

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

**With query:** Embeds the query, ranks candidates by cosine similarity.

**Without query:** Returns newest items first.

Metadata, memory_type, and status filters are applied as post-filters before ranking.

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of stored items |
| `__repr__` | `VectorMemoryStore(items=10, dimension=1536)` |

### Example

```python
import asyncio
from exo.memory.backends.vector import VectorMemoryStore, OpenAIEmbeddings
from exo.memory import HumanMemory, AIMemory

async def main():
    embeddings = OpenAIEmbeddings(api_key="sk-...")
    store = VectorMemoryStore(embeddings)

    await store.add(HumanMemory(content="I love hiking in the mountains"))
    await store.add(HumanMemory(content="Python is my favorite language"))
    await store.add(AIMemory(content="That's great! The outdoors are wonderful."))

    # Semantic search
    results = await store.search(query="outdoor activities", limit=2)
    for item in results:
        print(f"[{item.memory_type}] {item.content}")
    # Most relevant items about outdoors/hiking appear first

asyncio.run(main())
```
