# Memory Backends

The `exo-memory` package ships with three storage backends that implement the `MemoryStore` protocol: **SQLite** for local development, **Postgres** for production, and **Vector** for semantic search. All backends support soft deletes, version tracking, and async I/O.

## MemoryStore Protocol

All backends implement this protocol:

```python
class MemoryStore(Protocol):
    async def add(self, item: MemoryItem) -> None: ...
    async def get(self, item_id: str) -> MemoryItem | None: ...
    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]: ...
    async def clear(self, *, metadata: MemoryMetadata | None = None) -> int: ...
```

## SQLite Backend

Best for local development, testing, and single-process applications. Uses `aiosqlite` for async I/O with JSON metadata indexes.

### Setup

```python
from exo.memory.backends import SQLiteMemoryStore

# In-memory (testing)
store = SQLiteMemoryStore(":memory:")

# File-based (persistent)
store = SQLiteMemoryStore("/tmp/memory.db")

# Use as async context manager
async with SQLiteMemoryStore("/tmp/memory.db") as store:
    await store.add(memory_item)
    result = await store.get("item-id")
```

### Manual Lifecycle

```python
store = SQLiteMemoryStore("/tmp/memory.db")
await store.init()   # opens connection, creates tables
# ... use store ...
await store.close()  # closes connection
```

### Features

- **Soft deletes** -- items are marked `deleted = 1` rather than removed.
- **Version tracking** -- a `version` column bumps on every upsert.
- **JSON metadata queries** -- uses `json_extract()` for metadata filtering.
- **Partial indexes** -- indexes only cover non-deleted rows for query performance.

### Schema

```sql
CREATE TABLE IF NOT EXISTS memory_items (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'accepted',
    metadata    TEXT NOT NULL DEFAULT '{}',
    extra_json  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted     INTEGER NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1
)
```

### Search

SQLite search uses `LIKE` for content matching and `json_extract` for metadata filtering:

```python
results = await store.search(
    query="Python programming",          # LIKE %query%
    metadata=MemoryMetadata(user_id="u-1"),  # json_extract filter
    memory_type="human",
    limit=20,
)
```

## Postgres Backend

Best for production multi-process applications. Uses `asyncpg` with JSONB indexes and connection pooling.

### Setup

```python
from exo.memory.backends import PostgresMemoryStore

store = PostgresMemoryStore(dsn="postgresql://user:pass@localhost/exo")

async with PostgresMemoryStore(dsn="postgresql://localhost/exo") as store:
    await store.add(memory_item)
```

### Manual Lifecycle

```python
store = PostgresMemoryStore(dsn="postgresql://localhost/exo")
await store.init()   # creates connection pool, creates tables
# ... use store ...
await store.close()  # closes connection pool
```

### Features

- **JSONB metadata** -- native Postgres JSONB columns with expression indexes.
- **Connection pooling** -- `asyncpg.Pool` for concurrent access.
- **Case-insensitive search** -- uses `ILIKE` instead of `LIKE`.
- **Partial indexes** -- includes GIN-style expression indexes on `metadata->>'user_id'` and `metadata->>'session_id'`.

### Schema

```sql
CREATE TABLE IF NOT EXISTS memory_items (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'accepted',
    metadata    JSONB NOT NULL DEFAULT '{}',
    extra_json  JSONB NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted     SMALLINT NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1
)
```

### Search

Postgres search uses `ILIKE` for content and JSONB operators for metadata:

```python
results = await store.search(
    query="Python",                          # ILIKE %query%
    metadata=MemoryMetadata(session_id="s-1"),  # metadata->>'session_id' =
    status=MemoryStatus.ACCEPTED,
    limit=10,
)
```

## Vector Backend

Best for semantic search over memory content. Uses embeddings for cosine similarity search rather than keyword matching.

### Setup

```python
from exo.memory.backends import VectorMemoryStore, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="sk-...",
)

store = VectorMemoryStore(embeddings=embeddings)

# Or with a custom embedding dimension
store = VectorMemoryStore(embeddings=embeddings, dimension=1536)
```

### Embeddings ABC

The vector store requires an `Embeddings` implementation:

```python
from exo.memory.backends.vector import Embeddings

class CustomEmbeddings(Embeddings):
    @property
    def dimension(self) -> int:
        return 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding."""
        return [self._compute(t) for t in texts]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async embedding."""
        return self.embed(texts)
```

### OpenAI Embeddings

The built-in `OpenAIEmbeddings` class wraps the OpenAI API:

```python
from exo.memory.backends import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    api_key="sk-...",
    base_url=None,  # optional: custom API endpoint
)
```

### Search

Vector search uses cosine similarity instead of keyword matching:

```python
# Semantic search -- finds conceptually similar content
results = await store.search(query="How do I authenticate users?", limit=5)

# Results ranked by cosine similarity
for item in results:
    print(f"{item.id}: {item.content[:80]}...")
```

Metadata filtering still works alongside vector search:

```python
results = await store.search(
    query="authentication",
    metadata=MemoryMetadata(user_id="u-1"),
    limit=10,
)
```

## Counting Items

All backends support counting stored items:

```python
# Count non-deleted items
count = await store.count()

# Count all items including soft-deleted
count = await store.count(include_deleted=True)
```

## Clearing Memory

Soft-delete items by metadata filter or clear everything:

```python
# Clear all items
deleted_count = await store.clear()

# Clear items for a specific user
deleted_count = await store.clear(
    metadata=MemoryMetadata(user_id="u-1"),
)

# Clear items for a specific session
deleted_count = await store.clear(
    metadata=MemoryMetadata(session_id="s-42"),
)
```

## Choosing a Backend

| Backend | Best For | Async Library | Search Type |
|---------|----------|---------------|-------------|
| `SQLiteMemoryStore` | Development, testing, single-process | `aiosqlite` | Keyword (LIKE) |
| `PostgresMemoryStore` | Production, multi-process, teams | `asyncpg` | Keyword (ILIKE) |
| `VectorMemoryStore` | Semantic search, RAG pipelines | Depends on embeddings provider | Cosine similarity |

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `SQLiteMemoryStore` | `exo.memory.backends` | SQLite backend with JSON indexes |
| `PostgresMemoryStore` | `exo.memory.backends` | Postgres backend with JSONB and pooling |
| `VectorMemoryStore` | `exo.memory.backends` | Vector backend with cosine similarity |
| `Embeddings` | `exo.memory.backends.vector` | ABC for embedding providers |
| `OpenAIEmbeddings` | `exo.memory.backends` | OpenAI embedding provider |
| `MemoryStore` | `exo.memory` | Protocol: `add`, `get`, `search`, `clear` |
