# exo.context._internal.knowledge

In-memory artifact indexing with chunking and keyword search. This is an internal module used by `Workspace` for auto-indexing.

## Module Path

```python
from exo.context._internal.knowledge import (
    KnowledgeStore,
    KnowledgeError,
    Chunk,
    SearchResult,
    chunk_text,
)
```

---

## KnowledgeError

Exception raised for knowledge store operation errors.

```python
class KnowledgeError(Exception): ...
```

---

## Chunk

A segment of an artifact's content.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Fields

| Field | Type | Description |
|---|---|---|
| `artifact_name` | `str` | Name of the source artifact |
| `index` | `int` | Chunk index within the artifact |
| `content` | `str` | The chunk text |
| `start` | `int` | Start character position in the original content |
| `end` | `int` | End character position in the original content |

---

## SearchResult

A single search hit with relevance score.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Fields

| Field | Type | Description |
|---|---|---|
| `chunk` | `Chunk` | The matching chunk |
| `score` | `float` | TF-IDF-like relevance score |

---

## chunk_text()

Split text into overlapping segments.

```python
def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | Text to chunk |
| `chunk_size` | `int` | `512` | Maximum characters per chunk (must be positive) |
| `chunk_overlap` | `int` | `64` | Overlap between consecutive chunks (must be in `[0, chunk_size)`) |

**Returns:** List of chunk strings. Empty list for empty text. Single-element list if text fits in one chunk.

**Raises:** `KnowledgeError` if `chunk_size <= 0` or `chunk_overlap >= chunk_size`.

---

## KnowledgeStore

In-memory artifact index with chunking and keyword search.

### Constructor

```python
KnowledgeStore(
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `512` | Maximum characters per chunk |
| `chunk_overlap` | `int` | `64` | Overlap between consecutive chunks |

### Properties

| Property | Type | Description |
|---|---|---|
| `chunk_size` | `int` | Configured chunk size |
| `chunk_overlap` | `int` | Configured chunk overlap |
| `artifact_names` | `list[str]` | Names of indexed artifacts |

### Index Methods

#### add()

```python
def add(self, name: str, content: str) -> list[Chunk]
```

Index an artifact's content. Re-indexes if already present.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Artifact name (must be non-empty) |
| `content` | `str` | Content to index |

**Returns:** List of created `Chunk` objects.

**Raises:** `KnowledgeError` if name is empty.

#### remove()

```python
def remove(self, name: str) -> bool
```

Remove an artifact from the index. Returns `True` if removed.

#### get()

```python
def get(self, name: str) -> list[Chunk]
```

Get all chunks for an artifact. Returns empty list if missing.

#### get_range()

```python
def get_range(self, name: str, start: int, end: int) -> list[Chunk]
```

Get chunks within a character range `[start, end)` for an artifact.

### Search Methods

#### search()

```python
def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]
```

Keyword search across all indexed artifacts. Uses TF-IDF-like scoring: `sum(log(1 + tf))` for each matching query term. Returns up to `top_k` results sorted by descending score.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | *(required)* | Search query string |
| `top_k` | `int` | `5` | Maximum results to return |

**Returns:** List of `SearchResult` sorted by descending score.

### Introspection Methods

#### total_chunks()

```python
def total_chunks(self) -> int
```

Total number of chunks across all artifacts.

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of indexed artifacts |
| `__repr__` | `KnowledgeStore(artifacts=3, chunks=15, chunk_size=512)` |

### Example

```python
from exo.context._internal.knowledge import KnowledgeStore

store = KnowledgeStore(chunk_size=100, chunk_overlap=20)

# Index artifacts
store.add("readme", "This is a long document about Python programming...")
store.add("guide", "A guide to using the API with examples...")

# Search
results = store.search("Python programming", top_k=3)
for r in results:
    print(f"[{r.chunk.artifact_name}#{r.chunk.index}] score={r.score:.2f}")
    print(f"  {r.chunk.content[:80]}...")

# Get chunks for a specific artifact
chunks = store.get("readme")
print(f"readme has {len(chunks)} chunks")

# Range query
range_chunks = store.get_range("readme", start=0, end=200)
```

### Integration with Workspace

When a `KnowledgeStore` is attached to a `Workspace`, artifacts are auto-indexed on write and de-indexed on delete:

```python
from exo.context.workspace import Workspace
from exo.context._internal.knowledge import KnowledgeStore

ks = KnowledgeStore()
ws = Workspace("ws-1", knowledge_store=ks)

await ws.write("doc.md", "# Title\nSome content about AI...")
results = ks.search("AI")  # finds the indexed content
```
