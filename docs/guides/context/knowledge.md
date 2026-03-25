# Knowledge Store

The `KnowledgeStore` provides text chunking and TF-IDF-like search for ingested documents. It is the retrieval layer that powers the `KnowledgeNeuron` and the context knowledge tools, letting agents search over project documentation, tool outputs, and any other text content.

## Basic Usage

```python
from exo.context._internal.knowledge import KnowledgeStore

store = KnowledgeStore()

# Add documents
store.add("readme", "Exo is a multi-agent framework for building AI applications.")
store.add("guide", "To create an agent, use the Agent class with a model and tools.")

# Search with TF-IDF scoring
results = store.search("create agent")
for result in results:
    print(f"{result.key}: {result.score:.3f}")
    print(f"  {result.content[:80]}...")
```

## Text Chunking

When ingesting large documents, use `chunk_text()` to split them into overlapping chunks before indexing:

```python
from exo.context._internal.knowledge import chunk_text, KnowledgeStore

store = KnowledgeStore()

# Split a large document into chunks
document = open("long_document.txt").read()
chunks = chunk_text(document, chunk_size=500, overlap=50)

# Index each chunk
for i, chunk in enumerate(chunks):
    store.add(f"doc_chunk_{i}", chunk)

# Search across all chunks
results = store.search("specific topic", limit=5)
```

The `chunk_text()` function:

- **chunk_size** -- maximum characters per chunk.
- **overlap** -- number of characters shared between consecutive chunks for context continuity.

## Search

The `search()` method uses TF-IDF-like scoring to rank documents by relevance:

```python
results = store.search("query terms", limit=10)

for r in results:
    print(r.key)      # document identifier
    print(r.content)   # full chunk content
    print(r.score)     # relevance score (0.0 to 1.0)
```

Each `SearchResult` is a dataclass with:

- `key` -- the identifier used when adding the document.
- `content` -- the full text of the matched chunk.
- `score` -- TF-IDF relevance score.

## Document Management

```python
store = KnowledgeStore()

# Add a document
store.add("doc-1", "Content of document 1")

# Get a specific document by key
content = store.get("doc-1")  # "Content of document 1" or None

# Get a range of documents
docs = store.get_range(start=0, end=5)

# Remove a document
store.remove("doc-1")
```

## Integration with Workspace

The `Workspace` can automatically index artifacts into a `KnowledgeStore`:

```python
from exo.context import Workspace
from exo.context._internal.knowledge import KnowledgeStore

knowledge = KnowledgeStore()
workspace = Workspace(base_dir="/tmp/project", knowledge_store=knowledge)

# Writing to workspace auto-indexes the content
workspace.write("design.md", "The system uses event-driven architecture...")

# Now searchable via the knowledge store
results = knowledge.search("event-driven")
```

See the [Workspace guide](workspace.md) for more.

## Integration with Neurons

The built-in `KnowledgeNeuron` (priority 20) retrieves relevant knowledge during prompt building:

```python
from exo.context import PromptBuilder

builder = PromptBuilder(ctx)
builder.add("knowledge")  # retrieves relevant docs based on current task
prompt = builder.build()
```

## Integration with Context Tools

The knowledge tools let agents search the knowledge store directly:

```python
from exo.context import get_knowledge_tools

tools = get_knowledge_tools()
# Returns:
#   get_knowledge  -- retrieve a document by key
#   grep_knowledge -- keyword search across documents
#   search_knowledge -- TF-IDF ranked search
```

See the [Context Tools guide](context-tools.md) for more.

## Advanced Patterns

### Incremental Ingestion

Add documents as the agent discovers them during execution:

```python
async def ingest_tool_result(tool_name: str, result: str, store: KnowledgeStore):
    """Index tool results for future retrieval."""
    key = f"tool:{tool_name}:{hash(result) % 10000}"
    store.add(key, result)
```

### Multi-Source Knowledge Base

Combine multiple document sources into a single store:

```python
store = KnowledgeStore()

# Index project documentation
for path in Path("docs/").glob("*.md"):
    content = path.read_text()
    chunks = chunk_text(content, chunk_size=500, overlap=50)
    for i, chunk in enumerate(chunks):
        store.add(f"{path.stem}_{i}", chunk)

# Index API responses
store.add("api_users", api_response_text)

# Search across everything
results = store.search("authentication flow")
```

### Scoped Search with Prefixes

Use key prefixes to scope searches to specific document categories:

```python
# Add with prefixed keys
store.add("code:main.py", code_content)
store.add("docs:readme", docs_content)
store.add("tool:search_result", tool_output)

# Filter results by prefix
all_results = store.search("function definition")
code_results = [r for r in all_results if r.key.startswith("code:")]
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `KnowledgeStore` | `exo.context._internal.knowledge` | Text storage with TF-IDF search |
| `KnowledgeStore.add(key, content)` | | Add or update a document |
| `KnowledgeStore.get(key)` | | Retrieve a document by key |
| `KnowledgeStore.get_range(start, end)` | | Get documents by index range |
| `KnowledgeStore.remove(key)` | | Remove a document |
| `KnowledgeStore.search(query, limit)` | | TF-IDF ranked search |
| `chunk_text(text, chunk_size, overlap)` | | Split text into overlapping chunks |
| `SearchResult` | `exo.context._internal.knowledge` | Dataclass: `key`, `content`, `score` |
| `Chunk` | `exo.context._internal.knowledge` | Dataclass representing a text chunk |
