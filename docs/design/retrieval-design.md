# Retrieval Pipeline Design — Modular RAG for Exo

**Status:** Proposed
**Epic:** 3 — RAG/Retrieval Pipeline
**Package:** New `exo-retrieval` (depends on `exo-core`)
**Date:** 2026-03-10

---

## 1. Motivation

Exo currently has two partial retrieval-related capabilities:

- **exo-memory** — `Embeddings` ABC, `OpenAIEmbeddings`, and `VectorMemoryStore`
  providing cosine-similarity search over memory items.
- **exo-context** — `KnowledgeStore` with character-level chunking and TF-IDF
  keyword search over workspace artifacts.

Neither supports document ingestion pipelines, multiple vector store backends,
hybrid (dense + sparse) search, query rewriting, reranking, or knowledge graph
retrieval. Agent-core (`openjiuwen/core/retrieval/`) provides a comprehensive
RAG system with all of these capabilities.

This document designs `exo-retrieval`, a new package that ports agent-core's
retrieval architecture to Exo while maintaining clean boundaries with the
existing `exo-memory` and `exo-context` packages.

---

## 2. Agent-Core Reference Architecture

Agent-core's retrieval system (`openjiuwen/core/retrieval/`) consists of:

| Module | Purpose |
|--------|---------|
| `embedding/` | `OpenAIEmbeddings`, `VLLMEmbeddings`, `APIEmbeddings` |
| `vector_store/` | `Milvus`, `ChromaDB`, `PgVector` backends |
| `retriever/` | 5 retriever types (Vector, Sparse, Hybrid, Graph, Agentic) |
| `reranker/` | `StandardReranker` (vLLM-compatible), `AliyunReranker` |
| `query_rewriter/` | LLM-based query expansion with history compression |
| `indexing/` | Document processing, chunking, format parsers |

Key design choices:

1. **Embedding-agnostic** — vector stores accept any provider implementing
   the embedding interface.
2. **Hybrid search** — Dense vector + sparse BM25 combined via Reciprocal
   Rank Fusion (RRF).
3. **5 retriever types** — from simple vector to multi-round agentic retrieval.
4. **Pluggable chunking** — character, paragraph, token-based, and hybrid strategies.
5. **Document parsers** — PDF, Markdown, JSON, plain text, with extensible format
   support.
6. **Knowledge graph** — Triple extraction (subject-predicate-object) with beam
   search expansion.

---

## 3. Key Decision: Independent Embeddings ABC (Protocol-Compatible)

### Option A — Re-export exo-memory's Embeddings (rejected)

Re-using `exo.memory.backends.vector.Embeddings` directly would create a
dependency from `exo-retrieval` → `exo-memory`. This violates the no
lateral dependencies rule (both are same-level packages).

### Option B — Move Embeddings to exo-core (rejected)

Moving the `Embeddings` ABC into `exo-core` would add embedding concerns to
the core package, which should remain focused on Agent, Tool, and Hook primitives
with zero heavy dependencies.

### Option C — Independent Embeddings ABC in exo-retrieval (chosen)

Create a new `Embeddings` ABC in `exo-retrieval` that is **structurally
compatible** (duck-type / protocol-compatible) with `exo-memory`'s
`Embeddings` class. Both define:

- `embed(text: str) -> list[float]`
- `async aembed(text: str) -> list[float]` (retrieval uses `embed_batch` instead)
- `dimension -> int` property

The retrieval `Embeddings` ABC adds `embed_batch(texts) -> list[list[float]]`
for efficient batch operations needed by indexing pipelines. Callers that hold
either ABC instance can use the shared method signatures interchangeably.

**Why Option C:**
- No lateral dependency between packages.
- Retrieval-specific needs (batch embedding, dimension constraints) stay local.
- Users who install both packages can pass either embedding provider — they share
  the same method signatures via structural subtyping.

---

## 4. Package Layout

```
packages/exo-retrieval/
├── pyproject.toml
├── src/
│   └── exo/
│       ├── __init__.py              # extend_path
│       └── retrieval/
│           ├── __init__.py          # public API re-exports
│           ├── types.py             # Document, Chunk, RetrievalResult, Triple
│           ├── errors.py            # RetrievalError
│           ├── embeddings.py        # Embeddings ABC, OpenAIEmbeddings
│           ├── vector_store.py      # VectorStore ABC, InMemoryVectorStore
│           ├── retriever.py         # Retriever ABC, VectorRetriever
│           ├── sparse.py            # SparseRetriever (BM25)
│           ├── hybrid.py            # HybridRetriever (RRF fusion)
│           ├── graph.py             # GraphRetriever (knowledge graph)
│           ├── agentic.py           # AgenticRetriever (LLM-driven)
│           ├── reranker.py          # Reranker ABC, LLMReranker
│           ├── query_rewriter.py    # QueryRewriter
│           ├── chunker.py           # Chunker ABC + strategies
│           ├── parser.py            # DocumentParser ABC + format parsers
│           ├── tools.py             # Agent tool wrappers for retrieval
│           └── _internal/
│               ├── __init__.py
│               └── scoring.py       # BM25, TF-IDF, RRF helpers
└── tests/
    ├── __init__.py
    ├── test_retrieval_types.py
    ├── test_retrieval_embeddings.py
    ├── test_retrieval_vector_store.py
    ├── test_retrieval_retrievers.py
    ├── test_retrieval_chunker.py
    ├── test_retrieval_parser.py
    ├── test_retrieval_reranker.py
    ├── test_retrieval_query_rewriter.py
    ├── test_retrieval_tools.py
    └── test_retrieval_graph.py
```

---

## 5. Type Hierarchy

### 5.1 Document

```python
class Document(BaseModel):
    """A source document for retrieval."""
    id: str                              # unique identifier (UUID default)
    content: str                         # raw text content
    metadata: dict[str, Any] = {}        # arbitrary metadata (source, author, etc.)
    embedding: list[float] | None = None # optional pre-computed embedding
```

### 5.2 Chunk

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    """A segment of a document's content."""
    document_id: str    # parent document ID
    index: int          # chunk index within document
    content: str        # chunk text
    start: int          # character offset start
    end: int            # character offset end
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.3 RetrievalResult

```python
@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A single retrieval hit with relevance score."""
    chunk: Chunk
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.4 Triple (Knowledge Graph)

```python
@dataclass(frozen=True, slots=True)
class Triple:
    """A subject-predicate-object triple from knowledge graph extraction."""
    subject: str
    predicate: str
    object: str
    source_chunk: Chunk | None = None
    confidence: float = 1.0
```

### 5.5 RetrievalError

```python
class RetrievalError(Exception):
    """Raised for retrieval pipeline errors."""
```

---

## 6. Embedding Abstraction

```python
class Embeddings(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call for efficiency."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The embedding vector dimension."""
```

### Implementations

| Class | Backend | Optional Dep |
|-------|---------|-------------|
| `OpenAIEmbeddings` | OpenAI / Azure / vLLM compat | `openai` |

**OpenAIEmbeddings** wraps the OpenAI embeddings API with configurable:
- `model` (default: `text-embedding-3-small`)
- `dimension` (default: 1536)
- `api_key`, `base_url` for any OpenAI-compatible endpoint
- `embed_batch` sends all texts in a single API call

---

## 7. Vector Store Abstraction

```python
class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search by vector similarity."""

    @abstractmethod
    async def delete(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count removed."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all stored data."""
```

### Implementations

| Class | Backend | Optional Dep | Notes |
|-------|---------|-------------|-------|
| `InMemoryVectorStore` | Pure Python | None | Cosine similarity, dev/testing |
| `PgVectorStore` | PostgreSQL + pgvector | `asyncpg` | Production-grade, nearest-neighbor index |
| `ChromaVectorStore` | ChromaDB | `chromadb` | Local persistent, auto-embedding |

---

## 8. Retriever Types

### 8.1 Base Retriever

```python
class Retriever(ABC):
    """Abstract base class for all retriever types."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
```

### 8.2 VectorRetriever

Pure dense vector search. Embeds query, searches vector store.

```python
class VectorRetriever(Retriever):
    def __init__(self, embeddings: Embeddings, store: VectorStore): ...
```

### 8.3 SparseRetriever

BM25 keyword search over an in-memory inverted index.

```python
class SparseRetriever(Retriever):
    async def index(self, chunks: list[Chunk]) -> None: ...
    # search uses BM25 scoring
```

### 8.4 HybridRetriever

Combines dense + sparse results via Reciprocal Rank Fusion (RRF).

```python
class HybridRetriever(Retriever):
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        sparse_retriever: SparseRetriever,
        *,
        vector_weight: float = 0.5,  # RRF weight balance
    ): ...
```

RRF formula: `score(d) = Σ 1 / (k + rank_i(d))` where `k = 60` (standard constant).

### 8.5 GraphRetriever

Knowledge graph expansion via triple beam search.

```python
class GraphRetriever(Retriever):
    def __init__(
        self,
        base_retriever: Retriever,      # for initial retrieval
        triples: list[Triple],           # knowledge graph
        *,
        max_hops: int = 2,              # expansion depth
        beam_width: int = 5,            # beam search width
    ): ...
```

Retrieves initial chunks, then expands via related triples in the knowledge graph.

### 8.6 AgenticRetriever

Multi-round LLM-driven retrieval with query rewriting and sufficiency judgment.

```python
class AgenticRetriever(Retriever):
    def __init__(
        self,
        base_retriever: Retriever,
        provider: Any,                   # ModelProvider for LLM calls
        *,
        max_rounds: int = 3,
        rewriter: QueryRewriter | None = None,
    ): ...
```

Loop: rewrite query → retrieve → judge sufficiency → repeat or return.

---

## 9. Chunking Strategies

```python
class Chunker(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, *, document_id: str = "") -> list[Chunk]:
        """Split text into chunks."""
```

### Implementations

| Class | Strategy | Notes |
|-------|----------|-------|
| `CharacterChunker` | Fixed-size character windows | Configurable size + overlap |
| `ParagraphChunker` | Split on `\n\n` boundaries | Respects paragraph structure |
| `TokenChunker` | Token-based splitting | Optional dep: `tiktoken` |

All chunkers support configurable `chunk_size` and `chunk_overlap`.

---

## 10. Document Parsers

```python
class DocumentParser(ABC):
    """Abstract base class for document format parsers."""

    @abstractmethod
    def parse(self, content: bytes, *, metadata: dict[str, Any] | None = None) -> Document:
        """Parse raw bytes into a Document."""

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """MIME types this parser handles."""
```

### Implementations

| Class | Formats | Optional Dep |
|-------|---------|-------------|
| `TextParser` | `.txt`, `.md` | None |
| `JsonParser` | `.json` | None |
| `MarkdownParser` | `.md` (structured) | None |

PDF parsing is intentionally deferred to later stories or external integrations.

---

## 11. Reranking

```python
class Reranker(ABC):
    """Abstract base class for result reranking."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results. Returns re-scored and re-sorted results."""
```

### Implementations

| Class | Backend | Notes |
|-------|---------|-------|
| `LLMReranker` | Any ModelProvider | Uses LLM to score relevance |

---

## 12. Query Rewriting

```python
class QueryRewriter:
    """LLM-based query rewriting for improved retrieval."""

    def __init__(self, provider: Any): ...

    async def rewrite(
        self,
        query: str,
        *,
        context: str = "",
        history: list[str] | None = None,
    ) -> str:
        """Rewrite query for better retrieval coverage."""
```

Handles:
- Ambiguous queries — expand with context.
- History compression — fold conversation history into focused query.
- Multi-aspect queries — decompose into sub-queries.

---

## 13. Agent Tool Integration

Retrieval capabilities are exposed as Exo tools:

```python
def retrieval_tool(retriever: Retriever) -> Tool:
    """Create an Agent-compatible tool from a retriever."""

def index_tool(chunker: Chunker, embeddings: Embeddings, store: VectorStore) -> Tool:
    """Create an Agent-compatible tool for document indexing."""
```

These tools follow the `exo.tool` pattern, allowing agents to use retrieval
as part of their tool loop. The tools bridge the gap between the retrieval pipeline
and the agent execution model.

---

## 14. Optional Dependencies

The package uses optional extras for heavy dependencies:

```toml
[project.optional-dependencies]
openai = ["openai>=1.0"]
tiktoken = ["tiktoken>=0.5"]
chromadb = ["chromadb>=0.4"]
pgvector = ["asyncpg>=0.29", "pgvector>=0.2"]
all = ["openai>=1.0", "tiktoken>=0.5", "chromadb>=0.4", "asyncpg>=0.29", "pgvector>=0.2"]
```

Core types, ABCs, `InMemoryVectorStore`, and `CharacterChunker` work with
zero optional dependencies — only `pydantic` and `exo-core` are required.

---

## 15. Relationship to Existing Packages

### exo-memory (no changes)

- `exo.memory.backends.vector.Embeddings` remains unchanged.
- `VectorMemoryStore` continues to serve memory-specific use cases.
- Users who want to use memory embeddings with retrieval can pass them via
  structural compatibility (both share `embed()` + `dimension`).

### exo-context (no changes)

- `KnowledgeStore` continues to provide workspace artifact indexing.
- `exo-retrieval` does not replace `KnowledgeStore` — they serve different
  purposes (document retrieval vs. workspace artifact search).
- Future integration: exo-context could use an `exo-retrieval` retriever
  as a backend, but this is not part of Epic 3.

### exo-core (no changes)

- `exo-retrieval` depends on `exo-core` for the `Tool` type.
- No changes to Agent, HookManager, or any core types.

---

## 16. Implementation Order

The 17 implementation stories (US-034 through US-050) follow this dependency order:

1. **US-034** — Package scaffold + core types (`Document`, `Chunk`, `RetrievalResult`)
2. **US-035** — `Embeddings` ABC + `OpenAIEmbeddings`
3. **US-036** — `VectorStore` ABC + `InMemoryVectorStore`
4. **US-037** — `PgVectorStore` (asyncpg + pgvector)
5. **US-038** — `ChromaVectorStore`
6. **US-039** — `Retriever` ABC + `VectorRetriever`
7. **US-040** — `SparseRetriever` (BM25)
8. **US-041** — `HybridRetriever` (RRF fusion)
9. **US-042** — `Reranker` ABC + `LLMReranker`
10. **US-043** — `Chunker` ABC + strategies (Character, Paragraph, Token)
11. **US-044** — `DocumentParser` ABC + format parsers
12. **US-045** — `QueryRewriter`
13. **US-046** — `AgenticRetriever`
14. **US-047** — Triple extraction + `GraphRetriever`
15. **US-048** — Agent tool wrappers
16. **US-049** — Knowledge graph retriever (triple beam search)
17. **US-050** — Integration tests + `__init__.py` re-exports

Each story builds on the previous, with types and ABCs established first,
concrete implementations following, and integration tests last.
