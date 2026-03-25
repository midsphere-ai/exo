# RAG/Retrieval Pipeline — agent-core to Exo Mapping

**Epic:** 3 — RAG/Retrieval Pipeline
**Date:** 2026-03-11

This document maps agent-core's (openJiuwen) RAG pipeline to Exo's
`exo-retrieval` package, helping contributors familiar with either
framework navigate both.

---

## 1. Agent-Core Overview

Agent-core's RAG system lives in `openjiuwen/core/rag/` and provides
a full retrieval-augmented generation pipeline covering document
processing, embedding, vector storage, multiple retrieval strategies,
reranking, and knowledge graph construction.

### Key Components

**`EmbeddingProvider` ABC** — Base class for turning text into dense
vectors. Three built-in implementations:

| Provider | Backend |
|----------|---------|
| `OpenAIEmbeddingProvider` | OpenAI embeddings API |
| `VertexEmbeddingProvider` | Google Vertex AI text-embedding |
| `HTTPEmbeddingProvider` | Any HTTP endpoint returning vectors |

```python
# agent-core pattern
class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    @abstractmethod
    def dimension(self) -> int: ...
```

**`VectorStore` ABC** — Persistence layer for chunks and their
embeddings. Supports add, search (by embedding), delete, and clear.
Backends include pgvector (PostgreSQL) and ChromaDB.

**5 Retriever Types:**

| Retriever | Strategy |
|-----------|----------|
| `VectorRetriever` | Dense semantic search via embedding similarity |
| `SparseRetriever` | BM25 keyword matching with TF-IDF scoring |
| `HybridRetriever` | Reciprocal Rank Fusion of dense + sparse results |
| `AgenticRetriever` | Multi-round LLM-driven query refinement |
| `KnowledgeGraphRetriever` | Graph traversal expanding initial results via triples |

**`Reranker` ABC** — Reorders retrieval results by relevance. The
built-in `LLMReranker` sends passages to an LLM that returns a ranked
ordering.

**Document Processing Pipeline** — Converts raw documents into indexed
chunks:
1. **Parser** — extracts text from source formats (plain text, Markdown,
   JSON, PDF)
2. **Chunker** — splits text into overlapping or paragraph-aligned
   segments
3. **EmbeddingProvider** — vectorizes each chunk
4. **VectorStore** — persists chunks with their embeddings

**`QueryRewriter`** — LLM-based query expansion that adds synonyms,
resolves pronouns from conversation history, and disambiguates terms
before retrieval.

**Knowledge Graph Support** — `TripleExtractor` uses an LLM to extract
subject–predicate–object triples from chunks, enabling the
`KnowledgeGraphRetriever` to traverse relationships.

---

## 2. Exo Equivalent

Exo's RAG system lives in the `exo-retrieval` package
(`packages/exo-retrieval/`) as a separate installable package.

### Mapping Summary

| Agent-Core | Exo | Notes |
|------------|---------|-------|
| `EmbeddingProvider` ABC | `Embeddings` ABC | Renamed; same `embed()`, `embed_batch()`, `dimension` interface |
| `OpenAIEmbeddingProvider` | `OpenAIEmbeddings` | Uses httpx directly (no SDK dependency) |
| `VertexEmbeddingProvider` | `VertexEmbeddings` | Configurable location & output dimensionality |
| `HTTPEmbeddingProvider` | `HTTPEmbeddings` | Dot-path field extraction for flexible API shapes |
| `VectorStore` ABC | `VectorStore` ABC | Same interface: `add()`, `search()`, `delete()`, `clear()` |
| pgvector backend | `PgVectorStore` | asyncpg-based; in `backends/pgvector.py` |
| ChromaDB backend | `ChromaVectorStore` | Supports persistent and ephemeral modes; in `backends/chroma.py` |
| — | `InMemoryVectorStore` | New: pure-Python store for dev/testing |
| `VectorRetriever` | `VectorRetriever` | Same pattern: embed query → search store |
| `SparseRetriever` / BM25 | `SparseRetriever` | Pure-Python BM25 with configurable k1 and b |
| `HybridRetriever` / RRF | `HybridRetriever` | Concurrent dense+sparse with weighted RRF fusion |
| `AgenticRetriever` | `AgenticRetriever` | Multi-round with sufficiency threshold judging |
| `KnowledgeGraphRetriever` | `GraphRetriever` | Renamed; beam-search traversal with hop decay |
| `Reranker` ABC | `Reranker` ABC | Same `rerank()` interface |
| `LLMReranker` | `LLMReranker` | JSON index parsing with fallback handling |
| `QueryRewriter` | `QueryRewriter` | History-aware rewriting via LLM |
| Document / Chunk types | `Document` / `Chunk` | Pydantic models with metadata dicts |
| retrieval result | `RetrievalResult` | Immutable: `chunk`, `score`, `metadata` |
| retrieval error | `RetrievalError` | Carries `operation` and `details` |
| *(inline in retrievers)* | `Chunker` ABC | New: extracted text chunking hierarchy |
| — | `CharacterChunker` | Fixed-size with overlap |
| — | `ParagraphChunker` | Splits at blank lines |
| — | `TokenChunker` | tiktoken-based token counting |
| *(inline parsers)* | `Parser` ABC | New: extracted document parsing hierarchy |
| — | `TextParser`, `MarkdownParser`, `JSONParser`, `PDFParser` | Format-specific extractors |
| Triple extraction | `TripleExtractor` + `Triple` | LLM-based knowledge graph construction |
| *(no equivalent)* | `retrieve_tool()` / `index_tool()` | New: agent `FunctionTool` factories for retrieval |

### Architecture Difference

Agent-core bundles RAG inside the core framework. Exo extracts it
into a standalone package (`exo-retrieval`) that depends only on
`exo-core` for the `FunctionTool` type used by the agent integration
helpers. All LLM calls go through `exo.models.get_provider()`,
keeping the retrieval package model-agnostic.

All embedding and retriever methods are **async-first**. Chunkers and
parsers are synchronous since they operate on local data.

---

## 3. Side-by-Side Code Examples

### Building a Hybrid Retrieval Pipeline

**Agent-core:**

```python
from openjiuwen.core.rag import (
    OpenAIEmbeddingProvider,
    VectorStore,
    VectorRetriever,
    SparseRetriever,
    HybridRetriever,
)

embeddings = OpenAIEmbeddingProvider(api_key="sk-...", model="text-embedding-3-small")
store = VectorStore.create("pgvector", dsn="postgresql://...")
dense = VectorRetriever(embeddings=embeddings, store=store)
sparse = SparseRetriever()
sparse.index(chunks)
hybrid = HybridRetriever(dense=dense, sparse=sparse, vector_weight=0.6)

results = await hybrid.retrieve("How does authentication work?", top_k=10)
```

**Exo:**

```python
from exo.retrieval import (
    OpenAIEmbeddings,
    HybridRetriever,
    VectorRetriever,
    SparseRetriever,
)
from exo.retrieval.backends.pgvector import PgVectorStore

embeddings = OpenAIEmbeddings(api_key="sk-...", model="text-embedding-3-small")
store = PgVectorStore(dsn="postgresql://...", dimensions=1536)
await store.initialize()

dense = VectorRetriever(embeddings=embeddings, store=store)
sparse = SparseRetriever()
await sparse.index(chunks)
hybrid = HybridRetriever(
    vector_retriever=dense,
    sparse_retriever=sparse,
    vector_weight=0.6,
)

results = await hybrid.retrieve("How does authentication work?", top_k=10)
```

### Document Ingestion Pipeline

**Agent-core:**

```python
from openjiuwen.core.rag import (
    OpenAIEmbeddingProvider,
    VectorStore,
    Document,
)

embeddings = OpenAIEmbeddingProvider(api_key="sk-...")
store = VectorStore.create("chroma", path="./chroma_data")

doc = Document(id="doc-1", content=open("paper.txt").read())
chunks = doc.chunk(size=500, overlap=50)  # inline chunking
vectors = await embeddings.embed_batch([c.content for c in chunks])
await store.add(chunks, vectors)
```

**Exo:**

```python
from exo.retrieval import (
    OpenAIEmbeddings,
    Document,
    CharacterChunker,
)
from exo.retrieval.backends.chroma import ChromaVectorStore

embeddings = OpenAIEmbeddings(api_key="sk-...")
store = ChromaVectorStore(collection_name="papers", path="./chroma_data")
chunker = CharacterChunker(chunk_size=500, chunk_overlap=50)

doc = Document(id="doc-1", content=open("paper.txt").read(), metadata={})
chunks = chunker.chunk(doc)
vectors = await embeddings.embed_batch([c.content for c in chunks])
await store.add(chunks, vectors)
```

### Agentic Retrieval with Query Rewriting

**Agent-core:**

```python
from openjiuwen.core.rag import (
    AgenticRetriever,
    QueryRewriter,
    VectorRetriever,
)

rewriter = QueryRewriter(model="gpt-4o")
base = VectorRetriever(embeddings=emb, store=store)
agentic = AgenticRetriever(
    base_retriever=base,
    rewriter=rewriter,
    model="gpt-4o",
    max_rounds=3,
)
results = await agentic.retrieve("What are the auth options?")
```

**Exo:**

```python
from exo.retrieval import (
    AgenticRetriever,
    QueryRewriter,
    VectorRetriever,
)

rewriter = QueryRewriter(model="openai:gpt-4o")
base = VectorRetriever(embeddings=emb, store=store)
agentic = AgenticRetriever(
    base_retriever=base,
    rewriter=rewriter,
    model="openai:gpt-4o",  # provider:model format
    max_rounds=3,
    sufficiency_threshold=0.7,
)
results = await agentic.retrieve("What are the auth options?")
```

### Giving an Agent Retrieval Tools

```python
from exo.retrieval import (
    retrieve_tool,
    index_tool,
    VectorRetriever,
    CharacterChunker,
    OpenAIEmbeddings,
    InMemoryVectorStore,
)
from exo.agent import Agent

embeddings = OpenAIEmbeddings(api_key="sk-...")
store = InMemoryVectorStore()
retriever = VectorRetriever(embeddings=embeddings, store=store)
chunker = CharacterChunker()

agent = Agent(
    name="rag-agent",
    model="openai:gpt-4o",
    tools=[
        retrieve_tool(retriever),
        index_tool(chunker, store, embeddings),
    ],
)
```

---

## 4. Migration Table

| Agent-Core Path | Exo Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.core.rag.EmbeddingProvider` | `exo.retrieval.embeddings.Embeddings` | ABC: `embed()`, `embed_batch()`, `dimension` |
| `openjiuwen.core.rag.OpenAIEmbeddingProvider` | `exo.retrieval.openai_embeddings.OpenAIEmbeddings` | httpx-based, no SDK |
| `openjiuwen.core.rag.VertexEmbeddingProvider` | `exo.retrieval.vertex_embeddings.VertexEmbeddings` | GCP Vertex AI |
| `openjiuwen.core.rag.HTTPEmbeddingProvider` | `exo.retrieval.http_embeddings.HTTPEmbeddings` | Generic HTTP endpoint |
| `openjiuwen.core.rag.VectorStore` | `exo.retrieval.vector_store.VectorStore` | ABC: `add()`, `search()`, `delete()`, `clear()` |
| `openjiuwen.core.rag.PgVectorStore` | `exo.retrieval.backends.pgvector.PgVectorStore` | asyncpg + pgvector |
| `openjiuwen.core.rag.ChromaStore` | `exo.retrieval.backends.chroma.ChromaVectorStore` | Persistent or ephemeral |
| *(no equivalent)* | `exo.retrieval.vector_store.InMemoryVectorStore` | Pure-Python dev/test store |
| `openjiuwen.core.rag.VectorRetriever` | `exo.retrieval.retriever.VectorRetriever` | Dense semantic search |
| `openjiuwen.core.rag.SparseRetriever` | `exo.retrieval.sparse_retriever.SparseRetriever` | BM25 keyword matching |
| `openjiuwen.core.rag.HybridRetriever` | `exo.retrieval.hybrid_retriever.HybridRetriever` | RRF fusion |
| `openjiuwen.core.rag.AgenticRetriever` | `exo.retrieval.agentic_retriever.AgenticRetriever` | Multi-round LLM-driven |
| `openjiuwen.core.rag.KnowledgeGraphRetriever` | `exo.retrieval.graph_retriever.GraphRetriever` | Beam-search graph traversal |
| `openjiuwen.core.rag.Reranker` | `exo.retrieval.reranker.Reranker` | ABC: `rerank()` |
| `openjiuwen.core.rag.LLMReranker` | `exo.retrieval.reranker.LLMReranker` | LLM-based passage ranking |
| `openjiuwen.core.rag.QueryRewriter` | `exo.retrieval.query_rewriter.QueryRewriter` | LLM query expansion |
| `openjiuwen.core.rag.Document` | `exo.retrieval.types.Document` | Pydantic model |
| `openjiuwen.core.rag.Chunk` | `exo.retrieval.types.Chunk` | Immutable chunk slice |
| *(inline result type)* | `exo.retrieval.types.RetrievalResult` | Scored chunk with metadata |
| *(inline error)* | `exo.retrieval.types.RetrievalError` | `operation` + `details` |
| *(inline chunking)* | `exo.retrieval.chunker.Chunker` | ABC: `chunk(document)` |
| *(inline chunking)* | `exo.retrieval.chunker.CharacterChunker` | Fixed-size with overlap |
| *(inline chunking)* | `exo.retrieval.chunker.ParagraphChunker` | Blank-line splitting |
| *(inline chunking)* | `exo.retrieval.chunker.TokenChunker` | tiktoken-based |
| *(inline parsing)* | `exo.retrieval.parsers.Parser` | ABC: `parse(source)` |
| *(inline parsing)* | `exo.retrieval.parsers.TextParser` | Passthrough |
| *(inline parsing)* | `exo.retrieval.parsers.MarkdownParser` | Strip formatting |
| *(inline parsing)* | `exo.retrieval.parsers.JSONParser` | Flatten to key-paths |
| *(inline parsing)* | `exo.retrieval.parsers.PDFParser` | pymupdf extraction |
| `openjiuwen.core.rag.TripleExtractor` | `exo.retrieval.triple_extractor.TripleExtractor` | LLM knowledge-graph extraction |
| `openjiuwen.core.rag.Triple` | `exo.retrieval.triple_extractor.Triple` | Frozen dataclass |
| *(no equivalent)* | `exo.retrieval.tools.retrieve_tool` | `FunctionTool` factory for retrieval |
| *(no equivalent)* | `exo.retrieval.tools.index_tool` | `FunctionTool` factory for indexing |

All public symbols are also re-exported from `exo.retrieval` (the
package `__init__.py`), so `from exo.retrieval import VectorRetriever`
works as a convenience import.
