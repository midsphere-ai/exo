"""Tests for VectorMemoryStore and Embeddings ABC."""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import pytest

from exo.memory.backends.vector import (  # pyright: ignore[reportMissingImports]
    ChromaVectorMemoryStore,
    EmbeddingProvider,
    Embeddings,
    OpenAIEmbeddingProvider,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddingProvider,
    VectorMemoryStore,
    VertexEmbeddings,
    _build_where_filter,
    _cosine_similarity,
)
from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    HumanMemory,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
    SystemMemory,
)

# ---------------------------------------------------------------------------
# Mock embeddings — deterministic vectors for testing
# ---------------------------------------------------------------------------


class MockEmbeddings(Embeddings):
    """Deterministic embeddings for testing.

    Generates a simple vector based on the character-code average of the text.
    """

    __slots__ = ("_dim", "call_count")

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if not text:
            return [0.0] * self._dim
        # Simple deterministic embedding: use char codes
        base = sum(ord(c) for c in text) / len(text) / 256.0
        return [base + i * 0.1 for i in range(self._dim)]

    async def aembed(self, text: str) -> list[float]:
        return self.embed(text)


class FixedEmbeddings(Embeddings):
    """Returns pre-set vectors based on a lookup table."""

    __slots__ = ("_dim", "_vectors")

    def __init__(self, vectors: dict[str, list[float]], dim: int = 3) -> None:
        self._vectors = vectors
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return self._vectors.get(text, [0.0] * self._dim)

    async def aembed(self, text: str) -> list[float]:
        return self.embed(text)


# ---------------------------------------------------------------------------
# Embeddings ABC tests
# ---------------------------------------------------------------------------


class TestEmbeddingsABC:
    """Tests for the Embeddings abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Embeddings()  # type: ignore[abstract]

    def test_mock_implements_abc(self) -> None:
        emb = MockEmbeddings(dim=8)
        assert isinstance(emb, Embeddings)
        assert emb.dimension == 8

    def test_mock_embed_sync(self) -> None:
        emb = MockEmbeddings(dim=3)
        vec = emb.embed("hello")
        assert len(vec) == 3
        assert all(isinstance(v, float) for v in vec)

    async def test_mock_embed_async(self) -> None:
        emb = MockEmbeddings(dim=3)
        vec = await emb.aembed("hello")
        assert len(vec) == 3

    def test_empty_text(self) -> None:
        emb = MockEmbeddings(dim=4)
        vec = emb.embed("")
        assert vec == [0.0, 0.0, 0.0, 0.0]


class TestOpenAIEmbeddings:
    """Tests for OpenAIEmbeddings (construction only — no real API calls)."""

    def test_is_embeddings_subclass(self) -> None:
        assert issubclass(OpenAIEmbeddings, Embeddings)

    def test_dimension_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch openai import
        import types

        mock_openai = types.ModuleType("openai")
        mock_openai.OpenAI = lambda **kw: None  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "openai", mock_openai)

        emb = OpenAIEmbeddings(model="test", dimension=768, api_key="fake")
        assert emb.dimension == 768


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the _cosine_similarity helper."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_known_value(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
        na = math.sqrt(14)
        nb = math.sqrt(77)
        expected = dot / (na * nb)
        assert _cosine_similarity(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# VectorMemoryStore — protocol conformance
# ---------------------------------------------------------------------------


class TestVectorProtocol:
    """VectorMemoryStore satisfies the MemoryStore protocol."""

    def test_isinstance_check(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------------------
# VectorMemoryStore — lifecycle
# ---------------------------------------------------------------------------


class TestVectorLifecycle:
    def test_init(self) -> None:
        emb = MockEmbeddings(dim=4)
        store = VectorMemoryStore(emb)
        assert store.embeddings is emb
        assert len(store) == 0

    def test_repr(self) -> None:
        store = VectorMemoryStore(MockEmbeddings(dim=8))
        assert "VectorMemoryStore" in repr(store)
        assert "dimension=8" in repr(store)


# ---------------------------------------------------------------------------
# VectorMemoryStore — add / get
# ---------------------------------------------------------------------------


class TestVectorAddGet:
    async def test_add_and_get(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(content="hello world")
        await store.add(item)
        assert len(store) == 1
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "hello world"

    async def test_get_nonexistent(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        assert await store.get("nope") is None

    async def test_add_computes_embedding(self) -> None:
        emb = MockEmbeddings(dim=3)
        store = VectorMemoryStore(emb)
        item = HumanMemory(content="test")
        await store.add(item)
        assert emb.call_count == 1
        assert item.id in store._vectors
        assert len(store._vectors[item.id]) == 3

    async def test_add_multiple(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        for i in range(5):
            await store.add(HumanMemory(content=f"item {i}"))
        assert len(store) == 5

    async def test_upsert(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(id="x", content="v1")
        await store.add(item)
        updated = HumanMemory(id="x", content="v2")
        await store.add(updated)
        assert len(store) == 1
        got = await store.get("x")
        assert got is not None
        assert got.content == "v2"


# ---------------------------------------------------------------------------
# VectorMemoryStore — search (semantic)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    async def test_semantic_ranking(self) -> None:
        """Items closer to query in embedding space rank higher."""
        vecs = {
            "cats are pets": [1.0, 0.0, 0.0],
            "dogs are pets": [0.9, 0.1, 0.0],
            "quantum physics": [0.0, 0.0, 1.0],
            "search for pets": [0.95, 0.05, 0.0],
        }
        emb = FixedEmbeddings(vecs, dim=3)
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="cats are pets"))
        await store.add(HumanMemory(content="dogs are pets"))
        await store.add(HumanMemory(content="quantum physics"))

        results = await store.search(query="search for pets", limit=3)
        contents = [r.content for r in results]
        # Pets-related should rank before quantum physics
        assert contents.index("cats are pets") < contents.index("quantum physics")
        assert contents.index("dogs are pets") < contents.index("quantum physics")

    async def test_search_no_query_returns_newest(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item1 = HumanMemory(content="first", created_at="2024-01-01T00:00:00")
        item2 = HumanMemory(content="second", created_at="2024-01-02T00:00:00")
        await store.add(item1)
        await store.add(item2)
        results = await store.search()
        assert results[0].content == "second"

    async def test_search_limit(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        for i in range(10):
            await store.add(HumanMemory(content=f"item {i}"))
        results = await store.search(limit=3)
        assert len(results) == 3

    async def test_search_empty_store(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        results = await store.search(query="anything")
        assert results == []

    async def test_search_by_memory_type(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        await store.add(HumanMemory(content="user msg"))
        await store.add(SystemMemory(content="sys msg"))
        results = await store.search(memory_type="human")
        assert len(results) == 1
        assert results[0].memory_type == "human"

    async def test_search_by_status(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(content="draft item", status=MemoryStatus.DRAFT)
        await store.add(item)
        await store.add(HumanMemory(content="accepted item"))
        results = await store.search(status=MemoryStatus.DRAFT)
        assert len(results) == 1
        assert results[0].content == "draft item"

    async def test_search_by_metadata(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        meta = MemoryMetadata(user_id="u1", session_id="s1")
        await store.add(HumanMemory(content="user1", metadata=meta))
        await store.add(HumanMemory(content="user2", metadata=MemoryMetadata(user_id="u2")))
        results = await store.search(metadata=MemoryMetadata(user_id="u1"))
        assert len(results) == 1
        assert results[0].content == "user1"

    async def test_search_combined_filters(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        meta = MemoryMetadata(user_id="u1")
        await store.add(HumanMemory(content="human u1", metadata=meta))
        await store.add(SystemMemory(content="system u1", metadata=meta))
        await store.add(HumanMemory(content="human u2", metadata=MemoryMetadata(user_id="u2")))
        results = await store.search(
            memory_type="human",
            metadata=MemoryMetadata(user_id="u1"),
        )
        assert len(results) == 1
        assert results[0].content == "human u1"


# ---------------------------------------------------------------------------
# VectorMemoryStore — clear
# ---------------------------------------------------------------------------


class TestVectorClear:
    async def test_clear_all(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        await store.add(HumanMemory(content="a"))
        await store.add(HumanMemory(content="b"))
        count = await store.clear()
        assert count == 2
        assert len(store) == 0

    async def test_clear_with_metadata(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        meta1 = MemoryMetadata(user_id="u1")
        meta2 = MemoryMetadata(user_id="u2")
        await store.add(HumanMemory(content="a", metadata=meta1))
        await store.add(HumanMemory(content="b", metadata=meta2))
        count = await store.clear(metadata=MemoryMetadata(user_id="u1"))
        assert count == 1
        assert len(store) == 1
        remaining = await store.search()
        assert remaining[0].content == "b"

    async def test_clear_removes_vectors(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(content="test")
        await store.add(item)
        assert item.id in store._vectors
        await store.clear()
        assert item.id not in store._vectors

    async def test_clear_empty(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        count = await store.clear()
        assert count == 0


# ---------------------------------------------------------------------------
# VectorMemoryStore — embedding call tracking
# ---------------------------------------------------------------------------


class TestVectorEmbeddingCalls:
    async def test_add_calls_embed(self) -> None:
        emb = MockEmbeddings()
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="a"))
        await store.add(HumanMemory(content="b"))
        assert emb.call_count == 2

    async def test_search_calls_embed_for_query(self) -> None:
        emb = MockEmbeddings()
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="stored"))
        emb.call_count = 0  # reset
        await store.search(query="find me")
        assert emb.call_count == 1  # only the query is embedded

    async def test_search_no_query_no_embed(self) -> None:
        emb = MockEmbeddings()
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="stored"))
        emb.call_count = 0
        await store.search()  # no query
        assert emb.call_count == 0


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol
# ---------------------------------------------------------------------------


class TestEmbeddingProvider:
    """Tests for the EmbeddingProvider runtime-checkable protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """isinstance() checks work without explicit inheritance."""

        class MyProvider:
            async def embed(self, text: str) -> list[float]:
                return [0.0]

        assert isinstance(MyProvider(), EmbeddingProvider)

    def test_missing_embed_not_provider(self) -> None:
        class NotProvider:
            def generate(self, text: str) -> list[float]:  # wrong method name
                return [0.0]

        assert not isinstance(NotProvider(), EmbeddingProvider)

    def test_sync_embed_not_provider(self) -> None:
        """Protocol requires async embed, but isinstance only checks name presence."""

        # runtime_checkable only checks method existence, not signature
        class SyncOnly:
            def embed(self, text: str) -> list[float]:
                return [0.0]

        # Python's runtime_checkable only checks attribute presence
        assert isinstance(SyncOnly(), EmbeddingProvider)


# ---------------------------------------------------------------------------
# OpenAIEmbeddingProvider
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider (mocked openai calls)."""

    def test_satisfies_protocol(self) -> None:
        assert issubclass(OpenAIEmbeddingProvider, object)  # not required to inherit
        # Check it has async embed method
        import inspect

        assert inspect.iscoroutinefunction(OpenAIEmbeddingProvider.embed)

    def test_isinstance_embedding_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAIEmbeddingProvider satisfies EmbeddingProvider protocol."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_reads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")
        provider = OpenAIEmbeddingProvider()
        assert provider._api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        provider = OpenAIEmbeddingProvider(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    async def test_embed_calls_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """embed() calls the OpenAI embeddings API and returns the vector."""
        expected_vec = [0.1, 0.2, 0.3]

        mock_openai = types.ModuleType("openai")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=expected_vec)]
        mock_client.embeddings.create.return_value = mock_resp
        mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="k")
        result = await provider.embed("hello world")

        assert result == expected_vec
        mock_client.embeddings.create.assert_called_once_with(
            input="hello world", model="text-embedding-3-small"
        )


# ---------------------------------------------------------------------------
# SentenceTransformerEmbeddingProvider
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbeddingProvider:
    """Tests for SentenceTransformerEmbeddingProvider (mocked sentence-transformers)."""

    def _mock_sentence_transformers(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Inject a mock sentence_transformers module."""
        mock_st = types.ModuleType("sentence_transformers")
        mock_model = MagicMock()
        # Return an object with a .tolist() method, simulating a numpy array
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.5, 0.6, 0.7]
        mock_model.encode.return_value = mock_array
        mock_st.SentenceTransformer = MagicMock(return_value=mock_model)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
        return mock_model

    def test_satisfies_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._mock_sentence_transformers(monkeypatch)
        provider = SentenceTransformerEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_loads_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_st_mod = types.ModuleType("sentence_transformers")
        mock_st_class = MagicMock()
        mock_st_mod.SentenceTransformer = mock_st_class  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st_mod)

        SentenceTransformerEmbeddingProvider()
        mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")

    async def test_embed_returns_float_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_model = self._mock_sentence_transformers(monkeypatch)
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_array

        provider = SentenceTransformerEmbeddingProvider()
        result = await provider.embed("test text")

        assert result == pytest.approx([0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# ChromaVectorMemoryStore helpers
# ---------------------------------------------------------------------------


class TestBuildWhereFilter:
    """Tests for _build_where_filter helper."""

    def test_none_inputs_returns_none(self) -> None:
        assert _build_where_filter(None, None, None) is None

    def test_memory_type_only(self) -> None:
        result = _build_where_filter(None, "human", None)
        assert result == {"memory_type": {"$eq": "human"}}

    def test_status_only(self) -> None:
        result = _build_where_filter(None, None, MemoryStatus.DRAFT)
        assert result == {"status": {"$eq": "draft"}}

    def test_metadata_user_id(self) -> None:
        result = _build_where_filter(MemoryMetadata(user_id="u1"), None, None)
        assert result == {"user_id": {"$eq": "u1"}}

    def test_multiple_clauses_returns_and(self) -> None:
        result = _build_where_filter(MemoryMetadata(user_id="u1"), "human", None)
        assert result is not None
        assert "$and" in result
        clauses = result["$and"]
        assert {"memory_type": {"$eq": "human"}} in clauses
        assert {"user_id": {"$eq": "u1"}} in clauses

    def test_all_metadata_fields(self) -> None:
        meta = MemoryMetadata(user_id="u1", session_id="s1", task_id="t1", agent_id="a1")
        result = _build_where_filter(meta, None, None)
        assert result is not None
        assert "$and" in result
        assert len(result["$and"]) == 4

    def test_empty_metadata_returns_none(self) -> None:
        result = _build_where_filter(MemoryMetadata(), None, None)
        assert result is None


# ---------------------------------------------------------------------------
# ChromaVectorMemoryStore — with mocked ChromaDB
# ---------------------------------------------------------------------------


def _make_chroma_mock(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Inject a mock chromadb module; return (mock_client, mock_collection)."""
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_chromadb = types.ModuleType("chromadb")
    mock_chromadb.EphemeralClient = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_chromadb.PersistentClient = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "chromadb", mock_chromadb)
    return mock_client, mock_collection


class SimpleEmbeddingProvider:
    """Trivial EmbeddingProvider for tests — returns a fixed 3-dim vector."""

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class TestChromaVectorMemoryStore:
    """Tests for ChromaVectorMemoryStore with mocked ChromaDB."""

    def test_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_chroma_mock(monkeypatch)
        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        assert "ChromaVectorMemoryStore" in repr(store)
        assert "exo_memory" in repr(store)

    async def test_add_calls_collection_add(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        item = HumanMemory(content="hello")
        await store.add(item)

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args
        assert call_kwargs.kwargs["ids"] == [item.id]
        assert call_kwargs.kwargs["documents"] == ["hello"]
        assert call_kwargs.kwargs["embeddings"] == [[0.1, 0.2, 0.3]]

    async def test_get_returns_none_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        result = await store.get("nonexistent")
        assert result is None

    async def test_get_returns_item(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        item = HumanMemory(content="stored content")
        chroma_meta = {
            "memory_type": "human",
            "status": "accepted",
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "user_id": "",
            "session_id": "",
            "task_id": "",
            "agent_id": "",
            "extra_json": "{}",
        }
        mock_collection.get.return_value = {
            "ids": [item.id],
            "documents": ["stored content"],
            "metadatas": [chroma_meta],
        }

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        result = await store.get(item.id)
        assert result is not None
        assert result.content == "stored content"
        assert result.memory_type == "human"

    async def test_search_with_query_calls_collection_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        results = await store.search(query="find this", limit=5)

        mock_collection.query.assert_called_once()
        query_kwargs = mock_collection.query.call_args.kwargs
        assert query_kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]
        assert query_kwargs["n_results"] == 5
        assert results == []

    async def test_search_no_query_calls_collection_get(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        results = await store.search()

        mock_collection.get.assert_called()
        assert results == []

    async def test_search_returns_reconstructed_items(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        item = HumanMemory(content="test item")
        chroma_meta = {
            "memory_type": "human",
            "status": "accepted",
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "user_id": "",
            "session_id": "",
            "task_id": "",
            "agent_id": "",
            "extra_json": "{}",
        }
        mock_collection.query.return_value = {
            "ids": [[item.id]],
            "documents": [["test item"]],
            "metadatas": [[chroma_meta]],
        }

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        results = await store.search(query="test", limit=5)
        assert len(results) == 1
        assert results[0].content == "test item"

    async def test_clear_all_deletes_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        # First get() (in clear) returns two items; subsequent calls return empty
        mock_collection.get.return_value = {"ids": ["id1", "id2"], "documents": [], "metadatas": []}
        new_collection = MagicMock()
        # First call to get_or_create_collection returns mock_collection (lazy init)
        # Second call (after delete_collection in clear()) returns new_collection
        mock_client.get_or_create_collection.side_effect = [mock_collection, new_collection]

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        count = await store.clear()
        assert count == 2
        mock_client.delete_collection.assert_called_once_with("exo_memory")

    async def test_clear_with_metadata_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        mock_collection.get.return_value = {
            "ids": ["id1"],
            "documents": [],
            "metadatas": [],
        }

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        count = await store.clear(metadata=MemoryMetadata(user_id="u1"))
        assert count == 1
        mock_collection.delete.assert_called_once_with(ids=["id1"])

    async def test_get_recent_returns_sorted_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        older = HumanMemory(content="older", created_at="2024-01-01T00:00:00")
        newer = HumanMemory(content="newer", created_at="2024-06-01T00:00:00")

        def _make_meta(item: HumanMemory) -> dict:
            return {
                "memory_type": "human",
                "status": "accepted",
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "user_id": "",
                "session_id": "",
                "task_id": "",
                "agent_id": "",
                "extra_json": "{}",
            }

        mock_collection.get.return_value = {
            "ids": [older.id, newer.id],
            "documents": ["older", "newer"],
            "metadatas": [_make_meta(older), _make_meta(newer)],
        }

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        results = await store.get_recent(n=5)
        assert len(results) == 2
        assert results[0].content == "newer"  # newest first
        assert results[1].content == "older"

    async def test_get_recent_respects_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)

        def _make_meta(ts: str) -> dict:
            return {
                "memory_type": "human",
                "status": "accepted",
                "created_at": ts,
                "updated_at": ts,
                "user_id": "",
                "session_id": "",
                "task_id": "",
                "agent_id": "",
                "extra_json": "{}",
            }

        ids = [f"id{i}" for i in range(5)]
        docs = [f"item{i}" for i in range(5)]
        metas = [_make_meta(f"2024-0{i + 1}-01T00:00:00") for i in range(5)]
        mock_collection.get.return_value = {"ids": ids, "documents": docs, "metadatas": metas}

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        results = await store.get_recent(n=2)
        assert len(results) == 2

    async def test_search_applies_metadata_where_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        await store.search(query="q", metadata=MemoryMetadata(user_id="u42"), limit=3)

        query_kwargs = mock_collection.query.call_args.kwargs
        assert query_kwargs.get("where") == {"user_id": {"$eq": "u42"}}

    async def test_uses_persistent_client_when_path_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_client, _mock_collection = _make_chroma_mock(monkeypatch)
        import chromadb

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider(), path="/tmp/testdb")
        store._ensure_collection()
        chromadb.PersistentClient.assert_called_once_with(path="/tmp/testdb")  # type: ignore[attr-defined]

    async def test_add_and_search_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: add an item, then search returns it (mocked collection)."""
        _mock_client, mock_collection = _make_chroma_mock(monkeypatch)

        item = HumanMemory(content="round trip content")
        chroma_meta = {
            "memory_type": "human",
            "status": "accepted",
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "user_id": "",
            "session_id": "",
            "task_id": "",
            "agent_id": "",
            "extra_json": "{}",
        }
        # search returns the item we added
        mock_collection.query.return_value = {
            "ids": [[item.id]],
            "documents": [["round trip content"]],
            "metadatas": [[chroma_meta]],
        }

        store = ChromaVectorMemoryStore(SimpleEmbeddingProvider())
        await store.add(item)
        results = await store.search(query="round trip", limit=1)

        assert len(results) == 1
        assert results[0].id == item.id
        assert results[0].content == "round trip content"


# ---------------------------------------------------------------------------
# VertexEmbeddings
# ---------------------------------------------------------------------------


def _make_vertex_mocks(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Inject mock google.genai and google.oauth2.service_account modules.

    Returns (mock_genai_client_instance, mock_genai_module).
    """
    # Build fake google.genai module tree
    mock_genai = types.ModuleType("google.genai")
    mock_genai_types = types.ModuleType("google.genai.types")

    mock_client_instance = MagicMock()
    mock_genai.Client = MagicMock(return_value=mock_client_instance)  # type: ignore[attr-defined]

    # EmbedContentConfig mock
    mock_embed_config = MagicMock()
    mock_genai_types.EmbedContentConfig = MagicMock(return_value=mock_embed_config)  # type: ignore[attr-defined]

    # google namespace package
    mock_google = types.ModuleType("google")
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    # google.oauth2 / service_account
    mock_oauth2 = types.ModuleType("google.oauth2")
    mock_sa_mod = types.ModuleType("google.oauth2.service_account")
    mock_sa_creds = MagicMock()
    mock_sa_mod.Credentials = MagicMock(return_value=mock_sa_creds)  # type: ignore[attr-defined]
    mock_oauth2.service_account = mock_sa_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_genai_types)
    monkeypatch.setitem(sys.modules, "google.oauth2", mock_oauth2)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", mock_sa_mod)

    return mock_client_instance, mock_genai


class TestVertexEmbeddings:
    """Tests for VertexEmbeddings (all mocked — no real GCP calls)."""

    def test_is_embeddings_subclass(self) -> None:
        assert issubclass(VertexEmbeddings, Embeddings)

    def test_vertex_embeddings_default_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Defaults: model=text-embedding-005, dimension=768, output_dimensionality=None."""
        _make_vertex_mocks(monkeypatch)
        emb = VertexEmbeddings()
        assert emb._model == "text-embedding-005"
        assert emb.dimension == 768
        assert emb._output_dimensionality is None

    def test_vertex_embeddings_adc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No service account → credentials=None passed to genai.Client."""
        _mock_client, mock_genai = _make_vertex_mocks(monkeypatch)
        monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_BASE64", raising=False)

        VertexEmbeddings(project="my-project", location="us-east1")

        call_kwargs = mock_genai.Client.call_args.kwargs
        assert call_kwargs["credentials"] is None
        assert call_kwargs["project"] == "my-project"
        assert call_kwargs["location"] == "us-east1"
        assert call_kwargs["vertexai"] is True

    def test_vertex_embeddings_service_account(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Base64 SA JSON → credentials object passed to genai.Client."""
        import base64
        import json as _json

        _mock_client, mock_genai = _make_vertex_mocks(monkeypatch)

        fake_sa = {"type": "service_account", "project_id": "proj"}
        encoded = base64.b64encode(_json.dumps(fake_sa).encode()).decode()

        VertexEmbeddings(service_account_base64=encoded)

        call_kwargs = mock_genai.Client.call_args.kwargs
        assert call_kwargs["credentials"] is not None

    def test_vertex_embeddings_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """project/location/SA read from GOOGLE_CLOUD_PROJECT etc."""
        import base64
        import json as _json

        _mock_client, mock_genai = _make_vertex_mocks(monkeypatch)

        fake_sa = {"type": "service_account", "project_id": "env-proj"}
        encoded = base64.b64encode(_json.dumps(fake_sa).encode()).decode()

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
        monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_BASE64", encoded)

        VertexEmbeddings()

        call_kwargs = mock_genai.Client.call_args.kwargs
        assert call_kwargs["project"] == "env-project"
        assert call_kwargs["location"] == "europe-west1"
        assert call_kwargs["credentials"] is not None

    def test_vertex_embeddings_embed_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """embed() calls models.embed_content and returns the first embedding values."""
        mock_client, _mock_genai = _make_vertex_mocks(monkeypatch)

        expected = [0.1, 0.2, 0.3]
        mock_embedding = MagicMock()
        mock_embedding.values = expected
        mock_resp = MagicMock()
        mock_resp.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_resp

        emb = VertexEmbeddings(model="text-embedding-005", dimension=3)
        result = emb.embed("hello vertex")

        assert result == expected
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-005"
        assert call_kwargs["contents"] == "hello vertex"
        assert "config" not in call_kwargs  # no output_dimensionality set

    async def test_vertex_embeddings_aembed_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """aembed() delegates to embed() via asyncio.to_thread."""
        mock_client, _mock_genai = _make_vertex_mocks(monkeypatch)

        expected = [0.4, 0.5, 0.6]
        mock_embedding = MagicMock()
        mock_embedding.values = expected
        mock_resp = MagicMock()
        mock_resp.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_resp

        emb = VertexEmbeddings(model="text-embedding-005", dimension=3)
        result = await emb.aembed("async hello")

        assert result == expected

    def test_vertex_embeddings_output_dimensionality(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When output_dimensionality is set, EmbedContentConfig is passed to the API."""
        mock_client, _mock_genai = _make_vertex_mocks(monkeypatch)

        expected = [0.1, 0.2]
        mock_embedding = MagicMock()
        mock_embedding.values = expected
        mock_resp = MagicMock()
        mock_resp.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_resp

        emb = VertexEmbeddings(output_dimensionality=512)
        assert emb.dimension == 512  # _dimension = output_dimensionality when set

        emb.embed("truncated embedding")

        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert "config" in call_kwargs
        # EmbedContentConfig should have been constructed with output_dimensionality=512
        import sys as _sys

        genai_types = _sys.modules["google.genai.types"]
        genai_types.EmbedContentConfig.assert_called_once_with(output_dimensionality=512)
