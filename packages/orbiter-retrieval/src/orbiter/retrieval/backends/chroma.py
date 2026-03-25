"""ChromaDB vector store backend.

Requires the ``chromadb`` package::

    pip install orbiter-retrieval[chroma]
"""

from __future__ import annotations

import json
from typing import Any

from orbiter.retrieval.types import Chunk, RetrievalResult  # pyright: ignore[reportMissingImports]
from orbiter.retrieval.vector_store import VectorStore  # pyright: ignore[reportMissingImports]

try:
    import chromadb  # type: ignore[import-untyped]
except ImportError as exc:
    msg = (
        "chromadb is required for ChromaVectorStore. "
        "Install it with: pip install orbiter-retrieval[chroma]"
    )
    raise ImportError(msg) from exc

_DEFAULT_COLLECTION = "orbiter_vectors"


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store for local persistent or ephemeral vector search.

    Wraps the ChromaDB ``Collection`` API for similarity search using cosine
    distance.

    Args:
        collection_name: Name of the ChromaDB collection.
        path: Directory path for persistent storage.  When *None*, an
            ephemeral (in-memory) client is used.
        client: Optional pre-existing ``chromadb.ClientAPI`` instance.
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        *,
        path: str | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._path = path

        if client is not None:
            self._client = client
        elif path is not None:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their embedding vectors."""
        if len(chunks) != len(embeddings):
            msg = (
                f"Number of chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) must match"
            )
            raise ValueError(msg)

        if not chunks:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = f"{chunk.document_id}:{chunk.index}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            # ChromaDB metadata must be flat (str/int/float/bool values).
            # Store full chunk info so we can reconstruct on retrieval.
            meta: dict[str, Any] = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.index,
                "start_offset": chunk.start,
                "end_offset": chunk.end,
                "chunk_metadata": json.dumps(chunk.metadata),
            }
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks using ChromaDB cosine distance.

        Returns results ranked by similarity (highest score first).
        ChromaDB returns cosine distances; we convert to similarity via
        ``1 - distance``.
        """
        where: dict[str, Any] | None = None
        if filter:
            if len(filter) == 1:
                key, value = next(iter(filter.items()))
                where = {"chunk_metadata": {"$contains": json.dumps({key: value})[1:-1]}}
            else:
                where = {
                    "$and": [
                        {"chunk_metadata": {"$contains": json.dumps({k: v})[1:-1]}}
                        for k, v in filter.items()
                    ]
                }

        query_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievalResult] = []

        if not query_result["ids"] or not query_result["ids"][0]:
            return results

        ids_list = query_result["ids"][0]
        documents_list = query_result["documents"][0] if query_result["documents"] else [None] * len(ids_list)
        metadatas_list = query_result["metadatas"][0] if query_result["metadatas"] else [{}] * len(ids_list)
        distances_list = query_result["distances"][0] if query_result["distances"] else [0.0] * len(ids_list)

        for doc, meta, distance in zip(documents_list, metadatas_list, distances_list):
            chunk_metadata = {}
            raw_meta = meta.get("chunk_metadata", "{}") if meta else "{}"
            if isinstance(raw_meta, str):
                chunk_metadata = json.loads(raw_meta)

            chunk = Chunk(
                document_id=meta.get("document_id", "") if meta else "",
                index=meta.get("chunk_index", 0) if meta else 0,
                content=doc or "",
                start=meta.get("start_offset", 0) if meta else 0,
                end=meta.get("end_offset", 0) if meta else 0,
                metadata=chunk_metadata,
            )
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity: 1 - distance.
            score = 1.0 - float(distance)
            results.append(RetrievalResult(chunk=chunk, score=score))

        return results

    async def delete(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        self._collection.delete(
            where={"document_id": document_id},
        )

    async def clear(self) -> None:
        """Remove all stored chunks and embeddings.

        Deletes and re-creates the collection to clear all data.
        """
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
