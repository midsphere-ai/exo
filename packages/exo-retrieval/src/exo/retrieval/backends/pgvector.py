"""PostgreSQL/pgvector vector store backend.

Requires the ``asyncpg`` and ``pgvector`` packages::

    pip install exo-retrieval[pgvector]
"""

from __future__ import annotations

import json
from typing import Any

from exo.retrieval.types import Chunk, RetrievalResult  # pyright: ignore[reportMissingImports]
from exo.retrieval.vector_store import VectorStore  # pyright: ignore[reportMissingImports]

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError as exc:
    msg = (
        "asyncpg is required for PgVectorStore. "
        "Install it with: pip install exo-retrieval[pgvector]"
    )
    raise ImportError(msg) from exc


_DEFAULT_TABLE = "exo_vectors"


class PgVectorStore(VectorStore):
    """PostgreSQL vector store using the pgvector extension.

    Uses ``asyncpg`` for async PostgreSQL access and the ``<=>`` cosine
    distance operator for similarity search.

    Args:
        dsn: PostgreSQL connection string (e.g. ``postgresql://user:pass@host/db``).
        table: Name of the table to store vectors in.
        dimensions: Dimensionality of embedding vectors.
        pool: Optional pre-existing ``asyncpg.Pool`` to use instead of
            creating one from *dsn*.
    """

    def __init__(
        self,
        dsn: str = "",
        *,
        table: str = _DEFAULT_TABLE,
        dimensions: int = 1536,
        pool: asyncpg.Pool | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._dimensions = dimensions
        self._pool: asyncpg.Pool | None = pool  # type: ignore[type-arg]
        self._owns_pool = pool is None

    async def _get_pool(self) -> asyncpg.Pool:  # type: ignore[type-arg]
        """Return the connection pool, creating it if necessary."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
        return self._pool

    async def initialize(self) -> None:
        """Create the pgvector extension and table if they don't exist."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id BIGSERIAL PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self._dimensions}) NOT NULL
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_document_id
                ON {self._table} (document_id)
            """)

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their embedding vectors."""
        if len(chunks) != len(embeddings):
            msg = f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            raise ValueError(msg)

        if not chunks:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._table}
                    (document_id, chunk_index, content, start_offset, end_offset, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::vector)
                """,
                [
                    (
                        chunk.document_id,
                        chunk.index,
                        chunk.content,
                        chunk.start,
                        chunk.end,
                        json.dumps(chunk.metadata),
                        _vector_literal(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ],
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks using cosine distance (``<=>``).

        Returns results ranked by similarity (highest score first).
        Cosine distance is converted to similarity via ``1 - distance``.
        """
        pool = await self._get_pool()

        where_clauses: list[str] = []
        params: list[Any] = [_vector_literal(query_embedding), top_k]

        if filter:
            for key, value in filter.items():
                idx = len(params) + 1
                where_clauses.append(f"metadata->>'{key}' = ${idx}")
                params.append(str(value))

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
            SELECT document_id, chunk_index, content, start_offset, end_offset,
                   metadata, 1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results: list[RetrievalResult] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            chunk = Chunk(
                document_id=row["document_id"],
                index=row["chunk_index"],
                content=row["content"],
                start=row["start_offset"],
                end=row["end_offset"],
                metadata=metadata,
            )
            results.append(RetrievalResult(chunk=chunk, score=float(row["score"])))

        return results

    async def delete(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE document_id = $1",
                document_id,
            )

    async def clear(self) -> None:
        """Remove all stored chunks and embeddings."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {self._table}")

    async def close(self) -> None:
        """Close the connection pool if we own it."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None


def _vector_literal(vec: list[float]) -> str:
    """Convert a list of floats to a pgvector literal string, e.g. ``'[1,2,3]'``."""
    return "[" + ",".join(str(v) for v in vec) + "]"
