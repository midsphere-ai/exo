"""Embedding-based query deduplication."""
from __future__ import annotations
import logging

logger = logging.getLogger("deepsearch")

SIMILARITY_THRESHOLD = 0.86

async def dedup_queries(
    new_queries: list[str],
    existing_queries: list[str],
    embedding_provider,
) -> list[str]:
    """Remove semantically duplicate queries using embeddings.

    Args:
        new_queries: New queries to check
        existing_queries: Previously used queries
        embedding_provider: EmbeddingProvider instance

    Returns:
        List of unique queries
    """
    if len(new_queries) == 1 and not existing_queries:
        return new_queries

    from .embeddings import cosine_similarity

    all_queries = [*new_queries, *existing_queries]
    all_embeddings = await embedding_provider.embed(all_queries)

    if not all_embeddings:
        return new_queries

    new_embeddings = all_embeddings[:len(new_queries)]
    existing_embeddings = all_embeddings[len(new_queries):]

    unique = []
    used_indices: set[int] = set()

    for i, q in enumerate(new_queries):
        is_unique = True

        # Check against existing
        for j, emb in enumerate(existing_embeddings):
            sim = cosine_similarity(new_embeddings[i], emb)
            if sim >= SIMILARITY_THRESHOLD:
                is_unique = False
                break

        # Check against already accepted
        if is_unique:
            for idx in used_indices:
                sim = cosine_similarity(new_embeddings[i], new_embeddings[idx])
                if sim >= SIMILARITY_THRESHOLD:
                    is_unique = False
                    break

        if is_unique:
            unique.append(q)
            used_indices.add(i)

    logger.debug("Dedup: %d -> %d queries", len(new_queries), len(unique))
    return unique
