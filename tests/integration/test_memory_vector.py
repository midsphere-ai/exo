"""Integration tests for ChromaVectorMemoryStore semantic search.

US-INT-008: Vector memory semantic search test.
"""

from __future__ import annotations

import pytest

from exo.memory.base import HumanMemory  # pyright: ignore[reportMissingImports]


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_chroma_semantic_search_retrieves_relevant(vector_store) -> None:
    """Semantic search returns topically relevant items for each query domain."""
    # 3 astronomy facts
    astronomy_facts = [
        "The Milky Way galaxy contains hundreds of billions of stars",
        "Jupiter is the largest planet in our solar system",
        "Black holes form when massive stars collapse under their own gravity",
    ]
    # 2 cooking facts
    cooking_facts = [
        "Add pasta to heavily salted boiling water for the best flavour",
        "Caramelizing onions requires low heat and at least twenty minutes",
    ]
    for content in astronomy_facts + cooking_facts:
        await vector_store.add(HumanMemory(content=content))

    # Query about astronomy — expect at least 2 astronomy facts in top-5
    astro_results = await vector_store.search(query="planets and stars", limit=5)
    astro_contents = {r.content for r in astro_results}
    matched_astronomy = [f for f in astronomy_facts if f in astro_contents]
    assert len(matched_astronomy) >= 2, (
        f"Expected >= 2 astronomy facts in results, got {len(matched_astronomy)}. "
        f"Results: {[r.content for r in astro_results]}"
    )

    # Query about cooking — expect at least 1 cooking fact in top-5
    cook_results = await vector_store.search(query="recipes and ingredients", limit=5)
    cook_contents = {r.content for r in cook_results}
    matched_cooking = [f for f in cooking_facts if f in cook_contents]
    assert len(matched_cooking) >= 1, (
        f"Expected >= 1 cooking fact in results, got {len(matched_cooking)}. "
        f"Results: {[r.content for r in cook_results]}"
    )


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_chroma_similarity_threshold(vector_store) -> None:
    """No stored item should have cosine similarity >= 0.8 to a completely unrelated query.

    ChromaVectorMemoryStore.search() does not expose a min_score parameter, so
    similarity is computed manually by embedding both the query and each stored
    item with the same provider.  all-MiniLM-L6-v2 produces unit-norm vectors,
    so the dot product equals cosine similarity.
    """
    min_score = 0.8

    # Store 3 cooking-related items
    cooking_facts = [
        "Sauté the onions in butter until translucent and golden brown",
        "Knead the bread dough for ten minutes to develop the gluten structure",
        "Roast the chicken at 200 degrees Celsius until the juices run clear",
    ]
    for content in cooking_facts:
        await vector_store.add(HumanMemory(content=content))

    # Query about quantum physics — completely unrelated to cooking
    unrelated_query = (
        "quantum mechanics wave-particle duality Planck constant energy levels"
    )

    # Embed the query once
    query_emb = await vector_store._embedding_provider.embed(unrelated_query)  # type: ignore[attr-defined]

    above_threshold: list[float] = []
    for content in cooking_facts:
        item_emb = await vector_store._embedding_provider.embed(content)  # type: ignore[attr-defined]
        # Unit-norm vectors: dot product == cosine similarity
        cos_sim = sum(a * b for a, b in zip(query_emb, item_emb, strict=False))
        if cos_sim >= min_score:
            above_threshold.append(cos_sim)

    assert len(above_threshold) == 0, (
        f"Expected 0 cooking items with cosine similarity >= {min_score} "
        f"to an unrelated quantum-physics query, "
        f"but found {len(above_threshold)}: {above_threshold}"
    )
