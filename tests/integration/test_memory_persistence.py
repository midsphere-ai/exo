"""Integration tests for SQLiteMemoryStore persistence, retrieval, and search.

Tests that:
- 5 HumanMemory items written by session_id are all returned on load.
- Keyword search returns matching items and excludes non-matching ones.
- Metadata isolation by user_id ensures cross-user items are invisible.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_sqlite_write_then_read_by_session(memory_store) -> None:
    """Write 5 HumanMemory items with a session_id and read all back."""
    from exo.memory.base import (  # pyright: ignore[reportMissingImports]
        HumanMemory,
        MemoryMetadata,
    )

    session_id = "test-session-007"
    items_written = []
    for i in range(5):
        item = HumanMemory(
            content=f"Session message number {i}",
            metadata=MemoryMetadata(session_id=session_id),
        )
        await memory_store.add(item)
        items_written.append(item)

    results = await memory_store.search(
        metadata=MemoryMetadata(session_id=session_id),
        limit=10,
    )

    assert len(results) == 5
    returned_ids = {r.id for r in results}
    assert all(item.id in returned_ids for item in items_written)


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_sqlite_keyword_search(memory_store) -> None:
    """Keyword search returns matching items and excludes non-matching ones."""
    from exo.memory.base import (  # pyright: ignore[reportMissingImports]
        HumanMemory,
    )

    # Items containing the unique keyword
    matching_items = [
        HumanMemory(content="The ORBITERMAGICTOKEN protocol is defined in RFC 9999."),
        HumanMemory(content="ORBITERMAGICTOKEN enables distributed memory indexing."),
    ]
    for item in matching_items:
        await memory_store.add(item)

    # Items that should NOT match
    non_matching_items = [
        HumanMemory(content="The weather forecast shows rain tomorrow."),
        HumanMemory(content="Python is a popular programming language."),
        HumanMemory(content="The capital of France is Paris."),
    ]
    for item in non_matching_items:
        await memory_store.add(item)

    results = await memory_store.search(query="ORBITERMAGICTOKEN", limit=10)

    # All matching items must appear
    result_contents = [r.content for r in results]
    assert any("ORBITERMAGICTOKEN" in c for c in result_contents)
    assert len([c for c in result_contents if "ORBITERMAGICTOKEN" in c]) == 2

    # Non-matching items must not appear
    assert not any("weather forecast" in c for c in result_contents)
    assert not any("programming language" in c for c in result_contents)


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_sqlite_metadata_isolation(memory_store) -> None:
    """Items stored with user_id='user-A' are invisible to user_id='user-B' queries."""
    from exo.memory.base import (  # pyright: ignore[reportMissingImports]
        HumanMemory,
        MemoryMetadata,
    )

    # Write 3 items for user-A
    for i in range(3):
        await memory_store.add(
            HumanMemory(
                content=f"User A fact {i}: secret data alpha",
                metadata=MemoryMetadata(user_id="user-A"),
            )
        )

    # Write 2 items for user-B
    for i in range(2):
        await memory_store.add(
            HumanMemory(
                content=f"User B fact {i}: secret data beta",
                metadata=MemoryMetadata(user_id="user-B"),
            )
        )

    # Query user-A: only user-A items returned
    user_a_results = await memory_store.search(
        metadata=MemoryMetadata(user_id="user-A"),
        limit=10,
    )
    assert len(user_a_results) == 3
    assert all(item.metadata.user_id == "user-A" for item in user_a_results)

    # Query user-B: only user-B items returned
    user_b_results = await memory_store.search(
        metadata=MemoryMetadata(user_id="user-B"),
        limit=10,
    )
    assert len(user_b_results) == 2
    assert all(item.metadata.user_id == "user-B" for item in user_b_results)
