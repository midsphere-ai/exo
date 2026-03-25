"""Use the memory system — Exo quickstart.

Demonstrates short-term conversation memory: storing messages,
searching with keyword filters, and scope-based windowing.

Usage:
    uv run python examples/quickstart/use_memory.py
"""

import asyncio

from exo.memory import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    HumanMemory,
    MemoryMetadata,
    ShortTermMemory,
    SystemMemory,
)


async def main() -> None:
    # Create a short-term memory store scoped to a task, keeping last 3 rounds.
    memory = ShortTermMemory(scope="task", max_rounds=3)

    # Define metadata so items are scoped to a user + session + task.
    meta = MemoryMetadata(user_id="user-1", session_id="sess-1", task_id="task-1")

    # Populate a short conversation.
    await memory.add(SystemMemory(content="You are a helpful travel assistant.", metadata=meta))
    await memory.add(HumanMemory(content="What's the best time to visit Tokyo?", metadata=meta))
    await memory.add(
        AIMemory(content="Spring (March-May) is ideal for cherry blossoms.", metadata=meta)
    )
    await memory.add(HumanMemory(content="What about food recommendations?", metadata=meta))
    await memory.add(AIMemory(content="Try ramen in Shinjuku and sushi in Tsukiji.", metadata=meta))

    print(f"Total items stored: {len(memory)}")

    # Search by keyword.
    results = await memory.search(query="cherry")
    print(f"\nSearch 'cherry': {len(results)} result(s)")
    for item in results:
        print(f"  [{item.memory_type}] {item.content[:60]}")

    # Search with metadata filter.
    results = await memory.search(metadata=meta, memory_type="human")
    print(f"\nHuman messages in task-1: {len(results)}")
    for item in results:
        print(f"  {item.content}")

    # Windowing: only the last 3 rounds are visible (system always kept).
    all_items = await memory.search(limit=100)
    print(f"\nAfter windowing (max_rounds=3): {len(all_items)} items visible")


if __name__ == "__main__":
    asyncio.run(main())
