# exo-memory

Memory system for the [Exo](../../README.md) multi-agent framework. Typed short/long-term memory with multiple storage backends.

## Installation

```bash
pip install exo-memory

# With optional backends
pip install exo-memory[sqlite]     # SQLite backend
pip install exo-memory[postgres]   # PostgreSQL backend
pip install exo-memory[vector]     # ChromaDB vector search
```

Requires Python 3.11+ and `exo-core`.

## What's Included

- **MemoryItem** -- typed base class with subtypes: `SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory`.
- **ShortTermMemory** -- conversation-scoped memory with scope-based filtering and round limiting.
- **LongTermMemory** -- persistent memory with LLM-based extraction via `MemoryOrchestrator`.
- **Summary** -- configurable trigger + multi-template summary generation.
- **MemoryPersistence** -- hook-based auto-persistence. Attach to an agent to automatically save LLM responses (`AIMemory`) and tool results (`ToolMemory`) during `run()` or `run.stream()`.
- **Backends** -- in-memory (default), SQLite, PostgreSQL, and ChromaDB vector storage.

## Quick Example

```python
from exo.memory import (
    HumanMemory, AIMemory, ShortTermMemory, MemoryMetadata,
)

stm = ShortTermMemory(scope="session")

stm.add(HumanMemory(
    content="What is Python?",
    metadata=MemoryMetadata(session_id="s-1"),
))

stm.add(AIMemory(
    content="Python is a programming language.",
    tool_calls=[],
))

messages = stm.get_messages(
    metadata=MemoryMetadata(session_id="s-1"),
    max_rounds=10,
)
```

## Documentation

- [Memory Guide](../../docs/guides/memory.md)
- [Memory Backends Guide](../../docs/guides/memory-backends.md)
- [API Reference](../../docs/reference/memory/)
