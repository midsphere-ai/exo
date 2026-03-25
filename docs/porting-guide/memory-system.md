# Memory System — agent-core to Exo Mapping

**Epic:** 5 — Enhanced Memory System
**Date:** 2026-03-10

This document maps agent-core's (openJiuwen) 5-type memory system to Exo's
enhanced `exo-memory` package, covering encryption, deduplication, taxonomy,
migration, and unified search.

---

## 1. Agent-Core Overview

Agent-core's memory system lives in `openjiuwen/core/memory/` and provides
typed, encrypted, deduplicated memory storage for long-running agents.

### Key Components

**`MemoryType` (Taxonomy)** — An enum classifying what kind of knowledge a
memory item represents. Agent-core defines five types:

| MemoryType | Purpose |
|------------|---------|
| `USER_PROFILE` | User preferences, traits, and biographical info |
| `SEMANTIC_MEMORY` | Factual knowledge and learned concepts |
| `EPISODIC_MEMORY` | Event-specific memories tied to time and place |
| `VARIABLE` | Runtime variables and ephemeral state |
| `SUMMARY` | Compressed summaries of conversation or knowledge |

**AES-256 Encryption at Rest** — Agent-core encrypts memory content before
persisting it. Uses AES-256 with a password-derived key (PBKDF2). All content
fields are encrypted; metadata remains in plaintext for indexing. Keyword
search on encrypted content is not supported (ciphertext is opaque).

**`MemUpdateChecker` (LLM Dedup)** — Prevents memory bloat by comparing new
memories against existing ones using an LLM call. Returns one of four
decisions:

| Decision | Meaning |
|----------|---------|
| `ADD` | New information is distinct — store it |
| `SKIP` | Duplicate — same information already exists |
| `MERGE` | Overlapping — combine into a single memory |
| `DELETE` | Existing memories are outdated/superseded |

Falls back to exact content matching when no LLM is available.

**Migration System** — Versioned schema migrations for the underlying storage
(SQLite or Postgres). Tracks applied versions in a `_migrations` metadata
table with monotonically increasing version numbers.

**`SearchManager` (Unified Search)** — Queries multiple memory backends in
a single call. Gathers results in parallel, deduplicates by item ID, and
returns results sorted by recency.

---

## 2. Exo Equivalent

Exo's memory system lives in the `exo-memory` package
(`packages/exo-memory/`) and implements the same five-type taxonomy,
encryption, deduplication, and migration capabilities — restructured around
a `MemoryStore` protocol for pluggable backends.

### Architecture Difference

Where agent-core uses a monolithic memory manager, Exo separates concerns
into composable components behind a protocol interface:

```python
# Agent-core: monolithic
memory_manager = MemoryManager(config, encryption_key=key)
await memory_manager.store(content, memory_type=MemoryType.SEMANTIC_MEMORY)
await memory_manager.search("query")

# Exo: composable protocol-based
from exo.memory import (
    ShortTermMemory, EncryptedMemoryStore, MemUpdateChecker,
    SearchManager, MemoryCategory, HumanMemory
)

store = ShortTermMemory()
encrypted = EncryptedMemoryStore(store, key=aes_key)
dedup = MemUpdateChecker(checker=my_llm_call)
search = SearchManager(stores=[encrypted])
```

### Component Mapping

| Agent-Core Component | Exo Equivalent | Notes |
|---------------------|-------------------|-------|
| `MemoryType` enum | `MemoryCategory` enum | Same 5 types + `CONVERSATION`; uses `StrEnum` |
| `MemoryItem` (single class) | `MemoryItem` typed hierarchy | `SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory` subclasses |
| AES-256 encryption | `EncryptedMemoryStore` wrapper | AES-256-GCM via `cryptography` package; wraps any `MemoryStore` |
| `MemUpdateChecker` | `MemUpdateChecker` | Same name; returns `MergeResult` with `UpdateDecision` enum |
| Migration system | `MigrationRegistry` + `run_migrations()` | Registry pattern with `Migration` dataclass |
| `SearchManager` | `SearchManager` | Parallel `asyncio.gather()` across stores |
| *(no equivalent)* | `MemoryStore` protocol | `@runtime_checkable` async protocol for pluggable backends |
| *(no equivalent)* | `MemoryStatus` lifecycle | `DRAFT` → `ACCEPTED` → `DISCARD` with validated transitions |
| *(no equivalent)* | `MemoryMetadata` | Frozen Pydantic model with `user_id`, `session_id`, `task_id`, `agent_id` |
| *(no equivalent)* | `MemoryEventEmitter` | Wraps stores to emit `memory:added`, `memory:searched`, `memory:cleared` |
| *(no equivalent)* | Evolution strategies | `ACEStrategy`, `ReasoningBankStrategy`, `ReMeStrategy` — composable via `>>` and `\|` |

### Key Exo Additions Beyond Agent-Core

**Typed Memory Hierarchy** — Instead of a single item class, Exo provides
specialized subclasses with role-specific fields:

| Subclass | `memory_type` | Extra Fields |
|----------|--------------|--------------|
| `SystemMemory` | `"system"` | *(none)* |
| `HumanMemory` | `"human"` | *(none)* |
| `AIMemory` | `"ai"` | `tool_calls: list[dict]` |
| `ToolMemory` | `"tool"` | `tool_call_id`, `tool_name`, `is_error` |

**Status Lifecycle** — Every `MemoryItem` has a `MemoryStatus` with enforced
transitions:

```
DRAFT ──→ ACCEPTED ──→ DISCARD
  │                        ▲
  └────────────────────────┘
```

**Memory Evolution Strategies** — Composable transformations for memory lists:
- `ACEStrategy` — Scores memories with helpful/harmful/neutral counters; prunes low-quality
- `ReasoningBankStrategy` — Structured entries with title/description; embedding-based recall
- `ReMeStrategy` — Reflection-based memory evolution
- Compose with `>>` (sequential) or `|` (parallel)

---

## 3. Side-by-Side Examples

### Storing an Encrypted Memory

```python
# Agent-core
memory_manager = MemoryManager(config, encryption_key="my-password")
await memory_manager.store(
    content="User prefers dark mode",
    memory_type=MemoryType.USER_PROFILE,
)

# Exo
from exo.memory import (
    EncryptedMemoryStore, ShortTermMemory, HumanMemory,
    MemoryCategory, MemoryMetadata,
)
from exo.memory.encrypted import derive_key

key, salt = derive_key("my-password")
store = ShortTermMemory()
encrypted = EncryptedMemoryStore(store, key=key)

item = HumanMemory(
    content="User prefers dark mode",
    category=MemoryCategory.USER_PROFILE,
    metadata=MemoryMetadata(user_id="user-123"),
)
await encrypted.add(item)  # content encrypted transparently

# Retrieval decrypts automatically
result = await encrypted.get(item.id)
assert result.content == "User prefers dark mode"
```

### Running Deduplication

```python
# Agent-core
checker = MemUpdateChecker(llm_call=my_llm)
result = checker.check(new_memory, existing_memories)
if result.decision == "add":
    await memory_manager.store(new_memory)
elif result.decision == "merge":
    await memory_manager.update(result.merged_content)

# Exo
from exo.memory import MemUpdateChecker, UpdateDecision

checker = MemUpdateChecker(checker=my_llm_call, top_k=5)
result = await checker.check(new_item, existing_items)

if result.decision == UpdateDecision.ADD:
    await store.add(new_item)
elif result.decision == UpdateDecision.MERGE:
    merged = new_item.model_copy(update={"content": result.merged_content})
    for old_id in result.delete_ids:
        old = await store.get(old_id)
        if old:
            old.transition(MemoryStatus.DISCARD)
    await store.add(merged)
```

### Running Migrations

```python
# Agent-core
# Migrations run automatically on initialization

# Exo
from exo.memory.migrations import Migration, MigrationRegistry, run_migrations

registry = MigrationRegistry()
registry.register(Migration(
    version=1,
    description="Add embedding column",
    up=lambda db: db.execute("ALTER TABLE memories ADD COLUMN embedding BLOB"),
))

applied = await run_migrations(sqlite_store, registry)
# Returns count of newly applied migrations
```

### Unified Search Across Stores

```python
# Agent-core
results = await memory_manager.search("dark mode preference")

# Exo
from exo.memory import SearchManager, MemoryCategory

manager = SearchManager(stores=[short_term, long_term, vector_store])
results = await manager.search(
    "dark mode preference",
    category=MemoryCategory.USER_PROFILE,
    limit=10,
)
# Queries all stores in parallel, deduplicates, sorts by newest first
```

---

## 4. Migration Table

| Agent-Core Path | Exo Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.core.memory.MemoryType` | `exo.memory.MemoryCategory` | 5-type taxonomy enum (`USER_PROFILE`, `SEMANTIC`, `EPISODIC`, `VARIABLE`, `SUMMARY`) |
| `openjiuwen.core.memory.MemoryItem` | `exo.memory.MemoryItem` | Base class with typed subclasses (`SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory`) |
| *(AES-256 in MemoryManager)* | `exo.memory.EncryptedMemoryStore` | AES-256-GCM wrapper for any `MemoryStore`; encrypts `content` field only |
| *(PBKDF2 key derivation)* | `exo.memory.encrypted.derive_key` | PBKDF2-HMAC-SHA256, 480k iterations, returns `(key, salt)` |
| `openjiuwen.core.memory.MemUpdateChecker` | `exo.memory.MemUpdateChecker` | LLM-based dedup returning `MergeResult` with `UpdateDecision` |
| *(migration in MemoryManager)* | `exo.memory.migrations.run_migrations` | Applies pending `Migration` objects from a `MigrationRegistry` |
| *(migration tracking)* | `exo.memory.migrations.MigrationRegistry` | Ordered registry with `register()` and `list_pending()` |
| `openjiuwen.core.memory.SearchManager` | `exo.memory.SearchManager` | Parallel multi-store search with dedup and recency sort |
| *(no equivalent)* | `exo.memory.MemoryStore` | `@runtime_checkable` async protocol: `add`, `get`, `search`, `clear` |
| *(no equivalent)* | `exo.memory.MemoryStatus` | Lifecycle enum: `DRAFT`, `ACCEPTED`, `DISCARD` |
| *(no equivalent)* | `exo.memory.MemoryMetadata` | Frozen Pydantic model for scoping (`user_id`, `session_id`, `task_id`, `agent_id`) |
| *(no equivalent)* | `exo.memory.ShortTermMemory` | In-memory conversation store with scope-based windowing |
| *(no equivalent)* | `exo.memory.long_term.MemoryOrchestrator` | Batches and coordinates async LLM extraction for long-term storage |
| *(no equivalent)* | `exo.memory.events.MemoryEventEmitter` | Wraps stores to emit `memory:added`, `memory:searched`, `memory:cleared` |
| *(no equivalent)* | `exo.memory.summary.SummaryConfig` | Threshold-based summarization triggers with `SummaryTemplate` presets |
| *(no equivalent)* | `exo.memory.evolution.MemoryEvolutionStrategy` | ABC for memory transforms; composable via `>>` (sequential) and `\|` (parallel) |
| *(no equivalent)* | `exo.memory.evolution.ACEStrategy` | Score-based memory quality tracking with pruning |
| *(no equivalent)* | `exo.memory.evolution.ReasoningBankStrategy` | Structured entry storage with embedding-based recall |
| *(no equivalent)* | `exo.memory.evolution.ReMeStrategy` | Reflection-based memory evolution |
| *(no equivalent)* | `exo.memory.backends.sqlite.SQLiteMemoryStore` | SQLite backend with JSON metadata indexes |
| *(no equivalent)* | `exo.memory.backends.postgres.PostgresMemoryStore` | Async Postgres backend via asyncpg |
| *(no equivalent)* | `exo.memory.backends.vector.VectorMemoryStore` | Embedding-based similarity search backend |

All public symbols are re-exported from `exo.memory` (the package
`__init__.py`), so `from exo.memory import EncryptedMemoryStore` works
as a convenience import.
