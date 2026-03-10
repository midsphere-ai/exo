# Context Evolver — agent-core to Orbiter Mapping

**Epic:** 9 — Context Evolution
**Date:** 2026-03-11

This document maps agent-core's (openJiuwen) context evolution algorithms
(ACE, ReasoningBank, ReMe) to Orbiter's `orbiter-memory` evolution package,
covering memory quality scoring, structured recall, pattern extraction, and
composable pipeline operators.

---

## 1. Agent-Core Overview

Agent-core's context evolution system lives in `openjiuwen/context_evolver/`
and provides three composable strategies for transforming and curating agent
memory over time.

### ACE — Adaptive Context Engine

**Playbook-based memory scoring.** Each memory tracks three counters —
helpful, harmful, and neutral — that accumulate from user feedback.
A quality score maps `(helpful - harmful) / total` to the `[0, 1]` range.
Memories whose harmful ratio exceeds a configurable threshold are pruned.

Key capabilities:
- Per-memory counter persistence (JSON file or in-memory)
- LLM-based reflection: classifies each memory against user feedback as
  helpful, harmful, or neutral
- Periodic curation: removes memories below a quality score threshold

### ReasoningBank

**Structured memory storage with semantic retrieval.** Stores memories as
structured entries with title, description, and content fields. Supports
query-based recall using embedding similarity (with keyword fallback when
no embedding provider is available).

Key capabilities:
- Structured `ReasoningEntry` format (title/description/content)
- Semantic deduplication: items above a cosine similarity threshold are
  merged (longer entry wins)
- Query-based recall with top-k retrieval
- Embedding cache for repeated queries

### ReMe — Relevant Memory

**Reflection-based pattern extraction.** Uses an LLM to analyze memory items
and extract success/failure patterns with "when-to-use" metadata. Patterns
are then deduplicated by keyword similarity to prevent bloat.

Key capabilities:
- LLM-driven pattern extraction from raw memories
- Each pattern carries `when_to_use` and `pattern_type` ("success" or
  "failure") metadata
- Keyword-based deduplication (Jaccard similarity)

### Composition Operators

Agent-core composes strategies using two operators:
- **`>>` (sequential)** — Output of one strategy feeds into the next
- **`|` (parallel)** — All strategies process the same input; results are
  merged (union by item ID, last-write-wins for duplicates)

---

## 2. Orbiter Equivalent

Orbiter's context evolution lives in the `orbiter-memory` package at
`packages/orbiter-memory/src/orbiter/memory/evolution/` and implements
the same three strategies behind a shared `MemoryEvolutionStrategy` ABC.

### Architecture

All strategies inherit from `MemoryEvolutionStrategy`, which defines:

```python
class MemoryEvolutionStrategy(ABC):
    name: str

    @abstractmethod
    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]: ...

    def __rshift__(self, other):  # >> sequential
    def __or__(self, other):      # |  parallel
```

`MemoryEvolutionPipeline` composes strategies. Sequential pipelines chain
outputs; parallel pipelines run all strategies on the same input and merge
results (union by item ID, last-write-wins).

### ACEStrategy

```python
from orbiter.memory.evolution import ACEStrategy

ace = ACEStrategy(counter_path="counters.json", harmful_threshold=0.5)
```

- `evolve(items)` — Prunes items where `harmful / total > harmful_threshold`
- `reflect(items, feedback, model=...)` — LLM classifies each memory; updates counters
- `curate(items, threshold=0.3)` — Removes items with `score() < threshold`
- `record(memory_id, label)` — Manually record a feedback label
- `get_counters(memory_id)` — Returns `Counters` with `.helpful`, `.harmful`, `.neutral`, `.score()`

Counter persistence is optional: pass a file path for disk-backed counters
or `None` for in-memory-only operation.

### ReasoningBankStrategy

```python
from orbiter.memory.evolution import ReasoningBankStrategy

bank = ReasoningBankStrategy(embeddings=my_provider, similarity_threshold=0.85)
```

- `evolve(items)` — Deduplicates by semantic similarity (keeps longer entry)
- `recall(query, top_k=5)` — Returns `list[ReasoningEntry]` sorted by relevance

Falls back to keyword matching (Jaccard) when no embeddings provider is
configured. The `ReasoningEntry` dataclass has `title`, `description`,
`content`, and `item_id` fields.

### ReMeStrategy

```python
from orbiter.memory.evolution import ReMeStrategy

reme = ReMeStrategy(similarity_threshold=0.85)
```

- `evolve(items, context={"model": llm})` — Extracts patterns via LLM, then deduplicates
- `extract_patterns(items, model)` — Returns new `MemoryItem` objects with
  `metadata.extra["when_to_use"]` and `metadata.extra["pattern_type"]`

Without a model in context, `evolve()` returns items deduplicated as-is.
Both `PatternModel` and `ReflectionModel` follow the same protocol:
`async (prompt: str) -> str`.

---

## 3. Code Comparison — Composing a Multi-Strategy Evolution Pipeline

### Agent-core

```python
from openjiuwen.context_evolver import ACE, ReasoningBank, ReMe

# Sequential: score → structure → extract patterns
pipeline = ACE(config) >> ReasoningBank(embeddings) >> ReMe()

# Parallel: run ACE and ReasoningBank simultaneously, then ReMe
pipeline = (ACE(config) | ReasoningBank(embeddings)) >> ReMe()

# Execute
evolved = await pipeline.evolve(memory_items, context={"feedback": text})
```

### Orbiter

```python
from orbiter.memory.evolution import (
    ACEStrategy,
    ReasoningBankStrategy,
    ReMeStrategy,
)

# Sequential: score → structure → extract patterns
pipeline = (
    ACEStrategy(counter_path="counters.json")
    >> ReasoningBankStrategy(embeddings=my_provider)
    >> ReMeStrategy()
)

# Parallel: run ACE and ReasoningBank simultaneously, then ReMe
pipeline = (
    ACEStrategy(counter_path="counters.json")
    | ReasoningBankStrategy(embeddings=my_provider)
) >> ReMeStrategy()

# Execute — same async interface
evolved = await pipeline.evolve(memory_items, context={"model": my_llm})
```

Key differences:
- Orbiter uses explicit `counter_path` instead of a config object for ACE
- Orbiter passes the LLM via `context={"model": ...}` rather than constructor injection
- Pipeline flattening is automatic: `a >> b >> c` produces a single
  3-strategy sequential pipeline (no nesting)
- Parallel merge uses last-write-wins by item ID, matching agent-core semantics

### Using ACE Reflection Separately

```python
# Agent-core
ace = ACE(config)
labels = await ace.classify(memories, user_feedback, llm=my_model)

# Orbiter
ace = ACEStrategy(counter_path="counters.json")
labels = await ace.reflect(memories, feedback="User said this was wrong", model=my_llm)
# labels: {"mem-id-1": "harmful", "mem-id-2": "neutral", ...}
# Counters updated automatically; next evolve() will prune accordingly
```

### Using ReasoningBank Recall

```python
# Agent-core
bank = ReasoningBank(embeddings=emb_provider)
entries = await bank.query("how to handle rate limits", top_k=3)

# Orbiter
bank = ReasoningBankStrategy(embeddings=emb_provider)
await bank.evolve(memory_items)  # populate the bank first
entries = await bank.recall("how to handle rate limits", top_k=3)
# entries: list[ReasoningEntry] with .title, .description, .content, .item_id
```

---

## 4. Migration Table

| Agent-Core Path | Orbiter Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.context_evolver.ACE` | `orbiter.memory.evolution.ACEStrategy` | Playbook-based memory scoring with helpful/harmful/neutral counters |
| `openjiuwen.context_evolver.ACE.classify` | `orbiter.memory.evolution.ACEStrategy.reflect` | LLM-based memory classification against user feedback |
| `openjiuwen.context_evolver.ACE.curate` | `orbiter.memory.evolution.ACEStrategy.curate` | Score-threshold pruning (removes items below quality floor) |
| *(counter persistence)* | `ACEStrategy(counter_path=...)` | JSON file persistence for per-memory counters |
| `openjiuwen.context_evolver.ReasoningBank` | `orbiter.memory.evolution.ReasoningBankStrategy` | Structured entries (title/description/content) with semantic dedup |
| `openjiuwen.context_evolver.ReasoningBank.query` | `ReasoningBankStrategy.recall` | Top-k semantic search with embedding or keyword fallback |
| *(ReasoningBank entry)* | `orbiter.memory.evolution.ReasoningEntry` | Frozen dataclass: `title`, `description`, `content`, `item_id` |
| `openjiuwen.context_evolver.ReMe` | `orbiter.memory.evolution.ReMeStrategy` | LLM-based pattern extraction with when-to-use metadata |
| *(ReMe pattern type)* | `MemoryItem.metadata.extra["pattern_type"]` | `"success"` or `"failure"` — stored in metadata extra dict |
| *(ReMe when-to-use)* | `MemoryItem.metadata.extra["when_to_use"]` | Applicability hint stored in metadata extra dict |
| `openjiuwen.context_evolver.>>` operator | `MemoryEvolutionStrategy.__rshift__` | Sequential composition — output of each strategy feeds into the next |
| `openjiuwen.context_evolver.\|` operator | `MemoryEvolutionStrategy.__or__` | Parallel composition — union by item ID, last-write-wins |
| *(pipeline)* | `orbiter.memory.evolution.MemoryEvolutionPipeline` | Composes strategies; auto-flattens same-mode nesting |
| *(base class)* | `orbiter.memory.evolution.MemoryEvolutionStrategy` | ABC with `evolve()` method and composition operators |
| *(LLM protocol — classification)* | `orbiter.memory.evolution.ace.ReflectionModel` | `async (prompt: str) -> str` returning "helpful"/"harmful"/"neutral" |
| *(LLM protocol — extraction)* | `orbiter.memory.evolution.reme.PatternModel` | `async (prompt: str) -> str` returning JSON array of patterns |

All public symbols are re-exported from `orbiter.memory.evolution` (the
package `__init__.py`), so `from orbiter.memory.evolution import ACEStrategy`
works as a convenience import.
