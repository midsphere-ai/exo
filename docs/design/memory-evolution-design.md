# Memory Evolution Algorithms — ACE, ReasoningBank, ReMe

**Status:** Proposed
**Epic:** 9 — Memory Evolution Algorithms
**Date:** 2026-03-10
**Package:** `exo-memory` (new `evolution/` subpackage)

---

## 1. Motivation

Exo's `exo-memory` package provides a comprehensive memory foundation:

- **MemoryStore protocol** — async `add`, `get`, `search`, `clear` with pluggable backends.
- **MemoryItem hierarchy** — `SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory`
  with `MemoryCategory` taxonomy (user_profile, semantic, episodic, variable, summary).
- **LongTermMemory** — persistent knowledge store with content deduplication and
  namespace isolation.
- **MemoryOrchestrator** — async LLM-powered extraction (USER_PROFILE, AGENT_EXPERIENCE,
  FACTS) with task queuing.
- **MemUpdateChecker** — LLM-based semantic deduplication (ADD/SKIP/MERGE/DELETE decisions).
- **Summary system** — trigger-based LLM summarization with configurable thresholds.
- **Event system** — `MemoryEventEmitter` wraps any store with EventBus integration.
- **Encryption** — `EncryptedMemoryStore` with AES-256-GCM wrapper.

However, it lacks **memory evolution** — algorithms that actively transform, score,
and curate memories over time. Agent-core (`openjiuwen/extensions/context_evolver/`)
provides three research-grade evolution algorithms:

1. **ACE** (Adaptive Context Engine) — Playbook-based scoring with helpful/harmful/neutral
   counters. Reflection classifies memories via LLM, curation prunes low-quality ones.
2. **ReasoningBank** — Structured memories (title/description/content) with semantic
   deduplication and query-based recall via embeddings.
3. **ReMe** (Relevant Memory) — When-to-use based memories with success/failure pattern
   extraction from execution traces.

All three share a **composable pipeline** pattern: `strategy_a >> strategy_b` (sequential)
and `strategy_a | strategy_b` (parallel). This document designs how these evolution
algorithms integrate with Exo's existing memory architecture.

---

## 2. Key Decision: Evolution Strategies as New Module in exo-memory

### Option A — Separate `exo-evolution` package (rejected)

Create a new top-level package for memory evolution. This adds workspace complexity
(new pyproject.toml, UV sources entry, separate test suite) for what is fundamentally
a memory concern. Evolution strategies operate on `MemoryItem` objects and depend on
`MemoryStore` — they belong in the memory package.

### Option B — New `evolution/` subpackage inside exo-memory (chosen)

Add `packages/exo-memory/src/exo/memory/evolution/` containing the base class,
pipeline, and strategy implementations. This keeps evolution algorithms co-located
with the types they transform.

**Why Option B:**

- Evolution strategies consume and produce `MemoryItem` objects — they are a memory
  concern, not a training concern (exo-train's `EvolutionStrategy` is about model
  parameter evolution, not memory content evolution).
- Named `MemoryEvolutionStrategy` to avoid conflict with exo-train's `EvolutionStrategy`.
- Co-location means evolution code can use relative imports to access `base.py`,
  `long_term.py`, and `dedup.py` without cross-package dependencies.
- Tests integrate naturally into exo-memory's existing test suite.
- The `evolution/` subpackage provides a clean namespace without polluting the
  top-level `exo.memory` module.

---

## 3. Key Decision: Composable Pipeline Operators (`>>` and `|`)

### Overview

Agent-core's context evolver uses operator overloading for pipeline composition.
This pattern is elegant and expressive — a natural fit for Python.

### Design

```python
class MemoryEvolutionStrategy(ABC):
    """Base class for memory evolution algorithms.

    Subclasses implement evolve() to transform a list of MemoryItem objects.
    Strategies compose via >> (sequential) and | (parallel) operators.
    """

    name: str

    @abstractmethod
    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Transform memory items according to this strategy.

        Args:
            items: Input memory items to evolve.
            context: Optional context dict (e.g., feedback, model config).

        Returns:
            Transformed list of memory items.
        """
        ...

    def __rshift__(self, other: MemoryEvolutionStrategy) -> MemoryEvolutionPipeline:
        """Sequential composition: self >> other."""
        return MemoryEvolutionPipeline([self, other], mode="sequential")

    def __or__(self, other: MemoryEvolutionStrategy) -> MemoryEvolutionPipeline:
        """Parallel composition: self | other."""
        return MemoryEvolutionPipeline([self, other], mode="parallel")
```

### Pipeline Execution

```python
class MemoryEvolutionPipeline(MemoryEvolutionStrategy):
    """Composes multiple strategies sequentially or in parallel.

    Sequential (>>): Output of each strategy feeds into the next.
    Parallel (|): All strategies run on the same input, results are merged
    (union by item ID, last-write-wins for duplicates).
    """

    name: str = "pipeline"

    def __init__(
        self,
        strategies: list[MemoryEvolutionStrategy],
        mode: Literal["sequential", "parallel"] = "sequential",
    ) -> None: ...

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        if self._mode == "sequential":
            result = items
            for strategy in self._strategies:
                result = await strategy.evolve(result, context=context)
            return result
        else:  # parallel
            results = await asyncio.gather(
                *(s.evolve(items, context=context) for s in self._strategies)
            )
            return _merge_results(results)
```

### Operator Chaining

Pipelines are themselves strategies, so operators chain naturally:

```python
# Sequential: ACE -> ReasoningBank -> ReMe
pipeline = ace >> reasoning_bank >> reme

# Parallel then sequential
pipeline = (ace | reasoning_bank) >> reme

# All three in parallel
pipeline = ace | reasoning_bank | reme
```

When a `MemoryEvolutionPipeline` is composed with another strategy via `>>` or `|`,
it flattens into a single pipeline rather than nesting. This keeps execution
straightforward and avoids deep recursion.

---

## 4. Component Design: ACE Strategy

### Overview

ACE (Adaptive Context Engine) uses a playbook model to score memories based on
human feedback. Each memory gets helpful/harmful/neutral counters that determine
its quality score. Low-scoring memories are pruned during curation.

### Design

```python
class ACEStrategy(MemoryEvolutionStrategy):
    """Adaptive Context Engine — playbook-based memory evolution.

    Tracks per-memory counters (helpful, harmful, neutral) and scores
    memories based on counter ratios. Reflection classifies memories
    via LLM, curation removes memories below a quality threshold.
    """

    name: str = "ace"

    def __init__(
        self,
        *,
        counters_path: Path | str | None = None,
    ) -> None:
        """Initialize ACE strategy.

        Args:
            counters_path: Path for JSON file persistence of counters.
                If None, counters are in-memory only.
        """

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Score items and prune those with high harmful ratios."""

    async def reflect(
        self,
        items: list[MemoryItem],
        feedback: str,
        *,
        model: Any | None = None,
    ) -> None:
        """Classify each memory as helpful/harmful/neutral via LLM.

        Updates internal counters. The model parameter accepts any
        callable with signature async (prompt: str) -> str.
        """

    async def curate(self, *, threshold: float = 0.3) -> list[str]:
        """Remove memories with quality score below threshold.

        Returns list of pruned memory IDs.
        """

    def get_score(self, item_id: str) -> float:
        """Get quality score for a memory (helpful / total ratio)."""
```

### Counter Persistence

Counters are stored as a JSON file:

```json
{
  "memory_id_1": {"helpful": 5, "harmful": 1, "neutral": 2},
  "memory_id_2": {"helpful": 0, "harmful": 3, "neutral": 1}
}
```

The `counters_path` parameter controls persistence. When set, counters are loaded
on init and saved after each `reflect()` call. When `None`, counters exist only
in memory (useful for testing).

### Score Calculation

```
score = helpful / (helpful + harmful + neutral)
```

If total count is 0, score defaults to 0.5 (neutral). The `curate()` method
removes all memories with score below the threshold (default 0.3).

---

## 5. Component Design: ReasoningBank Strategy

### Overview

ReasoningBank stores memories as structured entries with title, description, and
content fields. It deduplicates by semantic similarity and provides query-based
recall using embeddings.

### Design

```python
@dataclass(frozen=True, slots=True)
class ReasoningEntry:
    """A structured memory entry in the ReasoningBank.

    Attributes:
        title: Short identifier for the memory.
        description: Summary of what this memory covers.
        content: Full memory content.
        item_id: ID of the corresponding MemoryItem.
    """

    title: str
    description: str
    content: str
    item_id: str


class ReasoningBankStrategy(MemoryEvolutionStrategy):
    """Structured memory evolution with semantic deduplication and recall.

    Stores memories as ReasoningEntry objects (title/description/content).
    Deduplicates entries by semantic similarity when an embeddings provider
    is available, falling back to keyword matching.
    """

    name: str = "reasoning_bank"

    def __init__(
        self,
        *,
        embeddings: Any | None = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize ReasoningBank.

        Args:
            embeddings: Optional embeddings provider with async embed(text) -> list[float].
                Falls back to keyword matching when None.
            similarity_threshold: Cosine similarity threshold for deduplication.
        """

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Deduplicate items by semantic similarity, summarize redundant entries."""

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[ReasoningEntry]:
        """Retrieve relevant entries by semantic search.

        Uses embeddings when available, falls back to keyword matching.
        """
```

### Embeddings Integration

The `embeddings` parameter is duck-typed — any object with an `async embed(text: str) -> list[float]`
method works. This avoids importing from exo-retrieval (which provides `Embeddings` ABC)
and keeps exo-memory dependency-free.

When `embeddings` is `None`, `recall()` uses keyword matching (case-insensitive substring
search across title, description, and content). Deduplication in `evolve()` falls back
to exact content comparison.

---

## 6. Component Design: ReMe Strategy

### Overview

ReMe (Relevant Memory) extracts success and failure patterns from execution traces.
Each extracted pattern includes `when_to_use` metadata indicating when the pattern
is applicable.

### Design

```python
class ReMeStrategy(MemoryEvolutionStrategy):
    """Relevant Memory — success/failure pattern extraction.

    Analyzes execution traces to extract patterns with when-to-use
    applicability metadata. Deduplicates extracted patterns by content
    similarity.
    """

    name: str = "reme"

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Extract success/failure patterns and deduplicate."""

    async def extract_patterns(
        self,
        items: list[MemoryItem],
        model: Any,
    ) -> list[MemoryItem]:
        """Extract patterns from items via LLM.

        The model parameter accepts any callable with signature
        async (prompt: str) -> str.

        Returns MemoryItem objects with when_to_use stored in
        metadata.extra["when_to_use"].
        """
```

### Pattern Storage

Extracted patterns are stored as regular `MemoryItem` objects. The `when_to_use`
field is stored in `metadata.extra["when_to_use"]` rather than adding a new field
to `MemoryItem`. This keeps the base model unchanged and uses the existing
extensibility mechanism.

### Deduplication

ReMe uses content similarity for deduplication. When items have similar content
(>85% overlap via sequence matching), the newer pattern replaces the older one.
This is simpler than ReasoningBank's embeddings-based dedup — patterns are typically
short enough for string comparison to work well.

---

## 7. Integration with Existing Memory Architecture

### MemoryOrchestrator

The orchestrator's extraction pipeline (submit → process → store) operates
independently from evolution. The intended flow is:

1. **Extraction** — `MemoryOrchestrator` extracts knowledge from conversations
   into `LongTermMemory`.
2. **Evolution** — `MemoryEvolutionStrategy.evolve()` transforms items already
   in long-term storage (scoring, deduplication, pattern extraction).
3. **Storage** — Evolved items are written back to the same `LongTermMemory` store.

Evolution strategies don't replace the orchestrator — they complement it. The
orchestrator handles *extraction* (conversation → knowledge); evolution handles
*refinement* (knowledge → better knowledge).

### MemoryStore Protocol

Evolution strategies operate on `list[MemoryItem]` directly, not on `MemoryStore`.
The caller is responsible for loading items from a store, running evolution, and
writing results back. This keeps strategies decoupled from storage backends:

```python
# Load items from any MemoryStore
items = await store.search(limit=100)

# Run evolution pipeline
evolved = await pipeline.evolve(items)

# Write back
await store.clear()
for item in evolved:
    await store.add(item)
```

### MemUpdateChecker

The existing `MemUpdateChecker` handles deduplication at *write time* (when adding
to `LongTermMemory`). Evolution strategies handle deduplication at *evolution time*
(when transforming a batch of memories). Both can coexist — `MemUpdateChecker` catches
duplicates on individual adds, while evolution strategies deduplicate across the
entire memory corpus.

### MemoryEventEmitter

Evolution operations are not automatically emitted as events. If event tracking is
needed, the caller wraps the store with `MemoryEventEmitter` — evolved items
written back via `add()` trigger `memory:added` events as usual.

### ExtractionType Mapping

Exo's existing `ExtractionType` maps to evolution strategies:

| ExtractionType | Evolution Strategy | Memory Category |
|---|---|---|
| `USER_PROFILE` | ACE (score user knowledge) | `USER_PROFILE` |
| `AGENT_EXPERIENCE` | ReMe (extract patterns) | `EPISODIC` |
| `FACTS` | ReasoningBank (structure + recall) | `SEMANTIC` |

This mapping is a guideline, not a constraint. Any strategy can operate on any
memory items regardless of extraction type or category.

---

## 8. File Layout

All additions are within `packages/exo-memory/`:

| Addition | Location |
|----------|----------|
| `MemoryEvolutionStrategy` ABC | `evolution/__init__.py` (~30 lines) |
| `MemoryEvolutionPipeline` | `evolution/pipeline.py` (~80 lines) |
| `ACEStrategy` | `evolution/ace.py` (~120 lines) |
| `ReasoningBankStrategy`, `ReasoningEntry` | `evolution/reasoning_bank.py` (~130 lines) |
| `ReMeStrategy` | `evolution/reme.py` (~100 lines) |
| New exports | `__init__.py` (extend existing) |
| Tests | `tests/test_evolution_pipeline.py`, `tests/test_ace.py`, `tests/test_reasoning_bank.py`, `tests/test_reme.py` |

Estimated total new code: ~460 lines across 5 new files.

Directory structure:

```
packages/exo-memory/src/exo/memory/
├── evolution/
│   ├── __init__.py          # MemoryEvolutionStrategy ABC, exports
│   ├── pipeline.py          # MemoryEvolutionPipeline, _merge_results
│   ├── ace.py               # ACEStrategy
│   ├── reasoning_bank.py    # ReasoningBankStrategy, ReasoningEntry
│   └── reme.py              # ReMeStrategy
├── base.py                  # Unchanged
├── long_term.py             # Unchanged
├── dedup.py                 # Unchanged
└── ...
```

---

## 9. Backward Compatibility

### Existing APIs — no changes required

| Component | Impact |
|-----------|--------|
| `MemoryItem` | No changes — evolution consumes/produces existing types |
| `MemoryStore` protocol | No changes — evolution operates on `list[MemoryItem]` |
| `LongTermMemory` | No changes — evolution reads/writes via existing methods |
| `ShortTermMemory` | No changes — not used by evolution strategies |
| `MemoryOrchestrator` | No changes — extraction and evolution are independent |
| `MemUpdateChecker` | No changes — write-time dedup coexists with evolution-time dedup |
| `MemoryEventEmitter` | No changes — wraps store transparently |
| All backends | No changes — no schema modifications required |

### New additions (purely additive)

| Addition | Description |
|----------|-------------|
| `MemoryEvolutionStrategy` | New ABC for evolution algorithms |
| `MemoryEvolutionPipeline` | New composable pipeline |
| `ACEStrategy` | New strategy — opt-in |
| `ReasoningBankStrategy` | New strategy — opt-in |
| `ReMeStrategy` | New strategy — opt-in |
| `ReasoningEntry` | New dataclass for ReasoningBank entries |

### Existing code paths — unchanged

All existing memory operations work identically. Evolution strategies are entirely
opt-in — using them requires explicit instantiation and invocation. No existing
imports, function signatures, or behaviors are modified.

---

## 10. Dependencies

- **No new dependencies** — all evolution strategies use stdlib and existing
  exo-memory types.
- LLM calls use duck-typed callables (`async (prompt: str) -> str`) — no dependency
  on specific model providers.
- Embeddings use duck-typed objects (`async embed(text: str) -> list[float]`) — no
  dependency on exo-retrieval.
- File persistence for ACE counters uses stdlib `json` and `pathlib`.

---

## 11. Open Questions

1. **Should evolution strategies persist their internal state to a MemoryStore?**
   Recommendation: No for v1. ACE uses file-based JSON persistence for counters.
   ReasoningBank and ReMe maintain in-memory state. Adding MemoryStore-based
   persistence can be layered on later without breaking the API.

2. **Should `MemoryEvolutionPipeline` support error handling per strategy?**
   Recommendation: No for v1. If one strategy in a pipeline fails, the entire
   `evolve()` call raises. Users who need fault tolerance can wrap individual
   strategies in try/except. A `continue_on_error` option can be added later.

3. **Should parallel pipeline merge use deduplication?**
   Recommendation: Yes — merge by item ID (last-write-wins). If two parallel
   strategies produce items with the same ID, the result from the later strategy
   in the list takes precedence. Items with unique IDs from both branches are
   included.

4. **Should ReasoningBank entries be stored as a separate model instead of MemoryItem?**
   Recommendation: No. `ReasoningEntry` is a view/projection used internally by
   `ReasoningBankStrategy`. Storage uses standard `MemoryItem` with structured
   content (title/description/content as formatted text or JSON in the content field).
   This avoids extending the MemoryStore protocol.

---

## 12. Test Strategy

| Component | Key test cases |
|-----------|---------------|
| `MemoryEvolutionStrategy` | ABC enforcement (cannot instantiate), subclass with evolve() |
| `MemoryEvolutionPipeline` | Sequential composition, parallel composition, operator chaining |
| `>>` operator | Two strategies in sequence, pipeline >> strategy, strategy >> pipeline |
| `\|` operator | Two strategies in parallel, merge deduplication, result union |
| `ACEStrategy` | Counter updates, score calculation, reflect with mock LLM, curate threshold |
| ACE persistence | Save/load counters JSON, missing file creates fresh counters |
| `ReasoningBankStrategy` | evolve deduplication, recall with embeddings, recall keyword fallback |
| `ReMeStrategy` | extract_patterns with mock LLM, deduplication, when_to_use metadata |
| Integration | Pipeline of all three strategies, evolve → store → recall roundtrip |
