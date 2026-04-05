# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Exo

Exo is a modular multi-agent framework for building LLM-powered applications in Python. It's a UV workspace monorepo with 18 packages. Requires Python 3.11+.

## Common Commands

```bash
# Install all workspace packages (editable mode)
uv sync

# Run all tests (~2,900 tests, asyncio_mode=auto)
uv run pytest

# Run tests for a single package
uv run pytest packages/exo-core/tests/

# Run a single test file
uv run pytest packages/exo-core/tests/test_agent.py

# Run a single test
uv run pytest packages/exo-core/tests/test_agent.py::test_function_name

# Lint (with auto-fix)
uv run ruff check packages/ --fix

# Format check
uv run ruff format --check packages/

# Type-check a package
uv run pyright packages/exo-core/

# Verify installation
uv run python -c "from exo import Agent, run, tool; print('OK')"
```

### exo-web (dual Node+Python package)

```bash
cd packages/exo-web

# Install frontend deps
npm install

# Dev server (runs Astro + FastAPI concurrently)
npm run dev

# Astro typecheck
npx astro check

# Run backend only
uv run uvicorn exo_web.app:app --reload
```

## Architecture

UV workspace monorepo. Packages live in `packages/`. The dependency graph flows upward from `exo-core`:

```
exo-core (foundation, only depends on pydantic)
    ↑
exo-models (OpenAI, Anthropic, Gemini, Vertex AI providers)
    ↑
exo-context, exo-memory, exo-mcp, exo-sandbox, exo-observability, exo-guardrail
    ↑
exo-retrieval, exo-search, exo-cli, exo-server, exo-eval, exo-a2a, exo-train, exo-web
    ↑
exo (meta-package, re-exports everything)
```

### Key Packages

- **exo-core** (`packages/exo-core/src/exo/`): `Agent`, `Tool`, `@tool` decorator, `run`/`run.sync`/`run.stream`, `Swarm`, hooks, events, config, registry. The `_internal/` subpackage has message building, output parsing, call execution, state machine, and graph algorithms.
- **exo-models** (`packages/exo-models/`): LLM provider implementations. Provider SDKs are isolated here — core has zero heavy deps.
- **exo-guardrail** (`packages/exo-guardrail/`): Security guardrails — pattern-based and LLM-based prompt injection/jailbreak detection with pluggable backends.
- **exo-retrieval** (`packages/exo-retrieval/`): RAG pipeline — embeddings (OpenAI, Vertex, HTTP), vector stores (pgvector, ChromaDB), hybrid search, reranking, knowledge graph, agentic retrieval.
- **exo-search** (`packages/exo-search/`): AI search engine with query classification, parallel research agents, result reranking, citation generation, and 3 quality modes (speed/balanced/quality).
- **exo-web** (`packages/exo-web/`): Full platform UI. Hybrid package — Astro 5.x frontend (`src/pages/`, `src/islands/`) + FastAPI backend (`src/exo_web/`). Has its own `package.json` AND `pyproject.toml`.

### exo-web Backend Structure

- `app.py` — FastAPI app entry point, middleware, route registration
- `config.py` — Settings dataclass (env vars: `EXO_DATABASE_URL`, `EXO_SECRET_KEY`, `EXO_DEBUG`)
- `database.py` — `get_db()` async context manager, WAL mode, foreign keys
- `engine.py` — Workflow execution engine (topological sort, node execution, retry)
- `migrations/` — Sequential SQL files, run automatically on startup via lifespan
- `routes/` — 30+ APIRouter modules, all under `/api/v1/` prefix
- `services/` — Business logic layer (agent runtime, sandbox, scheduler, memory)
- `middleware/` — CSRF, rate limiting, security headers, API version redirect

### exo-web Frontend Structure

- Astro 5.x pages in `src/pages/`, layouts in `src/layouts/`
- React islands in `src/islands/` (e.g., ReactFlow canvas)
- Tailwind CSS v4 via `@tailwindcss/vite`
- `cn()` utility at `src/utils/merge.ts` for class merging

## Code Conventions

- **Ruff**: line-length 100, rules `E,F,I,N,W,UP,B,SIM,RUF`, ignore `E501`. Use `datetime.UTC` not `timezone.utc`.
- **Pyright**: basic mode, Python 3.11 target.
- **Async-first**: all core APIs are async. Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).
- **Pydantic v2**: for all schemas and validation.
- **Test file names must be unique** across all packages (pytest importlib mode).
- **Tests use MockProvider** — never make real API calls.
- **Model strings**: format `"provider:model"` (e.g., `"openai:gpt-4o-mini"`).
- **FastAPI Depends()**: use `# noqa: B008` for ruff on function defaults.
- **CSRF**: auto-injected via fetch monkey-patch in PageLayout — no manual header needed in frontend.
- **API routes**: define static paths (`/search`) before param routes (`/{id}`) to prevent FastAPI mismatching.

## Adding a New Package to the Workspace

1. Create `packages/<name>/` with `pyproject.toml` and `src/` layout
2. Update root `pyproject.toml`: add to `[tool.uv.workspace].members`, `[dependency-groups].dev`, and `[tool.uv.sources]`
3. Run `uv sync`

## Important File Locations

- Root config: `pyproject.toml` (workspace definition, ruff, pyright, pytest config)
- Public API exports: `packages/exo-core/src/exo/__init__.py`
- Provider resolution: `packages/exo-models/`
- Web app entry: `packages/exo-web/src/exo_web/app.py`
- DB migrations: `packages/exo-web/src/exo_web/migrations/`
- Handle types (keep in sync): `packages/exo-web/src/islands/Canvas/handleTypes.ts` ↔ `routes/tools.py` (`_NODE_HANDLE_MAP`)

---

## Audit & Ongoing Work

An 83-finding audit report lives at `/home/atg/Github/orbiter-ai/audit.md`. It covers Bugs & Security, Logical Issues, Non-Completeness, Inconsistencies, and Duplications across all 15 packages. Consult it for context on every session below.

### Approach: Parallel Sub-agents

For multi-file work across multiple packages, use `TeamCreate` + multiple `Task` agents (`general-purpose` subagent_type) working in parallel — one agent per package. This pattern has been validated and cuts wall-clock time by ~3x.

```
TeamCreate → TaskCreate (per package) → Task(..., team_name=...) x N → all complete → TeamDelete
```

### Logging conventions (two patterns, do NOT mix)

- **exo-core internal files** (`_internal/`): `from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]` → `_log = get_logger(__name__)`
- **exo-models / exo-mcp** (and all other external packages): `import logging` → `logger = logging.getLogger(__name__)`

---

### Session 1 — 2026-02-19: Asyncio Error Clarity + Logging Coverage

**What was done:** Two bug fixes + logging coverage across 3 packages, 11 files.

**How:** `TeamCreate` with 3 concurrent `general-purpose` sub-agents (one per package).

**Bug fixes:**
1. **`ExceptionGroup` opacity in parallel execution** — `asyncio.TaskGroup` wraps all child failures in an opaque `ExceptionGroup`. Fixed in both parallel execution sites by wrapping the `_run_one` body in `try/except Exception as exc: raise TypedError(...) from exc`, then catching the group with `except* TypedError as eg` to produce a joined, human-readable error message.
2. **MCPToolWrapper race condition on lazy reconnect** — added `asyncio.Lock` with double-checked locking pattern in `execute()` so concurrent coroutines don't create duplicate connections.

**Files changed:**

| Package | File | Change |
|---|---|---|
| exo-core | `_internal/handlers.py` | `except*` ExceptionGroup fix in `_run_parallel`; `_log` + debug/warning logging throughout |
| exo-core | `_internal/agent_group.py` | `except*` ExceptionGroup fix in `ParallelGroup.run`; `_log` + debug logging |
| exo-core | `_internal/state.py` | `_log` + state transition logging (`→ RUNNING/SUCCESS/FAILED/TIMEOUT`) in all `RunNode`/`RunState` methods |
| exo-core | `_internal/background.py` | `_log` + lifecycle logging in `submit`, `handle_result`, `handle_error` |
| exo-models | `provider.py` | `logger` + resolved provider debug log in `get_provider()` |
| exo-models | `openai.py` | `logger` + debug before call + error with `exc_info=True` in `complete()` and `stream()` |
| exo-models | `anthropic.py` | Same pattern as `openai.py` |
| exo-models | `gemini.py` | Same pattern (catches bare `Exception`) |
| exo-models | `vertex.py` | Same pattern |
| exo-mcp | `tools.py` | `asyncio.Lock` double-check reconnect; `cleanup()` try/finally; `logger.debug/error` throughout |
| exo-mcp | `client.py` | `logger.debug` on connect reuse, cache hit, tool list, call, disconnect |

**Tests after changes:** exo-core 809 ✓ · exo-models 163 ✓ · exo-mcp 231 ✓

---

### Next: Items from audit.md Recommended Fix Priority

The session above did **not** address any items from the audit's Recommended Fix Priority list. Start there next.

**Immediate priority — all in `exo-web`:**

| # | ID | File | Issue |
|---|---|---|---|
| 1 | **B-1** | `services/sandbox.py:87-88` | Sandbox escape: `_build_runner_script` uses `.replace()` to escape user code but misses `\r`, `\t`, `\0`, Unicode escapes. Fix: use `ast.literal_eval`-safe serialization (e.g. `repr()` or `json.dumps()`) instead of manual escaping. |
| 2 | **B-2** | `config.py:14` | Hardcoded default secret key `"change-me-in-production"` with no warning. Fix: add a startup assertion/check in `app.py` lifespan that rejects the default in non-debug mode. |
| 3 | **B-4** | `routes/webhooks.py:146-236` | Webhook trigger `POST /api/v1/webhooks/{workflow_id}/{hook_id}` has no auth. `url_token` stored in DB is never validated. Fix: fetch the hook row and compare `url_token` from the query param against the stored value. |

After those three, continue with **B-8** (token in logs), **L-2** (stream_agent bypasses tool loop), **B-15** (async callable instructions never awaited).

---

### Session 2 — 2026-04-04: Context Snapshot Persistence

**What was done:** New feature — persist the processed msg_list (after summarization, truncation, hook mutations) at end of each run as a `SnapshotMemory`. On the next run, load the snapshot instead of rebuilding from raw history. Non-destructive: raw history preserved for restoration.

**Key design decisions:**
- Toggle via `ContextConfig.enable_snapshots` (off by default, on for `navigator` mode)
- Instruction SystemMessages excluded from snapshots (regenerated fresh each run). `[Conversation Summary]` SystemMessages preserved.
- Deterministic IDs (`snapshot_{agent}_{conversation}`) — upsert replaces previous snapshot
- Freshness check: latest raw item ID + context config hash
- External `messages` parameter invalidates snapshot (handoff/swarm safety)
- `branch()` never copies snapshots to forks
- All snapshot operations wrapped in try/except — never breaks a run

**Files created:**

| Package | File | Content |
|---|---|---|
| exo-memory | `snapshot.py` | `SnapshotMemory`, `serialize_msg_list()`, `deserialize_msg_list()`, `compute_config_hash()`, `has_message_content()`, `make_snapshot()` |
| exo-memory | `tests/test_snapshot_persistence.py` | 26 unit tests |
| exo-core | `tests/test_context_snapshot.py` | 8 integration tests |

**Files changed:**

| Package | File | Change |
|---|---|---|
| exo-context | `config.py` | Added `enable_snapshots` field + navigator preset |
| exo-memory | `persistence.py` | Added `save_snapshot()`, `load_snapshot()`, `is_snapshot_fresh()` |
| exo-memory | `__init__.py` | Exports for `SnapshotMemory`, serialization helpers |
| exo-memory | `backends/sqlite.py` | `_extra_fields()` + `_row_to_item()` snapshot dispatch |
| exo-memory | `backends/postgres.py` | Same as sqlite |
| exo-core | `agent.py` | Snapshot load/save in `_run_inner()`, `_save_snapshot_if_enabled()`, `clear_snapshot()`, `branch()` exclusion |
| exo-core | `runner.py` | Snapshot load/save in `_stream()`, `_save_stream_snapshot()` helper |

**Tests after changes:** exo-core 1524 ✓ · exo-memory 575 ✓ · exo-context 490 ✓

---

### Session 3 — 2026-04-05: Simplified Context Management API

**What was done:** Added a human-friendly API for context management (`context_limit`, `overflow`, `cache`) alongside the existing legacy API. Zero breaking changes — all 2593 tests pass without modification.

**Problem:** The old API exposed three interdependent thresholds (`history_rounds`, `summary_threshold`, `offload_threshold`) that formed a hidden priority cascade. Users had to understand internal implementation details to configure one simple concept: "what happens when the conversation gets too long."

**New API:**

```python
# Simple — on Agent directly
Agent(name="bot", context_limit=30, overflow="summarize", cache=True)

# Full control — ContextConfig
ContextConfig(limit=20, overflow="summarize", keep_recent=5, token_pressure=0.8, cache=True)
```

Three overflow strategies: `"summarize"` (LLM compression), `"truncate"` (drop oldest), `"none"` (no management).

**Design:**
- New fields are primary API; old fields stay as internal plumbing
- `model_validator(mode="before")` keeps both field sets in sync bidirectionally
- When new-API fields provided → old fields derived automatically
- When only old-API fields provided → new fields back-filled to match
- `OverflowStrategy` StrEnum exported from `exo.context`

**Files changed:**

| Package | File | Change |
|---|---|---|
| exo-context | `config.py` | `OverflowStrategy` enum, 5 new fields (`limit`, `overflow`, `keep_recent`, `token_pressure`, `cache`), `_normalize_api_fields` validator |
| exo-context | `__init__.py` | Export `OverflowStrategy` |
| exo-core | `agent.py` | 3 new Agent params (`context_limit`, `overflow`, `cache`), mutual exclusion with `context`/`context_mode`, overflow dispatch in `_apply_context_windowing()` |
| exo-core | `runner.py` | Fixed token_budget_trigger to unwrap `.config` (was reading from Context directly, using fallback by coincidence) |
| exo-core | `swarm.py` | 3 new Swarm params, propagation to member agents |
| skills | `context/SKILL.md` | Rewritten with new API as primary, legacy as secondary |
| exo-context | `README.md` | Updated quick start with new API |

**Key backward-compat decisions:**
- `ContextConfig()` bare constructor produces identical behavior to before (old defaults unchanged)
- `make_config()` and `AutomationMode` still work (old fields passed, new fields back-filled)
- `getattr(_cfg, "overflow", "summarize")` fallback means old duck-typed FakeConfig objects in tests work unchanged
- `compute_config_hash()` reads old field names — no snapshot invalidation on upgrade

**Tests after changes:** exo-core 1524 ✓ · exo-memory 575 ✓ · exo-context 490 ✓ (2593 total, 0 failures)

---

### Session 4 — 2026-04-05: Hook-Based Custom Context Management

**What was done:** New `overflow="hook"` strategy + `CONTEXT_WINDOW` hook point that lets users implement custom context management via hooks, with a rich `ContextWindowInfo` snapshot.

**Problem:** The three built-in overflow strategies (`summarize`, `truncate`, `none`) were closed. Users needing custom context management (semantic compression, importance-based retention, external memory offload) had no extension point.

**Design:**
- `overflow="hook"` delegates context windowing entirely to `CONTEXT_WINDOW` hooks (built-in cascade skipped)
- `overflow="summarize"|"truncate"` + registered `CONTEXT_WINDOW` hook fires as a post-processing augmentation pass
- `overflow="none"` never fires the hook (user explicitly opted out)
- Hooks mutate `messages` list in place (same pattern as `PRE_LLM_CALL`)
- `ContextWindowInfo` frozen dataclass provides rich read-only context: step position, message counts by type, token pressure (fill_ratio, context_window, input/output tokens), cumulative trajectory, config values, and agent identity
- `ContextWindowHook` ABC for typed convenience (optional — plain async functions work too)
- Hook fires at both windowing sites: initial (pre-first-LLM, step=-1) and token budget trigger (mid-run, force=True)

**Files created:**

| Package | File | Content |
|---|---|---|
| exo-context | `info.py` | `ContextWindowInfo` dataclass, `build_context_window_info()` builder |
| exo-context | `hook.py` | `ContextWindowHook` ABC |
| exo-core | `tests/test_context_window_hook.py` | 9 integration tests |
| exo-context | `tests/test_context_window_info.py` | 10 unit tests |

**Files changed:**

| Package | File | Change |
|---|---|---|
| exo-context | `config.py` | `HOOK = "hook"` in `OverflowStrategy`, `_normalize_api_fields` branch for hook |
| exo-context | `__init__.py` | Exports for `ContextWindowInfo`, `ContextWindowHook`, `build_context_window_info` |
| exo-core | `hooks.py` | `CONTEXT_WINDOW = "context_window"` in `HookPoint` |
| exo-core | `agent.py` | `_apply_context_windowing()` gains hook dispatch (overflow="hook" and augmentation), new params threaded; call sites updated; summarize cascade indented under `elif` |
| exo-core | `runner.py` | Model/context-window resolution moved before windowing; both call sites pass hook params |
| exo-core | `tests/test_hooks.py` | Updated HookPoint count (7→8) and expected names |
| skills | `hooks/SKILL.md` | `CONTEXT_WINDOW` in hook table, patterns, gotchas |
| skills | `context/SKILL.md` | `overflow="hook"` in decision guide, strategies table, full reference section, gotchas |

**Tests after changes:** exo-core 1533 ✓ · exo-context 500 ✓ (2033 total, 0 failures)
