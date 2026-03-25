# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orbiter

Orbiter is a modular multi-agent framework for building LLM-powered applications in Python. It's a UV workspace monorepo with 18 packages. Requires Python 3.11+.

## Common Commands

```bash
# Install all workspace packages (editable mode)
uv sync

# Run all tests (~2,900 tests, asyncio_mode=auto)
uv run pytest

# Run tests for a single package
uv run pytest packages/orbiter-core/tests/

# Run a single test file
uv run pytest packages/orbiter-core/tests/test_agent.py

# Run a single test
uv run pytest packages/orbiter-core/tests/test_agent.py::test_function_name

# Lint (with auto-fix)
uv run ruff check packages/ --fix

# Format check
uv run ruff format --check packages/

# Type-check a package
uv run pyright packages/orbiter-core/

# Verify installation
uv run python -c "from orbiter import Agent, run, tool; print('OK')"
```

### orbiter-web (dual Node+Python package)

```bash
cd packages/orbiter-web

# Install frontend deps
npm install

# Dev server (runs Astro + FastAPI concurrently)
npm run dev

# Astro typecheck
npx astro check

# Run backend only
uv run uvicorn orbiter_web.app:app --reload
```

## Architecture

UV workspace monorepo. Packages live in `packages/`. The dependency graph flows upward from `orbiter-core`:

```
orbiter-core (foundation, only depends on pydantic)
    ↑
orbiter-models (OpenAI, Anthropic, Gemini, Vertex AI providers)
    ↑
orbiter-context, orbiter-memory, orbiter-mcp, orbiter-sandbox, orbiter-observability, orbiter-guardrail
    ↑
orbiter-retrieval, orbiter-perplexica, orbiter-cli, orbiter-server, orbiter-eval, orbiter-a2a, orbiter-train, orbiter-web
    ↑
orbiter (meta-package, re-exports everything)
```

### Key Packages

- **orbiter-core** (`packages/orbiter-core/src/orbiter/`): `Agent`, `Tool`, `@tool` decorator, `run`/`run.sync`/`run.stream`, `Swarm`, hooks, events, config, registry. The `_internal/` subpackage has message building, output parsing, call execution, state machine, and graph algorithms.
- **orbiter-models** (`packages/orbiter-models/`): LLM provider implementations. Provider SDKs are isolated here — core has zero heavy deps.
- **orbiter-guardrail** (`packages/orbiter-guardrail/`): Security guardrails — pattern-based and LLM-based prompt injection/jailbreak detection with pluggable backends.
- **orbiter-retrieval** (`packages/orbiter-retrieval/`): RAG pipeline — embeddings (OpenAI, Vertex, HTTP), vector stores (pgvector, ChromaDB), hybrid search, reranking, knowledge graph, agentic retrieval.
- **orbiter-perplexica** (`packages/orbiter-perplexica/`): AI search engine with query classification, parallel research agents, result reranking, citation generation, and 3 quality modes (speed/balanced/quality).
- **orbiter-web** (`packages/orbiter-web/`): Full platform UI. Hybrid package — Astro 5.x frontend (`src/pages/`, `src/islands/`) + FastAPI backend (`src/orbiter_web/`). Has its own `package.json` AND `pyproject.toml`.

### orbiter-web Backend Structure

- `app.py` — FastAPI app entry point, middleware, route registration
- `config.py` — Settings dataclass (env vars: `ORBITER_DATABASE_URL`, `ORBITER_SECRET_KEY`, `ORBITER_DEBUG`)
- `database.py` — `get_db()` async context manager, WAL mode, foreign keys
- `engine.py` — Workflow execution engine (topological sort, node execution, retry)
- `migrations/` — Sequential SQL files, run automatically on startup via lifespan
- `routes/` — 30+ APIRouter modules, all under `/api/v1/` prefix
- `services/` — Business logic layer (agent runtime, sandbox, scheduler, memory)
- `middleware/` — CSRF, rate limiting, security headers, API version redirect

### orbiter-web Frontend Structure

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
- Public API exports: `packages/orbiter-core/src/orbiter/__init__.py`
- Provider resolution: `packages/orbiter-models/`
- Web app entry: `packages/orbiter-web/src/orbiter_web/app.py`
- DB migrations: `packages/orbiter-web/src/orbiter_web/migrations/`
- Handle types (keep in sync): `packages/orbiter-web/src/islands/Canvas/handleTypes.ts` ↔ `routes/tools.py` (`_NODE_HANDLE_MAP`)

---

## Audit & Ongoing Work

An 83-finding audit report lives at `/home/atg/Github/orbiter-ai/audit.md`. It covers Bugs & Security, Logical Issues, Non-Completeness, Inconsistencies, and Duplications across all 15 packages. Consult it for context on every session below.

### Approach: Parallel Sub-agents

For multi-file work across multiple packages, use `TeamCreate` + multiple `Task` agents (`general-purpose` subagent_type) working in parallel — one agent per package. This pattern has been validated and cuts wall-clock time by ~3x.

```
TeamCreate → TaskCreate (per package) → Task(..., team_name=...) x N → all complete → TeamDelete
```

### Logging conventions (two patterns, do NOT mix)

- **orbiter-core internal files** (`_internal/`): `from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]` → `_log = get_logger(__name__)`
- **orbiter-models / orbiter-mcp** (and all other external packages): `import logging` → `logger = logging.getLogger(__name__)`

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
| orbiter-core | `_internal/handlers.py` | `except*` ExceptionGroup fix in `_run_parallel`; `_log` + debug/warning logging throughout |
| orbiter-core | `_internal/agent_group.py` | `except*` ExceptionGroup fix in `ParallelGroup.run`; `_log` + debug logging |
| orbiter-core | `_internal/state.py` | `_log` + state transition logging (`→ RUNNING/SUCCESS/FAILED/TIMEOUT`) in all `RunNode`/`RunState` methods |
| orbiter-core | `_internal/background.py` | `_log` + lifecycle logging in `submit`, `handle_result`, `handle_error` |
| orbiter-models | `provider.py` | `logger` + resolved provider debug log in `get_provider()` |
| orbiter-models | `openai.py` | `logger` + debug before call + error with `exc_info=True` in `complete()` and `stream()` |
| orbiter-models | `anthropic.py` | Same pattern as `openai.py` |
| orbiter-models | `gemini.py` | Same pattern (catches bare `Exception`) |
| orbiter-models | `vertex.py` | Same pattern |
| orbiter-mcp | `tools.py` | `asyncio.Lock` double-check reconnect; `cleanup()` try/finally; `logger.debug/error` throughout |
| orbiter-mcp | `client.py` | `logger.debug` on connect reuse, cache hit, tool list, call, disconnect |

**Tests after changes:** orbiter-core 809 ✓ · orbiter-models 163 ✓ · orbiter-mcp 231 ✓

---

### Next: Items from audit.md Recommended Fix Priority

The session above did **not** address any items from the audit's Recommended Fix Priority list. Start there next.

**Immediate priority — all in `orbiter-web`:**

| # | ID | File | Issue |
|---|---|---|---|
| 1 | **B-1** | `services/sandbox.py:87-88` | Sandbox escape: `_build_runner_script` uses `.replace()` to escape user code but misses `\r`, `\t`, `\0`, Unicode escapes. Fix: use `ast.literal_eval`-safe serialization (e.g. `repr()` or `json.dumps()`) instead of manual escaping. |
| 2 | **B-2** | `config.py:14` | Hardcoded default secret key `"change-me-in-production"` with no warning. Fix: add a startup assertion/check in `app.py` lifespan that rejects the default in non-debug mode. |
| 3 | **B-4** | `routes/webhooks.py:146-236` | Webhook trigger `POST /api/v1/webhooks/{workflow_id}/{hook_id}` has no auth. `url_token` stored in DB is never validated. Fix: fetch the hook row and compare `url_token` from the query param against the stored value. |

After those three, continue with **B-8** (token in logs), **L-2** (stream_agent bypasses tool loop), **B-15** (async callable instructions never awaited).
