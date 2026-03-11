# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Orbiter is a modular multi-agent framework for building LLM-powered applications in Python. It's a UV workspace monorepo with 18 packages under `packages/`, all sharing the `orbiter` namespace.

## Common Commands

```bash
# Install all packages (editable) + dev dependencies
uv sync

# Run all tests
uv run pytest

# Run tests for a specific package
uv run pytest packages/orbiter-core/tests/

# Run a single test
uv run pytest packages/orbiter-core/tests/test_agent.py::test_agent_calls_tool

# Lint and format
uv run ruff check packages/
uv run ruff check packages/ --fix
uv run ruff format packages/

# Type check
uv run pyright packages/orbiter-core/
```

## Architecture

**Package dependency DAG** (bottom-up):
- `orbiter-core` — Agent, Tool, Swarm, Runner, hooks, events, types. Only depends on `pydantic`.
- `orbiter-models` — LLM provider abstractions (OpenAI, Anthropic, Gemini, Vertex AI). Depends on core.
- `orbiter-context`, `orbiter-memory`, `orbiter-mcp`, `orbiter-sandbox`, `orbiter-guardrail`, `orbiter-retrieval`, `orbiter-observability`, `orbiter-distributed` — Feature packages depending on core/models.
- `orbiter-eval`, `orbiter-a2a`, `orbiter-train`, `orbiter-perplexica` — Higher-level packages.
- `orbiter-cli`, `orbiter-server`, `orbiter-web` — Top-level entry points.
- `orbiter` — Meta-package that re-exports everything.

**Key execution flow:** `run()` → `call_runner` → `agent.run` → tool loop

**Core source layout:** `packages/orbiter-core/src/orbiter/` with public modules (`agent.py`, `tool.py`, `runner.py`, `swarm.py`, `types.py`, `hooks.py`, `events.py`, `config.py`) and `_internal/` for implementation details.

**Model string convention:** `"provider:model_name"` (e.g., `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-20250514"`). No colon defaults to `"openai"`.

## Code Conventions

- **Max ~200 lines per source file.** Split into `_internal/` submodules when larger. Test files can go up to ~300.
- **Async-first.** All internal functions are `async def`. Single `run.sync()` bridge for sync callers.
- **Pydantic v2 models** with `model_config = {"frozen": True}` for config/data classes. Use `@field_validator` not `@validator`.
- **Modern Python type syntax:** `X | None` not `Optional[X]`, lowercase `list`/`dict` not `List`/`Dict`.
- **Google-style docstrings** on public classes and functions.
- **Ruff:** line-length=100, double quotes, 4-space indent. Rules: E, F, I, N, W, UP, B, SIM, RUF.
- **All exceptions** inherit from `OrbiterError`.
- **Conventional commits:** `feat:`, `fix:`, `docs:`, `chore:`, etc.

## Testing Conventions

- `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed on async tests.
- `--import-mode=importlib` — test file names must be **unique across all packages** (prefix with package name to avoid collisions).
- **Never make real API calls** — always use mock providers.
- Cross-package imports in test files use `# pyright: ignore[reportMissingImports]`.
- Root `pyproject.toml` is NOT a package — it only configures workspace, pytest, ruff, and pyright.

## Environment Variables

- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — only needed for integration tests; unit tests use mocks.
