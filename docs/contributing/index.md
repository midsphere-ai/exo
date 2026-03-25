# Contributing to Exo

Thank you for your interest in contributing to Exo. This section covers everything you need to get started.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/anthropics/exo.git
cd exo

# Install dependencies (requires UV)
uv sync

# Run tests
uv run pytest

# Run linting
uv run ruff check packages/

# Run type checking
uv run pyright packages/exo-core/
```

## Contribution Areas

| Area | Docs | Description |
|------|------|-------------|
| Development setup | [Development](./development.md) | Environment setup, running tests, linting, type checking |
| Code style | [Code Style](./code-style.md) | Formatting, type hints, docstrings, Pydantic conventions |
| Package structure | [Package Structure](./package-structure.md) | Namespace packages, `__init__.py` patterns, layout |
| Testing | [Testing](./testing.md) | Test naming, async patterns, mocking, pyright workarounds |

## Workflow

1. **Find or create an issue** describing the change you want to make
2. **Create a feature branch** from `main`
3. **Make your changes** following the [code style](./code-style.md) and [package structure](./package-structure.md) guidelines
4. **Write tests** following the [testing guide](./testing.md)
5. **Run quality checks** -- all of these must pass:
   ```bash
   uv run pytest                           # tests
   uv run ruff check packages/             # linting
   uv run pyright packages/exo-core/   # type checking
   ```
6. **Commit with a descriptive message** following conventional commits:
   ```
   feat: add retry logic to tool execution
   fix: handle empty tool arguments in output parser
   docs: add migration guide for context engine
   ```
7. **Open a pull request** against `main`

## Architecture Overview

Before contributing, read the [Architecture Overview](../architecture/index.md) to understand:

- The 13-package monorepo structure
- The dependency DAG (exo-core at the bottom, everything depends on it)
- The execution flow (run -> call_runner -> agent.run -> tool loop)
- The design decisions (why single Agent class, why async-first, etc.)

## Key Principles

- **Max ~200 lines per source file.** If a file grows beyond this, split it into `_internal/` submodules.
- **Async-first.** All internal functions are `async def`. No sync/async duplication.
- **Type-safe.** Full pyright strict-mode compliance. All public functions must have type hints.
- **Composition over inheritance.** Max 2 levels of inheritance. Prefer tools, hooks, and processors.
- **Zero heavy deps in core.** `exo-core` depends only on `pydantic`. Provider SDKs live in `exo-models`.
