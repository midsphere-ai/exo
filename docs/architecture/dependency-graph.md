# Dependency Graph

This document details the package-level and module-level dependency structure of Exo.

## Package Dependency DAG

Exo's 13 packages form a strict dependency hierarchy. Arrows point from dependent to dependency.

```
                             exo (meta-package)
                                     |
            +--------+--------+------+------+--------+
            |        |        |      |      |        |
         cli     server    train   a2a    eval     ralph
            |        |        |      |      |        |
            +--------+--------+------+------+--------+
                                     |
            +--------+--------+------+------+--------+
            |        |        |             |        |
         models   context   memory        mcp    sandbox
            |        |        |             |        |
            +--------+--------+------+------+--------+
                                     |
                               exo-core
                            (pydantic only)
```

### Dependency Rules

1. **`exo-core` has ZERO heavy dependencies** -- only `pydantic`. No `openai`, no `anthropic`, no `httpx`. This means you can import `exo.types`, `exo.agent`, and `exo.tool` without pulling in any LLM SDK.

2. **Provider SDKs live in `exo-models` only.** The `openai` and `anthropic` packages are dependencies of `exo-models`, not `exo-core`.

3. **Optional heavy dependencies** (chromadb, kubernetes, etc.) are declared as extras in their respective packages. For example, `exo-memory[qdrant]` adds the Qdrant client.

4. **Lateral packages do not depend on each other.** `exo-models` does not depend on `exo-memory`. `exo-context` does not depend on `exo-sandbox`. This keeps the dependency graph a clean tree.

5. **`_internal/` modules are never imported from outside their package.** They are implementation details.

## Internal Module Dependencies (exo-core)

Within `exo-core`, modules have a clear dependency order:

```
                    runner.py
                    /       \
               swarm.py    _internal/call_runner.py
              /    |              |          \
         agent.py  |      _internal/state.py  \
        /  |  \    |              |         _internal/message_builder.py
       /   |   \   |              |                    |
  tool.py  |  hooks.py  _internal/graph.py            |
      |    |      |                                    |
  registry.py  config.py                              |
      |         |                                      |
      +---------+--------------------------------------+
                              |
                          types.py
                       (no internal deps)
```

### Module Descriptions

| Module | Dependencies | Purpose |
|--------|-------------|---------|
| `types.py` | none | Message types, RunResult, StreamEvent, Usage, ExoError |
| `config.py` | `types` | AgentConfig, ModelConfig, TaskConfig, RunConfig, parse_model_string |
| `registry.py` | `types` (ExoError) | Generic `Registry[T]`, agent_registry, tool_registry |
| `events.py` | none | `EventBus` for decoupled pub/sub |
| `hooks.py` | `types` | `HookPoint` enum, `Hook` type alias, `HookManager` |
| `tool.py` | `types`, `registry` | `Tool` ABC, `FunctionTool`, `@tool` decorator, schema generation |
| `agent.py` | `types`, `config`, `tool`, `hooks`, `_internal/*` | `Agent` class with tool loop |
| `swarm.py` | `agent`, `_internal/graph`, `_internal/call_runner` | `Swarm` multi-agent orchestration |
| `runner.py` | `agent`, `swarm`, `_internal/call_runner` | `run()`, `run.sync()`, `run.stream()` |
| `_internal/message_builder.py` | `types` | Build and validate message lists |
| `_internal/output_parser.py` | `types` | Parse LLM responses to AgentOutput |
| `_internal/call_runner.py` | `agent`, `_internal/state` | Core execution loop with state tracking |
| `_internal/state.py` | `types` | `RunState`, `RunNode`, status tracking |
| `_internal/graph.py` | none | Graph utilities, topological sort, flow DSL parser |

## Namespace Packages

Exo uses Python namespace packages so that multiple PyPI packages can contribute to the same `exo` import path. This is how `exo-core` provides `exo.types` while `exo-models` provides `exo.models`.

The mechanism uses `pkgutil.extend_path()` in `__init__.py`:

```python
"""Exo Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Public API re-exports
from exo.agent import Agent
from exo.runner import run
from exo.swarm import Swarm
from exo.tool import FunctionTool, Tool, tool
```

When multiple packages are installed (e.g., `exo-core` and `exo-models`), Python merges their `exo/` directories. The `extend_path()` call ensures that `import exo.models` finds the `exo/models/` directory contributed by `exo-models`, even though `exo-core` also has an `exo/` directory.

### Implications for Development

- Each package has its own `src/exo/` directory with an `__init__.py` that calls `extend_path()`
- Packages installed in editable mode (`uv pip install -e`) use `.pth`-based path entries
- Pyright may not resolve cross-namespace-package imports (e.g., `exo.models.types` from a test file in `exo-core`). Use `# pyright: ignore[reportMissingImports]` on such imports in test files.

## How to Add a New Package

1. Create `packages/exo-foo/` with the standard layout:
   ```
   packages/exo-foo/
   +-- pyproject.toml
   +-- src/
   |   +-- exo/
   |       +-- __init__.py       # extend_path + re-exports
   |       +-- foo/
   |           +-- __init__.py   # public API
   |           +-- ...
   +-- tests/
       +-- __init__.py
       +-- test_foo.py
   ```

2. Add the package to the workspace root `pyproject.toml` members list.

3. If the package depends on `exo-core`, add it to `[tool.uv.sources]` in the package's `pyproject.toml`.

4. Run `uv sync` to install the new package in the workspace.
