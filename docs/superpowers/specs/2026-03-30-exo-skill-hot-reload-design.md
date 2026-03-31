# Dynamic Skill Hot-Reloading for Exo

**Date:** 2026-03-30
**Status:** Draft
**Scope:** exo-core (abstractions) + new exo-skills package (concrete watchers)

## Context

Exo's `SkillRegistry` currently loads skills via a one-shot `load_all()` call. In production, long-running agents need to pick up new, modified, or removed skills without restarting. The existing Agent class already supports runtime tool mutation (`add_tool`, `remove_tool`, instruction reassignment, tool schema re-enumeration each step), so the infrastructure for mid-execution changes is in place. What's missing is the change detection and propagation layer.

## Goals

1. Detect skill file changes in real-time for local sources, periodically for remote (GitHub) sources
2. Push changes into running agents mid-execution (not just on next `run()`)
3. Provide an extensible watcher abstraction so any source type can implement its own change detection
4. Emit change events onto an EventBus for observability (logging, metrics, UI)
5. Maintain full backward compatibility with existing SkillRegistry API

## Non-Goals

- Python module hot-reloading (skills are metadata, not executable code)
- Automatic tool implementation discovery (users provide the tool mapping)
- Skill versioning or rollback

## Architecture

### Data Flow

```
skill.md changes
  -> Watcher detects (filesystem events or polling)
  -> yields list[SkillChangeEvent] batch
  -> SkillSyncManager receives batch
  -> Updates SkillRegistry._skills dict
  -> Calls ToolResolver to map skills to Tool objects
  -> Calls agent.add_tool()/remove_tool() for each bound agent
  -> Optionally rebuilds agent.instructions from active skills
  -> Emits events onto EventBus ("skill:added", "skill:modified", "skill:removed")
  -> Agent picks up tool changes on next step (schemas re-enumerated each step)
```

### Package Layout

```
packages/exo-core/src/exo/
    skills.py              <-- add: SkillChangeEvent, SkillWatcher ABC, ToolResolver Protocol,
                                    DictToolResolver, SkillSyncManager

packages/exo-skills/                   <-- NEW PACKAGE
    pyproject.toml                     <-- depends on exo-core + watchfiles
    src/exo_skills/
        __init__.py                    <-- re-exports LocalFileWatcher, GitHubPollingWatcher
        watchers/
            __init__.py
            local.py                   <-- LocalFileWatcher (watchfiles-backed)
            github.py                  <-- GitHubPollingWatcher (git pull polling)
```

## Core Abstractions (in exo-core/src/exo/skills.py)

### SkillChangeEvent

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class SkillChangeEvent:
    """Represents a single skill change detected by a watcher."""
    kind: Literal["added", "modified", "removed"]
    skill_name: str
    skill: Skill | None          # None only for kind="removed"
    source_path: str             # source directory this came from
```

### SkillWatcher ABC

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

class SkillWatcher(ABC):
    """Base class for skill source watchers.

    Implementations monitor a skill source (local directory, GitHub repo,
    etc.) and yield batches of change events. Each batch represents one
    "settle" cycle — filesystem events are debounced, polling results are
    diffed.
    """

    @abstractmethod
    async def watch(self) -> AsyncIterator[list[SkillChangeEvent]]:
        """Yield batches of change events. Blocks between batches."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Signal the watcher to shut down. Must cause watch() to return."""
        ...
```

The async iterator contract means each watcher controls its own cadence. Filesystem watchers yield on OS events; polling watchers yield on timer ticks.

### ToolResolver Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ToolResolver(Protocol):
    """Maps skill metadata to actual Tool implementations."""
    def resolve(self, skill: Skill) -> list[Tool]: ...
```

### DictToolResolver

```python
class DictToolResolver:
    """Simple resolver that maps skill names to Tool objects via a dict."""

    def __init__(self, tool_map: dict[str, Tool | list[Tool]]) -> None:
        self._map = tool_map

    def resolve(self, skill: Skill) -> list[Tool]:
        entry = self._map.get(skill.name)
        if entry is None:
            return []
        if isinstance(entry, list):
            return entry
        return [entry]
```

### SkillSyncManager

```python
class SkillSyncManager:
    """Orchestrates skill watchers and pushes changes to bound agents.

    Args:
        registry: The SkillRegistry whose state is kept in sync.
        tool_resolver: Maps skills to Tool objects. Accepts a ToolResolver
            instance or a plain dict[str, Tool | list[Tool]].
        event_bus: Optional EventBus for emitting change events.
        instructions_builder: Optional callable that rebuilds agent
            instructions from the current list of active skills.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        tool_resolver: ToolResolver | dict[str, Tool | list[Tool]],
        *,
        event_bus: EventBus | None = None,
        instructions_builder: Callable[[list[Skill]], str] | None = None,
    ) -> None: ...

    def add_watcher(self, watcher: SkillWatcher) -> None:
        """Register a watcher. Must be called before start()."""

    def bind_agent(self, agent: Agent) -> None:
        """Bind an agent to receive skill change updates."""

    def unbind_agent(self, agent: Agent) -> None:
        """Stop pushing updates to an agent."""

    async def start(self) -> None:
        """Start all watchers as background asyncio.Tasks."""

    async def stop(self) -> None:
        """Stop all watchers, cancel tasks, await cleanup."""

    async def __aenter__(self) -> SkillSyncManager: ...
    async def __aexit__(self, *exc) -> None: ...
```

**Internal state:**

| Field | Type | Purpose |
|-------|------|---------|
| `_watchers` | `list[SkillWatcher]` | Registered watchers |
| `_agents` | `set[Agent]` | Bound agents |
| `_tasks` | `list[asyncio.Task]` | One per watcher |
| `_skill_tools` | `dict[str, list[str]]` | Tracks tool names added per skill, for cleanup on removal |
| `_resolver` | `ToolResolver` | Wrapped from init arg (dict → DictToolResolver) |
| `_bus` | `EventBus \| None` | Optional event emission |
| `_instructions_builder` | `Callable \| None` | Rebuilds agent instructions |

**Change propagation per event:**

- **added**: Add skill to `registry._skills`, resolve tools, call `agent.add_tool()` for each, track in `_skill_tools`, emit `"skill:added"` on bus
- **modified**: Remove old tools (from `_skill_tools`), add new ones, update registry, emit `"skill:modified"`
- **removed**: Remove tools (from `_skill_tools`), delete from registry, emit `"skill:removed"`
- After each batch: if `instructions_builder` is set, rebuild and reassign `agent.instructions` for all bound agents

**Error handling:**

- Watcher exceptions are caught and logged (don't crash the manager)
- Individual agent.add_tool/remove_tool failures are caught and logged (one agent failing doesn't block others)
- Watcher restart: on error, wait with exponential backoff and retry the watch() loop

## Concrete Watchers (exo-skills package)

### LocalFileWatcher

```python
class LocalFileWatcher(SkillWatcher):
    """Watches a local directory for skill.md changes using watchfiles.

    Args:
        path: Directory to watch (recursive).
        debounce_ms: Milliseconds to wait for filesystem to settle.
    """

    def __init__(self, path: str | Path, debounce_ms: int = 500) -> None: ...
```

**Implementation details:**
- Uses `watchfiles.awatch(path, watch_filter=skill_filter)` where `skill_filter` only passes `skill.md` / `SKILL.md` files
- Maintains a snapshot `dict[str, Skill]` of previously-seen skills
- On each change set from watchfiles:
  - Re-reads changed files via `extract_front_matter()`
  - Diffs against snapshot to produce SkillChangeEvent list
  - Updates snapshot
  - Yields the event batch
- `stop()` sets an `asyncio.Event` and the watch loop checks it

### GitHubPollingWatcher

```python
class GitHubPollingWatcher(SkillWatcher):
    """Polls a GitHub repository for skill changes via periodic git pull.

    Args:
        source_url: GitHub URL (parsed via parse_github_url).
        poll_interval: Seconds between polls (default 300 = 5 min).
        cache_dir: Override for clone cache directory.
    """

    def __init__(
        self,
        source_url: str,
        poll_interval: float = 300.0,
        cache_dir: Path | None = None,
    ) -> None: ...
```

**Implementation details:**
- Uses existing `parse_github_url()` and `_clone_github()` for initial setup
- On each poll: runs `git -C <clone_dir> pull --ff-only` via `asyncio.create_subprocess_exec`
- Diffs skill state before/after pull using `_collect_skills()`
- Skips yielding when nothing changed (no unnecessary empty batches)
- `stop()` sets an `asyncio.Event` checked between polls

## EventBus Integration

The SkillSyncManager optionally emits events onto a provided `EventBus`:

| Event | Payload |
|-------|---------|
| `"skill:added"` | `skill: Skill` |
| `"skill:modified"` | `old_skill: Skill, new_skill: Skill` |
| `"skill:removed"` | `skill: Skill` |

This enables observability without coupling:
- Logging subscriber: logs all changes
- Metrics subscriber: counts adds/removes/modifications
- UI subscriber: notifies dashboard of capability changes

## Usage Examples

### Basic Hot-Reload

```python
from exo import Agent, run, tool
from exo.skills import SkillRegistry, SkillSyncManager
from exo_skills.watchers import LocalFileWatcher

@tool
async def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

registry = SkillRegistry()
registry.register_source("./skills")
registry.load_all()

agent = Agent(name="assistant", model="openai:gpt-4o", tools=[web_search])

async with SkillSyncManager(
    registry=registry,
    tool_resolver={"web_search": web_search},
) as manager:
    manager.add_watcher(LocalFileWatcher("./skills"))
    manager.bind_agent(agent)
    await manager.start()

    # Agent now auto-picks up skill changes
    result = await run(agent, "Search for AI news")
```

### With EventBus Observability

```python
from exo.events import EventBus

bus = EventBus()

async def log_skill_change(**data):
    print(f"Skill change: {data}")

bus.on("skill:added", log_skill_change)
bus.on("skill:modified", log_skill_change)
bus.on("skill:removed", log_skill_change)

async with SkillSyncManager(
    registry=registry,
    tool_resolver=tool_map,
    event_bus=bus,
) as manager:
    ...
```

### With Instructions Rebuilding

```python
def build_instructions(skills: list[Skill]) -> str:
    parts = ["You have these capabilities:\n"]
    for s in skills:
        parts.append(f"- **{s.name}**: {s.description}")
        if s.usage:
            parts.append(f"  {s.usage.splitlines()[0]}")
    return "\n".join(parts)

async with SkillSyncManager(
    registry=registry,
    tool_resolver=tool_map,
    instructions_builder=build_instructions,
) as manager:
    ...
```

### Multi-Source with GitHub Polling

```python
from exo_skills.watchers import LocalFileWatcher, GitHubPollingWatcher

async with SkillSyncManager(registry=registry, tool_resolver=tool_map) as manager:
    manager.add_watcher(LocalFileWatcher("./local-skills"))
    manager.add_watcher(GitHubPollingWatcher(
        "https://github.com/org/shared-skills/tree/main/skills",
        poll_interval=600,  # 10 minutes
    ))
    manager.bind_agent(agent)
    await manager.start()
```

## Testing Strategy

### Unit Tests (exo-core)

- **SkillChangeEvent**: Construction, frozen immutability, None skill for removals
- **DictToolResolver**: Single tool, list of tools, missing skill returns empty list
- **SkillSyncManager**:
  - Use `FakeWatcher` that yields pre-built event batches from a list
  - Verify registry state updates after events
  - Verify `agent.add_tool()`/`remove_tool()` calls (mock agent)
  - Verify `_skill_tools` tracking for cleanup
  - Verify EventBus emissions
  - Verify instructions_builder called after each batch
  - Verify error handling (watcher exception, agent mutation failure)
  - Verify start/stop lifecycle and context manager

### Integration Tests (exo-skills)

- **LocalFileWatcher**: Use `tmp_path`, write/modify/delete skill.md files, assert correct events within timeout
- **GitHubPollingWatcher**: Mock `asyncio.create_subprocess_exec` for git pull, verify diff logic

### Test Patterns

- All tests async (asyncio_mode=auto)
- Use `MockProvider` for agent tests
- No real API calls or network access
- Unique test file names across packages

## Files to Modify

| File | Change |
|------|--------|
| `packages/exo-core/src/exo/skills.py` | Add SkillChangeEvent, SkillWatcher, ToolResolver, DictToolResolver, SkillSyncManager |
| `packages/exo-core/tests/test_skills.py` | Add tests for new types and SkillSyncManager |
| `packages/exo-skills/pyproject.toml` | New package (depends on exo-core, watchfiles) |
| `packages/exo-skills/src/exo_skills/__init__.py` | Re-export watchers |
| `packages/exo-skills/src/exo_skills/watchers/__init__.py` | Re-export LocalFileWatcher, GitHubPollingWatcher |
| `packages/exo-skills/src/exo_skills/watchers/local.py` | LocalFileWatcher implementation |
| `packages/exo-skills/src/exo_skills/watchers/github.py` | GitHubPollingWatcher implementation |
| `packages/exo-skills/tests/test_local_watcher.py` | LocalFileWatcher integration tests |
| `packages/exo-skills/tests/test_github_watcher.py` | GitHubPollingWatcher tests |
| `pyproject.toml` (root) | Add exo-skills to workspace members, dev deps, uv.sources |

## Verification

1. `uv sync` succeeds with new package
2. `uv run pytest packages/exo-core/tests/test_skills.py` — all existing + new tests pass
3. `uv run pytest packages/exo-skills/` — watcher tests pass
4. `uv run ruff check packages/exo-core/ packages/exo-skills/` — no lint errors
5. `uv run ruff format --check packages/exo-core/ packages/exo-skills/` — formatted
6. Manual smoke test: start an agent with LocalFileWatcher, edit a skill.md, verify agent picks up the change
