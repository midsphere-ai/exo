# Deep Agent Toolkit Design — Built-in Agent Tools

**Status:** Proposed
**Epic:** 8 — Deep Agent Toolkit
**Date:** 2026-03-10

---

## 1. Motivation

Orbiter agents can invoke arbitrary tools via the `Tool` ABC, but there is
no curated toolkit for **autonomous task execution** — the pattern where an
agent independently reads files, runs commands, executes code, and tracks
progress without human intervention.

Agent-core's `deepagents/tools/` provides four built-in tools for this
purpose:

| Agent-Core Tool   | Purpose                                      |
|-------------------|----------------------------------------------|
| `TodoTool`        | CRUD todo list with status tracking           |
| `FileSystemTool`  | File I/O (read, write, edit, glob, list, grep)|
| `ShellTool`       | Shell command execution                       |
| `CodeTool`        | Python code execution in isolated environment |

Orbiter already has equivalents for three of these, spread across two
packages:

| Capability     | Orbiter Package      | Module                        |
|----------------|----------------------|-------------------------------|
| Todo / Planning| `orbiter-context`    | `context/tools.py` — `get_planning_tools()` |
| File I/O       | `orbiter-sandbox`    | `sandbox/tools.py` — `FilesystemTool`       |
| Shell          | `orbiter-sandbox`    | `sandbox/tools.py` — `TerminalTool`         |
| Code execution | —                    | Not yet implemented                          |

The gap is a **CodeExecutionTool** for sandboxed Python execution, plus a
**convenience API** that bundles all four tool categories into a single
`get_deep_agent_tools()` call. Additionally, the existing `FilesystemTool`
lacks agent-core's `edit`, `glob`, and `grep` operations, and the
`orbiter-context` planning tools lack status tracking (PENDING /
IN_PROGRESS / COMPLETED).

---

## 2. Key Decision: Extend Existing Packages, No New Package

### Option A — New package `orbiter-toolkit` (rejected)

A separate package adds dependency overhead and forces a third import for
what is fundamentally a composition of existing tools. The tools themselves
belong logically to their current packages.

### Option B — All tools in `orbiter-core` (rejected)

Moving sandbox tools into core would create a circular dependency (core
depends on sandbox's `Sandbox` for execution context). It also collapses
the separation of concerns between core abstractions and execution
environments.

### Option C — Extend `orbiter-sandbox` + `orbiter-context`, aggregate in core (chosen)

1. **Code execution** → new `CodeExecutionTool` in `orbiter-sandbox/tools.py`
   (alongside `FilesystemTool` and `TerminalTool`), backed by `LocalSandbox`.
2. **File tool extensions** (edit, glob, grep) → extend `FilesystemTool` in
   `orbiter-sandbox/tools.py`.
3. **Todo status tracking** → extend planning tools in
   `orbiter-context/tools.py` with status field.
4. **Aggregator** → `get_deep_agent_tools()` function in
   `orbiter-core/src/orbiter/toolkit.py` that imports and bundles all four
   tool categories.

**Why Option C:**

- Keeps tools co-located with their execution environments (sandbox tools
  with sandbox, context tools with context).
- `orbiter-core` already depends on both packages, so the aggregator
  introduces no new dependency edges.
- Follows the established pattern: `get_planning_tools()`,
  `get_context_tools()`, `get_file_tools()` are already per-package
  convenience functions.

---

## 3. Component Design

### 3.1 CodeExecutionTool (new)

```python
class CodeExecutionTool(Tool):
    """Execute Python code in a sandboxed environment."""

    name = "execute_code"
    description = "Execute Python code and return stdout, stderr, and result."
    parameters = {
        "code": {"type": "string", "description": "Python code to execute"},
        "timeout": {"type": "number", "description": "Max seconds (default 30)"},
    }

    def __init__(
        self,
        *,
        sandbox: Sandbox | None = None,
        allowed_imports: frozenset[str] | None = None,
        timeout: float = 30.0,
    ) -> None: ...

    async def execute(self, **kwargs) -> dict[str, Any]:
        # Returns: {"stdout": str, "stderr": str, "result": str | None,
        #           "success": bool, "execution_time_ms": float}
```

Execution strategy:

- If a `Sandbox` instance is provided, delegate to `sandbox.run_tool()`.
- Otherwise, use `asyncio.create_subprocess_exec()` with a temporary
  script file, import allowlist enforcement via AST inspection, and
  `asyncio.wait_for()` timeout.
- The tool does **not** manage sandbox lifecycle — the caller (agent or
  builder) is responsible for starting/stopping sandboxes.

### 3.2 FilesystemTool Extensions

Add three new actions to the existing `FilesystemTool`:

| Action | Parameters                    | Description                              |
|--------|-------------------------------|------------------------------------------|
| `edit` | `path`, `old_text`, `new_text`| Replace first occurrence of old_text      |
| `glob` | `pattern`, `path` (optional)  | Find files matching glob pattern          |
| `grep` | `pattern`, `path` (optional)  | Search file contents with regex           |

These mirror agent-core's `FileSystemTool` operations and follow the
existing action-dispatch pattern in `FilesystemTool.execute()`.

### 3.3 Planning Tool Status Extension

Extend the todo state model in `orbiter-context/tools.py`:

```python
# Current: {"item": str, "done": bool}
# Extended: {"item": str, "done": bool, "status": str}
# status ∈ {"pending", "in_progress", "completed"}
```

- `add_todo` sets `status = "pending"` by default.
- New `update_todo_status(ctx, index, status)` tool transitions status.
- `complete_todo` sets both `done = True` and `status = "completed"`.
- Single-active constraint: at most one item may be `"in_progress"`.
- Backward compatible: existing callers that only check `done` are
  unaffected.

### 3.4 Toolkit Aggregator

```python
# orbiter-core/src/orbiter/toolkit.py

def get_deep_agent_tools(
    *,
    sandbox: Sandbox | None = None,
    workspace: list[str] | None = None,
    allowed_imports: frozenset[str] | None = None,
    terminal_blacklist: frozenset[str] | None = None,
    terminal_timeout: float = 30.0,
    code_timeout: float = 30.0,
    include_planning: bool = True,
    include_filesystem: bool = True,
    include_terminal: bool = True,
    include_code: bool = True,
) -> list[Tool]:
    """Return a curated set of tools for autonomous agent execution."""
```

This function assembles tools from both packages, passing through
configuration. Agents opt in with:

```python
agent = Agent(
    name="deep-agent",
    tools=[*get_deep_agent_tools(sandbox=my_sandbox), custom_tool],
)
```

---

## 4. Security Model

### 4.1 Filesystem Security

- **Path traversal prevention**: `FilesystemTool._validate_path()` resolves
  symlinks and rejects paths outside `allowed_directories`. This already
  exists and applies to all new actions (edit, glob, grep).
- **Glob restriction**: patterns are resolved relative to allowed
  directories only; absolute patterns that escape are rejected.
- **Write protection**: optional `read_only=True` mode disables write and
  edit actions.

### 4.2 Terminal Security

- **Command blacklist**: `TerminalTool` blocks destructive commands
  (`rm`, `mkfs`, `dd`, `shutdown`, `kill`, etc.) via configurable
  `blacklist` parameter. Default blacklist covers 14 dangerous commands.
- **Timeout enforcement**: `asyncio.wait_for()` kills long-running
  processes. Default 30s, configurable per-tool.
- **No shell expansion**: commands run via `asyncio.create_subprocess_exec()`
  (not `shell=True`) to prevent injection.

### 4.3 Code Execution Security

- **Import allowlist**: AST inspection of the code before execution blocks
  imports outside the allowed set. Default allowlist includes safe stdlib
  modules (`math`, `json`, `datetime`, `re`, `collections`, `itertools`,
  `functools`, `typing`, `dataclasses`, `pathlib`, `os.path`).
- **No network by default**: `socket`, `urllib`, `http`, `requests` are
  excluded from the default allowlist.
- **Subprocess isolation**: code runs in a separate process with its own
  memory space; parent process is unaffected by crashes.
- **Timeout**: hard timeout via `asyncio.wait_for()` on the subprocess.
- **Sandbox delegation**: when a `Sandbox` (e.g., `KubernetesSandbox`) is
  provided, all isolation is delegated to the sandbox environment, which
  may enforce container-level restrictions.

### 4.4 Planning Tools Security

- No security concerns — planning tools operate on in-memory state only,
  with no I/O side effects.

---

## 5. File Layout

```
packages/orbiter-sandbox/src/orbiter/sandbox/
├── tools.py              # + CodeExecutionTool, FilesystemTool extensions
└── ...

packages/orbiter-context/src/orbiter/context/
├── tools.py              # + update_todo_status, status field
└── ...

packages/orbiter-core/src/orbiter/
├── toolkit.py            # NEW — get_deep_agent_tools() aggregator
└── ...

packages/orbiter-sandbox/tests/
├── test_sandbox_tools.py # + tests for CodeExecutionTool, edit/glob/grep
└── ...

packages/orbiter-context/tests/
├── test_context_tools.py # + tests for todo status tracking
└── ...

packages/orbiter-core/tests/
├── test_toolkit.py       # NEW — aggregator tests
└── ...
```

---

## 6. Backward Compatibility

- **FilesystemTool**: new actions are additive. Existing `read`, `write`,
  `list` actions are unchanged. The `parameters` schema adds new enum
  values — callers that only use existing actions are unaffected.
- **TerminalTool**: no changes to the existing interface.
- **Planning tools**: the `status` field is added with a default value.
  Existing code that checks `todo["done"]` continues to work. The
  `complete_todo` function continues to set `done = True`.
- **Public API**: `get_deep_agent_tools()` is a new export. No existing
  exports are modified or removed.
- **Agent class**: no changes. Tools are passed via the existing
  `tools=[]` parameter.

---

## 7. Implementation Plan

| Story  | Title                              | Est. LOC |
|--------|------------------------------------|----------|
| US-059 | CodeExecutionTool implementation   | ~150     |
| US-060 | FilesystemTool edit/glob/grep      | ~120     |
| US-061 | Planning tool status tracking      | ~60      |
| US-062 | Toolkit aggregator                 | ~80      |
| US-063 | Integration tests                  | ~120     |

---

## 8. Open Questions

1. **Code execution output capture**: should `CodeExecutionTool` capture
   the return value of the last expression (like a REPL) or only
   stdout/stderr? **Recommendation:** capture both — use `exec()` for
   statements and `eval()` for expressions, with stdout capture via
   `contextlib.redirect_stdout`.

2. **File size limits**: should `FilesystemTool.read` and `grep` enforce
   maximum file size? **Recommendation:** yes, configurable with a
   sensible default (e.g., 1 MB for read, 10 MB for grep). This prevents
   agents from accidentally loading huge files into context.

3. **Sandbox requirement for CodeExecutionTool**: should code execution
   require an explicit `Sandbox` instance or allow bare subprocess
   execution? **Recommendation:** allow both, with a warning log when
   running without a sandbox in production mode.

---

## 9. Summary

The deep agent toolkit extends Orbiter's existing tool infrastructure
rather than introducing a parallel system:

- **CodeExecutionTool** joins `FilesystemTool` and `TerminalTool` in
  `orbiter-sandbox`, providing sandboxed Python execution.
- **FilesystemTool** gains `edit`, `glob`, and `grep` actions to match
  agent-core's file tool capabilities.
- **Planning tools** gain status tracking for autonomous task management.
- **`get_deep_agent_tools()`** in `orbiter-core` aggregates all tools
  behind a single convenience function.

Security is enforced at every layer: path validation for files, command
blacklists for terminal, import allowlists for code, and timeouts
everywhere. When a `Sandbox` is provided, container-level isolation adds
a further defense boundary.
