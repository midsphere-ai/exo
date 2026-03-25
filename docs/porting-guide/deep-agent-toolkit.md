# DeepAgent Toolkit — agent-core to Orbiter Mapping

**Epic:** 8 — Deep Agent Toolkit
**Date:** 2026-03-11

This document maps agent-core's (openJiuwen) deep_agent toolkit to
Orbiter's built-in tool system, covering file I/O, shell execution,
sandboxed code execution, todo/task CRUD, and the priority event queue.

---

## 1. Agent-Core Overview

Agent-core's deep_agent directory provides a full autonomous-agent toolkit
with file manipulation, shell access, code execution, task tracking, and
a priority event queue for mid-execution steering.

### File Tools

**`FileReaderTool`** — Reads file contents with optional line-range support
(1-indexed). Returns line-numbered output with UTF-8 encoding and fallback.

**`FileEditorTool`** — Line-based file editing supporting append, insert,
and replace operations. Multiple edits per call are applied bottom-to-top
to preserve line numbering.

**`FileAnalyzerTool`** — Reads local files or URLs, chunks large content,
and uses an LLM to perform task-driven analysis and extraction.

### Shell Execution

**`BashTool`** — Executes arbitrary shell commands via `asyncio.subprocess`.
Configurable timeout (30s default), captures stdout/stderr separately, and
returns the exit code. No command allowlist — the agent has full shell
access.

### Sandboxed Code Execution

**`PythonInterpreterTool`** — Runs Python code through
`smolagents.LocalPythonExecutor` with an authorized-imports allowlist
(pandas, numpy, matplotlib, scipy, sklearn, etc.). State resets between
executions for isolation.

### Todo CRUD System

**`TodoTool`** — Step-based task decomposition with JSON persistence and
synchronized markdown (`todo.md`) output. Supports:

| Action | Description |
|--------|-------------|
| `add` | Create a step with name, parameters, priority, category |
| `complete` | Mark a step as success/failed with result |
| `update` | Modify step fields |
| `list` | List steps, optionally filtered by status |
| `clear` | Remove all steps |
| `show` | Display a single step's details |
| `export` | Write current state to todo.md |

Steps have priority levels (high/medium/low), status tracking
(pending/success/failed), and support insertion after a specific step ID.

**`DoneTool`** — Signals task completion with a structured result and
reasoning string. Used as the termination signal for the agent loop.

### Priority Event Queue

Agent-core's `EventQueue` (via `MessageQueueInMemory`) provides pub/sub
messaging with per-session topic routing. In the deep_agent context, this
enables reactive event handling — but it lacks explicit priority ordering.

The prioritized steering concept (ABORT > STEER > FOLLOWUP) is implemented
in Orbiter's `TaskLoopQueue` rather than in agent-core's deep_agent
directory. Agent-core's deep_agent relies on the controller module's
`EventQueue` for cross-component messaging.

---

## 2. Orbiter Equivalent

Orbiter distributes deep_agent functionality across two packages:
`orbiter-core` (base tool system) and `orbiter-sandbox` (file, shell, and
code execution tools). Task management lives in `orbiter.task_controller`.

### Tool Base System (`orbiter.tool`)

- **`Tool`** — Abstract base class: `name`, `description`, `parameters`,
  async `execute()`
- **`FunctionTool`** — Wraps sync/async Python functions with automatic
  JSON Schema generation from type hints
- **`@tool` decorator** — Converts a plain function into a `FunctionTool`

### File Tools (`orbiter.sandbox.tools.FilesystemTool`)

`FilesystemTool` combines read/write/list into a single tool with an
`action` parameter. Path validation restricts access to configured
`allowed_directories`. Simpler than agent-core's split reader/editor —
no line-range reads or multi-edit batching.

```python
from orbiter.sandbox.tools import filesystem_tool

fs = filesystem_tool(allowed_directories=["/workspace"])
result = await fs.execute(action="read", path="/workspace/data.csv")
```

### Shell Tool (`orbiter.sandbox.tools.ShellTool`)

Two variants:

- **`ShellTool`** — Allowlist-based command execution. Uses
  `create_subprocess_exec` (no shell interpretation). Output truncated at
  10,000 characters. Factory: `shell_tool(allowed_commands=None, timeout=30.0)`

- **`TerminalTool`** — Full shell access (like agent-core's `BashTool`).
  Blocks dangerous commands via a blacklist (`rm -rf`, `dd`, `shutdown`,
  etc.). Factory: `terminal_tool(blocked_commands=None, timeout=30.0)`

### Code Execution (`orbiter.sandbox.tools.CodeTool`)

Sandboxed Python execution with a blocked-names list (`__import__`, `open`,
`os`, `sys`, `subprocess`, `shutil` by default). Can optionally delegate to
a container-based `Sandbox` for stronger isolation. Factory:
`code_tool(sandbox=None, blocked_names=None, timeout=10.0)`

### Todo / Task Management (`orbiter.task_controller`)

Orbiter has no direct `TodoTool` equivalent. Instead, the
`TaskLoopQueue` from `orbiter.task_controller` provides the priority
event queue for mid-execution steering:

| Priority | Type | Purpose |
|----------|------|---------|
| 0 (highest) | `ABORT` | Stop the agent immediately |
| 1 | `STEER` | Redirect the agent mid-task |
| 2 (lowest) | `FOLLOWUP` | Queue additional work |

Queue-bound tools (`steer_agent_tool`, `abort_agent_tool`) let an LLM
push events into the queue. `TaskManager` provides CRUD operations for
higher-level task tracking (see the
[Task Management porting guide](./task-management.md)).

---

## 3. Side-by-Side Examples

### Registering Built-in Tools on an Agent

```python
# Agent-core (deep_agent)
from openjiuwen.deep_agent.tools import (
    FileReaderTool, FileEditorTool, BashTool,
    PythonInterpreterTool, TodoTool, DoneTool,
)

tools = [
    FileReaderTool(),
    FileEditorTool(),
    BashTool(timeout=60),
    PythonInterpreterTool(),
    TodoTool(base_dir="/workspace"),
    DoneTool(),
]
agent = SuperReActAgent(tools=tools, model=my_model)
result = await agent.run("Analyze the CSV file in /workspace")

# Orbiter
from orbiter.agent import Agent
from orbiter.sandbox.tools import (
    filesystem_tool, shell_tool, code_tool,
)

agent = Agent(
    model="openai:gpt-4o",
    tools=[
        filesystem_tool(allowed_directories=["/workspace"]),
        shell_tool(allowed_commands=["ls", "cat", "grep", "find", "wc"]),
        code_tool(timeout=30.0),
    ],
)
result = await agent.run("Analyze the CSV file in /workspace")
```

### Running a Shell Command

```python
# Agent-core (deep_agent) — full shell, no restrictions
from openjiuwen.deep_agent.tools import BashTool

bash = BashTool(timeout=30)
output = await bash.execute(command="find /workspace -name '*.py' | wc -l")
# Returns: "stdout: 42\nstderr: \nexit_code: 0"

# Orbiter — allowlisted commands (ShellTool)
from orbiter.sandbox.tools import shell_tool

shell = shell_tool(allowed_commands=["find", "wc"], timeout=30.0)
output = await shell.execute(command="find /workspace -name '*.py'")
# Only allowlisted commands execute; others raise an error

# Orbiter — full shell with blacklist (TerminalTool)
from orbiter.sandbox.tools import terminal_tool

terminal = terminal_tool(timeout=30.0)
output = await terminal.execute(
    command="find /workspace -name '*.py' | wc -l",
)
# Full shell access; only destructive commands are blocked
```

### File Read with Line Range

```python
# Agent-core — line-range support
from openjiuwen.deep_agent.tools import FileReaderTool

reader = FileReaderTool()
output = await reader.execute(
    file_path="/workspace/main.py", start_line=10, end_line=25,
)
# Returns lines 10-25 with line numbers

# Orbiter — reads entire file (no line-range API)
from orbiter.sandbox.tools import filesystem_tool

fs = filesystem_tool(allowed_directories=["/workspace"])
output = await fs.execute(action="read", path="/workspace/main.py")
# Returns full file content; line slicing is caller's responsibility
```

---

## 4. Migration Table

| Agent-Core Path | Orbiter Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.deep_agent.tools.FileReaderTool` | `orbiter.sandbox.tools.filesystem_tool` | `FilesystemTool` with `action="read"` — no line-range support |
| `openjiuwen.deep_agent.tools.FileEditorTool` | `orbiter.sandbox.tools.filesystem_tool` | `FilesystemTool` with `action="write"` — no multi-edit batching |
| `openjiuwen.deep_agent.tools.FileAnalyzerTool` | *(no direct equivalent)* | Combine `FilesystemTool` read + custom LLM analysis |
| `openjiuwen.deep_agent.tools.BashTool` | `orbiter.sandbox.tools.terminal_tool` | `TerminalTool` — full shell with destructive-command blacklist |
| `openjiuwen.deep_agent.tools.BashTool` (restricted) | `orbiter.sandbox.tools.shell_tool` | `ShellTool` — allowlist-based, no shell interpretation |
| `openjiuwen.deep_agent.tools.PythonInterpreterTool` | `orbiter.sandbox.tools.code_tool` | `CodeTool` — blocked-names list; optional container sandbox |
| `openjiuwen.deep_agent.tools.TodoTool` | *(no direct equivalent)* | Use `orbiter.task_controller.TaskManager` for CRUD task tracking |
| `openjiuwen.deep_agent.tools.DoneTool` | *(no direct equivalent)* | Agent loop termination is handled by `Agent.run()` return |
| `openjiuwen.deep_agent.tools.WebSearchTool` | `orbiter.perplexica` | Perplexica pipeline provides web search + summarization |
| `openjiuwen.deep_agent.tools.DeepResearcherTool` | *(no direct equivalent)* | Multi-round research orchestration not in Orbiter core |
| `openjiuwen.deep_agent.tools.BrowserTool` | *(no direct equivalent)* | Browser automation not in Orbiter core |
| `openjiuwen.deep_agent.tools.MdifyTool` | *(no direct equivalent)* | File format conversion not in Orbiter core |
| `openjiuwen.deep_agent.tools.ToolGeneratorTool` | *(no direct equivalent)* | Dynamic tool generation not in Orbiter core |
| `openjiuwen.deep_agent.tools.ReformulatorTool` | *(no direct equivalent)* | Answer formatting not in Orbiter core |
| `openjiuwen.core.controller.EventQueue` | `orbiter.task_controller.TaskEventBus` | Typed pub/sub with `TaskEventType` enum |
| *(priority steering)* | `orbiter.task_controller.TaskLoopQueue` | Thread-safe priority queue: ABORT (0) > STEER (1) > FOLLOWUP (2) |
| *(priority events)* | `orbiter.task_controller.TaskLoopEvent` | Priority-ordered event with `type`, `content`, `metadata` |
| *(steering tools)* | `orbiter.task_controller.steer_agent_tool` | Queue-bound tool to redirect agent mid-task |
| *(abort tools)* | `orbiter.task_controller.abort_agent_tool` | Queue-bound tool to stop agent execution |
| `orbiter.tool.Tool` | `orbiter.tool.Tool` | Abstract base class for all tools |
| `orbiter.tool.FunctionTool` | `orbiter.tool.FunctionTool` | Wraps Python functions with auto JSON Schema |
| `orbiter.tool.tool` | `orbiter.tool.tool` | `@tool` decorator for quick tool creation |
