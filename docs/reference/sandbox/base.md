# exo.sandbox.base

Sandbox interface and local implementation for safe agent execution.

```python
from exo.sandbox.base import (
    LocalSandbox,
    Sandbox,
    SandboxError,
    SandboxStatus,
)
```

---

## SandboxError

```python
class SandboxError(Exception)
```

Raised for sandbox-level errors (invalid transitions, resource failures, etc.).

---

## SandboxStatus

```python
class SandboxStatus(StrEnum)
```

Lifecycle states for a sandbox.

| Value | Description |
|---|---|
| `INIT = "init"` | Created but not started |
| `RUNNING = "running"` | Active and accepting tool calls |
| `IDLE = "idle"` | Stopped but can be restarted |
| `ERROR = "error"` | Error state (can retry start) |
| `CLOSED = "closed"` | Permanently shut down |

### Valid transitions

| From | Allowed targets |
|---|---|
| `init` | `running`, `closed` |
| `running` | `idle`, `error`, `closed` |
| `idle` | `running`, `error`, `closed` |
| `error` | `running`, `closed` |
| `closed` | *(none -- terminal state)* |

---

## Sandbox

```python
class Sandbox(ABC)(
    *,
    sandbox_id: str | None = None,
    workspace: list[str] | None = None,
    mcp_config: dict[str, Any] | None = None,
    agents: dict[str, Any] | None = None,
    timeout: float = 30.0,
)
```

Abstract sandbox providing isolated execution for agents. Subclasses implement `start`, `stop`, and `cleanup` to manage the concrete environment (local process, Kubernetes pod, etc.).

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `sandbox_id` | `str \| None` | `None` | Unique identifier (auto-generated 12-char hex if omitted) |
| `workspace` | `list[str] \| None` | `None` | Allowed workspace directories |
| `mcp_config` | `dict[str, Any] \| None` | `None` | MCP server configuration |
| `agents` | `dict[str, Any] \| None` | `None` | Agent configurations |
| `timeout` | `float` | `30.0` | Execution timeout in seconds |

### Properties

| Property | Type | Description |
|---|---|---|
| `sandbox_id` | `str` | Unique sandbox identifier |
| `status` | `SandboxStatus` | Current lifecycle state |
| `workspace` | `list[str]` | Copy of allowed workspace directories |
| `mcp_config` | `dict[str, Any]` | Copy of MCP configuration |
| `agents` | `dict[str, Any]` | Copy of agent configurations |
| `timeout` | `float` | Execution timeout in seconds |

### Abstract methods

#### start

```python
async def start(self) -> None
```

Start the sandbox environment. Must be implemented by subclasses.

#### stop

```python
async def stop(self) -> None
```

Stop the sandbox (may be restarted later). Must be implemented by subclasses.

#### cleanup

```python
async def cleanup(self) -> None
```

Release all resources and close the sandbox permanently. Must be implemented by subclasses.

### Methods

#### describe

```python
def describe(self) -> dict[str, Any]
```

Return a dict describing the sandbox state.

**Returns:** Dict with keys `sandbox_id`, `status`, `workspace`, `timeout`.

---

## LocalSandbox

```python
class LocalSandbox(Sandbox)
```

Sandbox that executes on the local machine. Inherits all constructor parameters from `Sandbox`.

### Methods

#### run_tool

```python
async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any
```

Execute a tool within this sandbox.

| Name | Type | Default | Description |
|---|---|---|---|
| `tool_name` | `str` | *(required)* | Name of the tool to invoke |
| `arguments` | `dict[str, Any]` | *(required)* | Tool arguments |

**Returns:** Dict with `tool`, `arguments`, and `status` keys.

**Raises:** `SandboxError` -- If the sandbox is not in `RUNNING` status.

### Async context manager

```python
async with LocalSandbox(workspace=["/tmp/ws"]) as sb:
    result = await sb.run_tool("echo", {"text": "hello"})
# Automatically cleaned up on exit
```

### Example

```python
from exo.sandbox import LocalSandbox

sandbox = LocalSandbox(
    workspace=["/tmp/workspace"],
    timeout=60.0,
)

await sandbox.start()
print(sandbox.status)  # SandboxStatus.RUNNING

result = await sandbox.run_tool("my_tool", {"arg": "value"})

await sandbox.stop()
print(sandbox.status)  # SandboxStatus.IDLE

await sandbox.cleanup()
print(sandbox.status)  # SandboxStatus.CLOSED
```
