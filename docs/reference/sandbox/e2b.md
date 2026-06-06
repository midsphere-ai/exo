# exo.sandbox.e2b

E2B cloud sandbox for remote agent execution.

```python
from exo.sandbox.e2b import E2BSandbox
```

---

## E2BSandbox

```python
class E2BSandbox(Sandbox)(
    *,
    sandbox_id: str | None = None,
    workspace: list[str] | None = None,
    mcp_config: dict[str, Any] | None = None,
    agents: dict[str, Any] | None = None,
    timeout: float = 300.0,
    api_key: str | None = None,
    template: str | None = None,
    existing_sandbox_id: str | None = None,
    metadata: dict[str, Any] | None = None,
)
```

Sandbox backed by [E2B](https://e2b.dev) cloud sandboxes. Requires the `e2b` extra (`pip install exo-sandbox[e2b]`).

Unlike `LocalSandbox` or `KubernetesSandbox`, `run_tool` dispatches to real E2B operations (shell commands, file read/write/list) rather than returning stub metadata.

Inherits from `exo.sandbox.base.Sandbox`.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `sandbox_id` | `str \| None` | `None` | Local tracking ID (auto-generated if omitted) |
| `api_key` | `str \| None` | `None` | E2B API key. Falls back to `E2B_API_KEY` env var |
| `template` | `str \| None` | `None` | E2B template ID. Falls back to `E2B_TEMPLATE_ID` env var |
| `existing_sandbox_id` | `str \| None` | `None` | Connect to a running sandbox instead of creating one |
| `metadata` | `dict \| None` | `None` | Metadata dict attached to the sandbox on creation |
| `timeout` | `float` | `300.0` | Sandbox lifetime in seconds |
| `workspace` | `list[str] \| None` | `None` | Allowed workspace directories |
| `mcp_config` | `dict \| None` | `None` | MCP server configuration |
| `agents` | `dict \| None` | `None` | Agent configurations |

### Properties

| Property | Type | Description |
|---|---|---|
| `api_key` | `str \| None` | Configured API key |
| `template` | `str \| None` | E2B template ID |
| `e2b_sandbox_id` | `str \| None` | Remote sandbox ID assigned by E2B (set after `start()`) |
| `existing_sandbox_id` | `str \| None` | ID of existing sandbox to connect to |
| `registered_tools` | `list[str]` | Names of all registered tool handlers |

### Methods

#### start

```python
async def start(self) -> None
```

Create or connect to an E2B sandbox. If `existing_sandbox_id` is set, connects via `Sandbox.connect()`. Otherwise creates a new sandbox via `Sandbox.create()`.

Sets status to `RUNNING` on success, `ERROR` on failure.

#### stop

```python
async def stop(self) -> None
```

Kill the E2B sandbox. Sets status to `IDLE`. Can be followed by a new `start()`.

#### cleanup

```python
async def cleanup(self) -> None
```

Kill the E2B sandbox and release all resources permanently. Sets status to `CLOSED`.

#### run_tool

```python
async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any
```

Execute a tool within the E2B sandbox. Dispatch order:

1. **Registered tools** -- added via `register_tool()`.
2. **Built-in tools:**

| tool_name | arguments | Returns |
|---|---|---|
| `"shell"` / `"command"` | `{"command": "..."}` | `{stdout, stderr, exit_code, ...}` |
| `"file_read"` | `{"path": "..."}` | `{content, path, ...}` |
| `"file_write"` | `{"path": "...", "content": "..."}` | `{path, bytes_written, ...}` |
| `"file_list"` | `{"path": "..."}` | `{path, entries, ...}` |

3. **Fallback** -- returns a metadata dict for unknown tool names.

**Raises:** `SandboxError` if the sandbox is not running or the E2B operation fails.

#### register_tool

```python
def register_tool(self, name: str, handler: ToolHandler) -> None
```

Register a custom tool handler. Handlers are async callables with signature:

```python
async def handler(sandbox: E2BSandbox, arguments: dict[str, Any]) -> Any: ...
```

Registered handlers take precedence over built-in dispatch in `run_tool`. **Raises:** `SandboxError` if a tool with the same name is already registered.

#### unregister_tool

```python
def unregister_tool(self, name: str) -> None
```

Remove a previously registered tool handler. **Raises:** `SandboxError` if no tool with the given name is registered.

#### Tool factory methods

| Method | Returns | Description |
|---|---|---|
| `filesystem_tool(allowed_directories=None)` | `FilesystemTool` | Pre-wired to this sandbox |
| `terminal_tool(*, blacklist=None, timeout=30.0)` | `TerminalTool` | Pre-wired to this sandbox |
| `shell_tool(*, allowed_commands=None, timeout=30.0)` | `ShellTool` | Pre-wired to this sandbox |
| `code_tool(*, blocked_names=None, timeout=10.0)` | `CodeTool` | Pre-wired to this sandbox |

#### describe

```python
def describe(self) -> dict[str, Any]
```

Returns sandbox metadata including `template`, `e2b_sandbox_id`, `existing_sandbox_id`, masked `api_key`, and `registered_tools`.

### Context manager

```python
async with E2BSandbox(api_key="...") as sandbox:
    result = await sandbox.run_tool("shell", {"command": "echo hello"})
# sandbox is automatically cleaned up
```

### Example

```python
from exo.sandbox import E2BSandbox

async with E2BSandbox(api_key="e2b_...", template="my-template") as sandbox:
    # Shell commands
    result = await sandbox.run_tool("shell", {"command": "python --version"})
    print(result["stdout"])

    # File operations
    await sandbox.run_tool("file_write", {
        "path": "/home/user/script.py",
        "content": "print('hello from E2B')",
    })
    result = await sandbox.run_tool("shell", {"command": "python /home/user/script.py"})

    # Register custom tools
    async def lint(sandbox, args):
        return await sandbox.run_tool("shell", {"command": f"ruff check {args['path']}"})

    sandbox.register_tool("lint", lint)
    result = await sandbox.run_tool("lint", {"path": "/home/user/script.py"})
```

---

## ToolHandler

```python
ToolHandler = Callable[[E2BSandbox, dict[str, Any]], Awaitable[Any]]
```

Type alias for registered tool handlers. A handler receives the `E2BSandbox` instance and the tool arguments dict, and returns any result.
