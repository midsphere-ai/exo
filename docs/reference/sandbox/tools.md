# exo.sandbox.tools

Built-in sandbox tools: filesystem access, terminal execution, shell commands, and code execution. All tools accept an optional `sandbox` parameter — when provided, operations execute remotely via the sandbox's `run_tool()` instead of locally.

```python
from exo.sandbox.tools import FilesystemTool, TerminalTool, ShellTool, CodeTool
```

---

## FilesystemTool

```python
class FilesystemTool(Tool)(
    allowed_directories: list[str] | None = None,
    *,
    sandbox: Any | None = None,
)
```

Sandboxed filesystem tool with allowed-directory restrictions. Only paths that resolve within one of `allowed_directories` are permitted. This prevents agents from reading or writing files outside the designated workspace.

When a `sandbox` is provided, all operations are delegated to the sandbox via `run_tool()` instead of executing locally.

Inherits from `exo.tool.Tool`.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `allowed_directories` | `list[str] \| None` | `None` | List of allowed directory paths. If empty/None, all paths are allowed |
| `sandbox` | `Any \| None` | `None` | Sandbox instance for remote execution. When set, operations delegate to `sandbox.run_tool()` |

### Class attributes

| Attribute | Value | Description |
|---|---|---|
| `name` | `"filesystem"` | Tool name |
| `description` | `"Read, write, or list files within the sandbox workspace."` | Tool description |

### Parameters schema

```json
{
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["read", "write", "list"],
            "description": "Filesystem action to perform."
        },
        "path": {
            "type": "string",
            "description": "Absolute or relative file/directory path."
        },
        "content": {
            "type": "string",
            "description": "Content to write (required for 'write' action)."
        }
    },
    "required": ["action", "path"]
}
```

### Methods

#### execute

```python
async def execute(self, **kwargs: Any) -> str | dict[str, Any]
```

Execute a filesystem action.

| Argument | Type | Required | Description |
|---|---|---|---|
| `action` | `str` | Yes | One of `"read"`, `"write"`, `"list"` |
| `path` | `str` | Yes | File or directory path |
| `content` | `str` | For `write` | Content to write |

**Returns:**
- `read`: File contents as a string
- `write`: Confirmation string (e.g. `"Wrote 42 chars to /tmp/file.txt"`)
- `list`: Dict with `directory` and `entries` (list of `{name, type}` dicts)

**Raises:** `ToolError` -- If the path is outside allowed directories, the action is unknown, or the filesystem operation fails.

### Example

```python
from exo.sandbox.tools import FilesystemTool

fs = FilesystemTool(allowed_directories=["/tmp/workspace"])

# Read a file
content = await fs.execute(action="read", path="/tmp/workspace/data.txt")

# Write a file
result = await fs.execute(
    action="write",
    path="/tmp/workspace/output.txt",
    content="Hello, world!",
)

# List a directory
listing = await fs.execute(action="list", path="/tmp/workspace")
# {"directory": "/tmp/workspace", "entries": [{"name": "data.txt", "type": "file"}]}

# Path outside allowed directories raises ToolError
await fs.execute(action="read", path="/etc/passwd")  # ToolError!
```

---

## TerminalTool

```python
class TerminalTool(Tool)(
    *,
    blacklist: frozenset[str] | None = None,
    timeout: float = 30.0,
    sandbox: Any | None = None,
)
```

Sandboxed terminal tool with command filtering and timeout. Dangerous commands (`rm`, `shutdown`, etc.) are blocked by default. Custom blacklists can be provided. All commands run with a configurable timeout to prevent runaway processes.

When a `sandbox` is provided, commands are executed remotely via `sandbox.run_tool()` instead of on the local machine.

Inherits from `exo.tool.Tool`.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `blacklist` | `frozenset[str] \| None` | `None` | Set of blocked command names. Defaults to built-in dangerous commands |
| `timeout` | `float` | `30.0` | Command execution timeout in seconds |
| `sandbox` | `Any \| None` | `None` | Sandbox instance for remote execution |

### Default blacklist

The following commands are blocked by default:

`rm`, `rmdir`, `mkfs`, `dd`, `shutdown`, `reboot`, `halt`, `poweroff`, `kill`, `killall`, `pkill`, `format`, `del`, `erase`, `rd`

### Class attributes

| Attribute | Value | Description |
|---|---|---|
| `name` | `"terminal"` | Tool name |
| `description` | `"Execute a shell command in the sandbox."` | Tool description |

### Properties

| Property | Type | Description |
|---|---|---|
| `platform` | `str` | Current platform identifier (`sys.platform`) |

### Parameters schema

```json
{
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute."
        }
    },
    "required": ["command"]
}
```

### Methods

#### execute

```python
async def execute(self, **kwargs: Any) -> str | dict[str, Any]
```

Execute a shell command.

| Argument | Type | Required | Description |
|---|---|---|---|
| `command` | `str` | Yes | Shell command to execute |

**Returns:** Dict with keys:
- `exit_code` (`int`): Process exit code
- `stdout` (`str`): Standard output
- `stderr` (`str`): Standard error
- `platform` (`str`): Platform identifier

**Raises:** `ToolError` -- If the command is empty, blacklisted, times out, or fails to execute.

### Example

```python
from exo.sandbox.tools import TerminalTool

term = TerminalTool(timeout=10.0)

result = await term.execute(command="echo 'Hello, world!'")
# {"exit_code": 0, "stdout": "Hello, world!\n", "stderr": "", "platform": "linux"}

result = await term.execute(command="python --version")
# {"exit_code": 0, "stdout": "Python 3.12.0\n", ...}

# Blocked commands raise ToolError
await term.execute(command="rm -rf /")  # ToolError: Command 'rm' is blocked

# Custom blacklist
strict = TerminalTool(blacklist=frozenset({"curl", "wget", "ssh"}))
```
