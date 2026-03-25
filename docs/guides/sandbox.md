# Sandbox

The `exo-sandbox` package provides isolated execution environments for agents. Sandboxes wrap code execution, filesystem access, and terminal operations behind a safe abstraction with status management, resource cleanup, and security controls.

## Basic Usage

```python
from exo.sandbox import LocalSandbox, FilesystemTool, TerminalTool

# Create a local sandbox
sandbox = LocalSandbox(work_dir="/tmp/sandbox_work")
await sandbox.start()

# Create safe tools for the sandbox
fs_tool = FilesystemTool(allowed_directories=["/tmp/sandbox_work"])
terminal = TerminalTool(timeout=30)

# Use tools
content = await fs_tool.execute(action="read", path="/tmp/sandbox_work/data.txt")
output = await terminal.execute(command="ls -la /tmp/sandbox_work")

# Clean up
await sandbox.stop()
```

## Sandbox Lifecycle

Every sandbox follows a status-driven lifecycle:

```
CREATED -> RUNNING -> IDLE -> RUNNING -> ... -> CLOSED
                  \-> ERROR -> CLOSED
```

| Status | Description |
|--------|-------------|
| `CREATED` | Initial state after construction |
| `RUNNING` | Actively executing code |
| `IDLE` | Started but not currently executing |
| `ERROR` | An error occurred |
| `CLOSED` | Stopped and cleaned up |

```python
from exo.sandbox import SandboxStatus

sandbox = LocalSandbox()
print(sandbox.status)  # SandboxStatus.CREATED

await sandbox.start()
print(sandbox.status)  # SandboxStatus.IDLE

await sandbox.stop()
print(sandbox.status)  # SandboxStatus.CLOSED
```

## LocalSandbox

Runs code in a local subprocess environment:

```python
from exo.sandbox import LocalSandbox

sandbox = LocalSandbox(
    work_dir="/tmp/sandbox",  # working directory
    env={"API_KEY": "sk-..."},  # environment variables
)

async with sandbox:  # auto start/stop
    # sandbox is running
    pass
# sandbox is now closed
```

## KubernetesSandbox

Runs code in an isolated Kubernetes pod:

```python
from exo.sandbox import KubernetesSandbox

sandbox = KubernetesSandbox(
    namespace="exo-sandboxes",
    image="python:3.12-slim",
    cpu="500m",
    memory="512Mi",
)

await sandbox.start()
# Creates pod + service in the cluster
# Waits for pod to be ready

await sandbox.stop()
# Cleans up pod + service
```

The Kubernetes sandbox:

- Creates a pod manifest with the specified image and resource limits.
- Creates a ClusterIP service for network access.
- Polls the pod status until it reaches `Running` state.
- Cleans up all resources on `stop()` or `cleanup()`.

## SandboxBuilder

The `SandboxBuilder` provides a fluent API for constructing sandboxes:

```python
from exo.sandbox import SandboxBuilder

sandbox = (
    SandboxBuilder()
    .with_type("local")
    .with_work_dir("/tmp/project")
    .with_env({"PYTHONPATH": "/tmp/project/src"})
    .with_tools([FilesystemTool(), TerminalTool()])
    .build()
)
```

The builder supports lazy evaluation -- it defers sandbox creation until `build()` is called or until an attribute is accessed:

```python
builder = (
    SandboxBuilder()
    .with_type("kubernetes")
    .with_image("python:3.12")
    .with_cpu("1000m")
    .with_memory("1Gi")
)

# Lazy: sandbox isn't created until build() or attribute access
sandbox = builder.build()
```

## FilesystemTool

Provides safe read/write/list operations within allowed directories:

```python
from exo.sandbox import FilesystemTool

fs = FilesystemTool(
    allowed_directories=["/tmp/sandbox", "/tmp/shared"],
)

# Read a file
content = await fs.execute(action="read", path="/tmp/sandbox/data.txt")

# Write a file
await fs.execute(action="write", path="/tmp/sandbox/output.txt", content="Hello")

# List directory contents
listing = await fs.execute(action="list", path="/tmp/sandbox")
```

Security: any path outside `allowed_directories` is rejected.

## TerminalTool

Executes shell commands with timeout and command blacklisting:

```python
from exo.sandbox import TerminalTool

terminal = TerminalTool(
    timeout=30,  # seconds
    blacklist=["rm -rf /", "sudo", "curl"],  # blocked commands
)

# Run a command
output = await terminal.execute(command="python --version")

# Command with timeout
output = await terminal.execute(command="python long_script.py")
# Raises error after 30 seconds
```

Blocked commands are checked by substring match against the blacklist.

## Advanced Patterns

### Agent with Sandbox Tools

Give an agent filesystem and terminal access within a sandbox:

```python
from exo.agent import Agent
from exo.sandbox import LocalSandbox, FilesystemTool, TerminalTool

sandbox = LocalSandbox(work_dir="/tmp/project")
await sandbox.start()

agent = Agent(
    name="coder",
    model="openai:gpt-4o",
    instructions="You are a coding assistant. Use the filesystem and terminal tools.",
    tools=[
        FilesystemTool(allowed_directories=["/tmp/project"]),
        TerminalTool(timeout=60, blacklist=["rm -rf"]),
    ],
)
```

### Kubernetes Sandbox for Untrusted Code

Use Kubernetes sandboxes for running user-provided code safely:

```python
from exo.sandbox import KubernetesSandbox

async def run_user_code(code: str) -> str:
    sandbox = KubernetesSandbox(
        namespace="user-sandboxes",
        image="python:3.12-slim",
        cpu="500m",
        memory="256Mi",
    )

    try:
        await sandbox.start()
        # Write code to sandbox and execute
        terminal = TerminalTool(timeout=30)
        result = await terminal.execute(command=f"python -c '{code}'")
        return result
    finally:
        await sandbox.cleanup()
```

### Builder with Custom Configuration

```python
def create_sandbox_for_task(task_type: str) -> Any:
    builder = SandboxBuilder()

    if task_type == "data_analysis":
        return (
            builder
            .with_type("local")
            .with_work_dir("/tmp/analysis")
            .with_env({"MPLBACKEND": "Agg"})
            .build()
        )
    elif task_type == "code_execution":
        return (
            builder
            .with_type("kubernetes")
            .with_image("python:3.12")
            .with_cpu("1000m")
            .with_memory("2Gi")
            .build()
        )
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Sandbox` | `exo.sandbox` | ABC for execution environments |
| `SandboxStatus` | `exo.sandbox` | Enum: `CREATED`, `RUNNING`, `IDLE`, `ERROR`, `CLOSED` |
| `SandboxError` | `exo.sandbox` | Error raised during sandbox operations |
| `LocalSandbox` | `exo.sandbox` | Local subprocess sandbox |
| `KubernetesSandbox` | `exo.sandbox` | Kubernetes pod-based sandbox |
| `SandboxBuilder` | `exo.sandbox` | Fluent builder for sandbox construction |
| `FilesystemTool` | `exo.sandbox` | Safe file read/write/list within allowed directories |
| `TerminalTool` | `exo.sandbox` | Shell command execution with timeout and blacklist |
