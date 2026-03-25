# exo.sandbox

Isolated execution environments for safe agent operation.

## Installation

```bash
pip install exo-sandbox

# For Kubernetes support:
pip install exo-sandbox[kubernetes]
```

## Module path

```python
import exo.sandbox
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `Sandbox` | `exo.sandbox.base` | Abstract sandbox providing isolated execution |
| `LocalSandbox` | `exo.sandbox.base` | Sandbox that executes on the local machine |
| `SandboxStatus` | `exo.sandbox.base` | Lifecycle states for a sandbox |
| `SandboxError` | `exo.sandbox.base` | Error raised for sandbox-level errors |
| `SandboxBuilder` | `exo.sandbox.builder` | Fluent builder for constructing sandbox instances |
| `KubernetesSandbox` | `exo.sandbox.kubernetes` | Sandbox that manages a Kubernetes pod |
| `FilesystemTool` | `exo.sandbox.tools` | Sandboxed filesystem tool with directory restrictions |
| `TerminalTool` | `exo.sandbox.tools` | Sandboxed terminal tool with command filtering |

## Submodules

- [exo.sandbox.base](base.md) -- Sandbox ABC, LocalSandbox, SandboxStatus, SandboxError
- [exo.sandbox.builder](builder.md) -- SandboxBuilder with fluent API
- [exo.sandbox.kubernetes](kubernetes.md) -- KubernetesSandbox for remote execution
- [exo.sandbox.tools](tools.md) -- FilesystemTool and TerminalTool
