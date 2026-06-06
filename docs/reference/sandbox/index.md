# exo.sandbox

Isolated execution environments for safe agent operation.

## Installation

```bash
pip install exo-sandbox

# For Kubernetes support:
pip install exo-sandbox[kubernetes]

# For E2B cloud sandbox support:
pip install exo-sandbox[e2b]
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
| `E2BSandbox` | `exo.sandbox.e2b` | Sandbox backed by E2B cloud VMs |
| `FilesystemTool` | `exo.sandbox.tools` | Sandboxed filesystem tool with directory restrictions |
| `TerminalTool` | `exo.sandbox.tools` | Sandboxed terminal tool with command filtering |
| `ShellTool` | `exo.sandbox.tools` | Allowlist-based shell tool |
| `CodeTool` | `exo.sandbox.tools` | Python code execution tool |

## Submodules

- [exo.sandbox.base](base.md) -- Sandbox ABC, LocalSandbox, SandboxStatus, SandboxError
- [exo.sandbox.builder](builder.md) -- SandboxBuilder with fluent API
- [exo.sandbox.kubernetes](kubernetes.md) -- KubernetesSandbox for remote execution
- [exo.sandbox.e2b](e2b.md) -- E2BSandbox for E2B cloud execution
- [exo.sandbox.tools](tools.md) -- FilesystemTool, TerminalTool, ShellTool, CodeTool
