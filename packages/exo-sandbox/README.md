# exo-sandbox

Sandboxed execution environments for the [Exo](../../README.md) multi-agent framework. Run agent tools in isolated local or Kubernetes environments.

## Installation

```bash
pip install exo-sandbox

# With Kubernetes support
pip install exo-sandbox[kubernetes]
```

Requires Python 3.11+ and `exo-core`.

## What's Included

- **LocalSandbox** -- process-isolated execution on the local machine with filesystem and terminal tools.
- **KubernetesSandbox** -- pod-based sandboxing for production workloads.
- **Filesystem tools** -- read, write, list, and search files within the sandbox.
- **Terminal tools** -- execute commands within the sandbox environment.
- **Sandbox builder** -- declarative sandbox configuration and agent integration.

## Quick Example

```python
from exo.sandbox import LocalSandbox

sandbox = LocalSandbox(working_dir="/tmp/workspace")
tools = sandbox.get_tools()

# Use sandbox tools with an agent
agent = Agent(
    name="coder",
    model="openai:gpt-4o",
    tools=tools,
)
```

## Documentation

- [Sandbox Guide](../../docs/guides/sandbox.md)
- [API Reference](../../docs/reference/sandbox/)
