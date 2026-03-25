# exo.a2a.client

A2A client -- HTTP client and RemoteAgent for calling remote A2A agents.

```python
from exo.a2a.client import (
    A2AClient,
    A2AClientError,
    ClientManager,
    RemoteAgent,
)
```

---

## A2AClientError

```python
class A2AClientError(ExoError)
```

Raised for A2A client-level errors. Inherits from `exo.types.ExoError`.

---

## A2AClient

```python
class A2AClient(
    agent_card: AgentCard | str,
    config: ClientConfig | None = None,
)
```

HTTP client for communicating with a remote A2A agent. Resolves agent cards from URLs or local files, sends tasks, and optionally streams responses. Uses `httpx.AsyncClient` internally.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `agent_card` | `AgentCard \| str` | *(required)* | An `AgentCard` instance, a URL string pointing to `/.well-known/agent-card`, or a local file path |
| `config` | `ClientConfig \| None` | `None` | Client configuration (timeout, streaming prefs, etc.). Defaults to `ClientConfig()` |

**Raises:** `A2AClientError` -- If `agent_card` is an empty string or an unsupported type.

### Methods

#### resolve_agent_card

```python
async def resolve_agent_card(self) -> AgentCard
```

Resolve and cache the agent card. Fetches from URL or loads from file on first call; returns cached card on subsequent calls.

**Returns:** The resolved `AgentCard`.

**Raises:** `A2AClientError` -- If resolution fails.

#### send_task

```python
async def send_task(
    self,
    text: str,
    *,
    task_id: str | None = None,
) -> dict[str, Any]
```

Send a task to the remote agent and return the response.

| Name | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | The input text for the task |
| `task_id` | `str \| None` | `None` | Optional task identifier. Auto-generated if omitted |

**Returns:** Response dict from the server.

**Raises:** `A2AClientError` -- If the request fails.

#### send_task_streaming

```python
async def send_task_streaming(
    self,
    text: str,
    *,
    task_id: str | None = None,
) -> list[dict[str, Any]]
```

Send a task with streaming and collect all events.

| Name | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | The input text for the task |
| `task_id` | `str \| None` | `None` | Optional task identifier |

**Returns:** List of parsed NDJSON event dicts.

**Raises:** `A2AClientError` -- If the request fails or streaming is not supported by the remote agent.

#### close

```python
async def close(self) -> None
```

Close the underlying HTTP client.

### Example

```python
from exo.a2a import A2AClient

# From URL
client = A2AClient("http://localhost:8000/.well-known/agent-card")

# From AgentCard
client = A2AClient(AgentCard(name="helper", url="http://localhost:8000"))

# Send a task
response = await client.send_task("What is Python?")
print(response)

await client.close()
```

---

## ClientManager

```python
class ClientManager(
    agent_card: AgentCard | str,
    config: ClientConfig | None = None,
)
```

Thread-safe manager that provides per-thread `A2AClient` instances. Each thread gets its own client via `get_client()`. Clients are cleaned up when `shutdown()` is called.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `agent_card` | `AgentCard \| str` | *(required)* | Agent card or source string to pass to each client |
| `config` | `ClientConfig \| None` | `None` | Client configuration shared across all threads |

### Methods

#### get_client

```python
def get_client(self) -> A2AClient
```

Return the A2A client for the current thread, creating one if needed.

#### shutdown

```python
async def shutdown(self) -> None
```

Close all client instances across all threads.

### Example

```python
manager = ClientManager("http://localhost:8000/.well-known/agent-card")

# In any thread:
client = manager.get_client()
response = await client.send_task("Hello")

# Cleanup
await manager.shutdown()
```

---

## RemoteAgent

```python
class RemoteAgent(
    *,
    name: str,
    agent_card: AgentCard | str,
    config: ClientConfig | None = None,
)
```

Agent-compatible wrapper for calling a remote A2A agent. Provides the same `run(input, ...)` interface as `exo.agent.Agent` so it can be used as a handoff target or standalone caller.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Local name for this remote agent |
| `agent_card` | `AgentCard \| str` | *(required)* | `AgentCard`, URL, or file path for the remote agent |
| `config` | `ClientConfig \| None` | `None` | Client configuration |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Local name for this remote agent |

### Methods

#### run

```python
async def run(self, input: str, **kwargs: Any) -> AgentOutput
```

Send input to the remote agent and return the parsed output.

| Name | Type | Default | Description |
|---|---|---|---|
| `input` | `str` | *(required)* | The user query text |
| `**kwargs` | `Any` | | Ignored (compatibility with Agent.run signature) |

**Returns:** `AgentOutput` with the remote agent's response text.

#### describe

```python
async def describe(self) -> dict[str, Any]
```

Return a description using the resolved agent card. Keys: `name`, `remote_name`, `description`, `url`, `capabilities`.

#### close

```python
async def close(self) -> None
```

Close the underlying client.

### Example

```python
from exo.a2a import RemoteAgent

agent = RemoteAgent(
    name="remote-helper",
    agent_card="http://localhost:8000/.well-known/agent-card",
)

output = await agent.run("Explain quantum computing")
print(output.text)

await agent.close()
```
