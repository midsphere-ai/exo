# exo.a2a.types

A2A protocol types -- agent cards, configs, and task events. All model classes use Pydantic `BaseModel` with `frozen=True`.

```python
from exo.a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    ClientConfig,
    ServingConfig,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TransportMode,
)
```

---

## TransportMode

```python
class TransportMode(StrEnum)
```

Supported A2A transport protocols.

| Value | Description |
|---|---|
| `JSONRPC = "jsonrpc"` | JSON-RPC over HTTP |
| `GRPC = "grpc"` | gRPC transport |
| `WEBSOCKET = "websocket"` | WebSocket transport |

---

## TaskState

```python
class TaskState(StrEnum)
```

Lifecycle states for a remote A2A task.

| Value | Description |
|---|---|
| `SUBMITTED = "submitted"` | Task has been submitted |
| `WORKING = "working"` | Agent is processing the task |
| `COMPLETED = "completed"` | Task completed successfully |
| `FAILED = "failed"` | Task execution failed |
| `CANCELED = "canceled"` | Task was canceled |

---

## AgentSkill

```python
class AgentSkill(BaseModel, frozen=True)
```

A single capability advertised by an agent.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | *(required)* | Unique skill identifier |
| `name` | `str` | *(required)* | Human-readable name |
| `description` | `str` | `""` | What the skill does |
| `tags` | `tuple[str, ...]` | `()` | Classification tags |

---

## AgentCapabilities

```python
class AgentCapabilities(BaseModel, frozen=True)
```

Runtime capabilities of an A2A agent.

| Field | Type | Default | Description |
|---|---|---|---|
| `streaming` | `bool` | `False` | Supports streaming responses |
| `push_notifications` | `bool` | `False` | Supports push notifications |
| `state_transition_history` | `bool` | `False` | Tracks state transitions |

---

## AgentCard

```python
class AgentCard(BaseModel, frozen=True)
```

Complete metadata descriptor for a remote A2A agent. Published at `/.well-known/agent-card` for discovery.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Agent identifier |
| `description` | `str` | `""` | Agent purpose |
| `version` | `str` | `"0.0.1"` | Agent version |
| `url` | `str` | `""` | Agent endpoint URL |
| `capabilities` | `AgentCapabilities` | `AgentCapabilities()` | Runtime capabilities |
| `skills` | `tuple[AgentSkill, ...]` | `()` | Advertised skills |
| `default_input_modes` | `tuple[str, ...]` | `("text",)` | Accepted input formats |
| `default_output_modes` | `tuple[str, ...]` | `("text",)` | Produced output formats |
| `supported_transports` | `tuple[TransportMode, ...]` | `(TransportMode.JSONRPC,)` | Transport protocols |

### Example

```python
card = AgentCard(
    name="research-agent",
    description="Performs web research",
    url="http://localhost:8000",
    capabilities=AgentCapabilities(streaming=True),
    skills=(
        AgentSkill(id="search", name="Web Search", tags=("research",)),
    ),
)
```

---

## ServingConfig

```python
class ServingConfig(BaseModel, frozen=True)
```

Server-side configuration for publishing an agent via A2A.

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"localhost"` | Bind host |
| `port` | `int` | `0` | Bind port (0 = auto) |
| `endpoint` | `str` | `"/"` | Base URL path |
| `streaming` | `bool` | `False` | Enable streaming |
| `version` | `str` | `"0.0.1"` | Advertised version |
| `skills` | `tuple[AgentSkill, ...]` | `()` | Skills to advertise |
| `input_modes` | `tuple[str, ...]` | `("text",)` | Accepted input formats |
| `output_modes` | `tuple[str, ...]` | `("text",)` | Produced output formats |
| `transports` | `tuple[TransportMode, ...]` | `(TransportMode.JSONRPC,)` | Enabled transports |
| `extra` | `dict[str, Any]` | `{}` | Extension point for custom config |

---

## ClientConfig

```python
class ClientConfig(BaseModel, frozen=True)
```

Client-side configuration for connecting to a remote A2A agent.

| Field | Type | Default | Description |
|---|---|---|---|
| `streaming` | `bool` | `False` | Request streaming |
| `timeout` | `float` | `600.0` | Request timeout in seconds (must be > 0) |
| `transports` | `tuple[TransportMode, ...]` | `(TransportMode.JSONRPC,)` | Preferred transports |
| `accepted_output_modes` | `tuple[str, ...]` | `()` | Accepted output formats (empty = any) |
| `extra` | `dict[str, Any]` | `{}` | Extension point for custom config |

---

## TaskStatus

```python
class TaskStatus(BaseModel, frozen=True)
```

Current status of a remote A2A task.

| Field | Type | Default | Description |
|---|---|---|---|
| `state` | `TaskState` | *(required)* | Task lifecycle state |
| `reason` | `str` | `""` | Reason / error message |

---

## TaskStatusUpdateEvent

```python
class TaskStatusUpdateEvent(BaseModel, frozen=True)
```

Emitted when a remote task changes state.

| Field | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | *(required)* | Task being updated |
| `status` | `TaskStatus` | *(required)* | New status |

---

## TaskArtifactUpdateEvent

```python
class TaskArtifactUpdateEvent(BaseModel, frozen=True)
```

Emitted when a remote task produces output.

| Field | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | *(required)* | Task being updated |
| `text` | `str` | `""` | Artifact text content |
| `last_chunk` | `bool` | `False` | Whether this is the final chunk |
