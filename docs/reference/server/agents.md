# exo_server.agents

Agent management and workspace routes. Provides endpoints for listing registered agents, inspecting agent details, and accessing workspace artifacts.

```python
from exo_server.agents import AgentInfo, WorkspaceFile, WorkspaceFileContent, agent_router
```

---

## AgentInfo

```python
class AgentInfo(BaseModel)
```

Summary information about a registered agent.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Agent name |
| `model` | `str` | `""` | Model string |
| `is_default` | `bool` | `False` | Whether this is the default agent |
| `tools` | `list[str]` | `[]` | Names of registered tools |
| `handoffs` | `list[str]` | `[]` | Names of handoff targets |
| `max_steps` | `int` | `0` | Maximum LLM call steps |
| `temperature` | `float` | `0.0` | Temperature setting |
| `max_tokens` | `int \| None` | `None` | Maximum token limit |

---

## WorkspaceFile

```python
class WorkspaceFile(BaseModel)
```

Metadata about a file/artifact in an agent's workspace.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | File name |
| `artifact_type` | `str` | `"text"` | Artifact type |
| `version_count` | `int` | `1` | Number of versions |

---

## WorkspaceFileContent

```python
class WorkspaceFileContent(BaseModel)
```

Full content of a workspace file.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | File name |
| `content` | `str` | *(required)* | File content |
| `artifact_type` | `str` | `"text"` | Artifact type |
| `version_count` | `int` | `1` | Number of versions |

---

## agent_router

```python
agent_router = APIRouter(prefix="/agents", tags=["agents"])
```

FastAPI router for agent management endpoints.

### Endpoints

#### GET /agents

List all registered agents.

**Response:** `list[AgentInfo]`

```bash
curl http://localhost:8000/agents
```

```json
[
  {
    "name": "helper",
    "model": "openai:gpt-4o",
    "is_default": true,
    "tools": ["web_search", "calculator"],
    "handoffs": [],
    "max_steps": 10,
    "temperature": 0.7,
    "max_tokens": null
  }
]
```

#### GET /agents/{agent_name}

Get details for a specific agent.

| Parameter | Type | Description |
|---|---|---|
| `agent_name` | `str` (path) | Agent name |

**Response:** `AgentInfo`

**Errors:** 404 if agent not found.

```bash
curl http://localhost:8000/agents/helper
```

#### GET /agents/{agent_name}/workspace

List files in an agent's workspace.

| Parameter | Type | Description |
|---|---|---|
| `agent_name` | `str` (path) | Agent name |

**Response:** `list[WorkspaceFile]`

Returns an empty list if the agent has no workspace. Returns 404 if the agent is not found.

```bash
curl http://localhost:8000/agents/helper/workspace
```

#### GET /agents/{agent_name}/workspace/{file_name}

Read the content of a specific workspace file.

| Parameter | Type | Description |
|---|---|---|
| `agent_name` | `str` (path) | Agent name |
| `file_name` | `str` (path) | File name (supports nested paths) |

**Response:** `WorkspaceFileContent`

**Errors:**

| Status | Condition |
|---|---|
| 404 | Agent not found, agent has no workspace, or file not found |

```bash
curl http://localhost:8000/agents/helper/workspace/notes.txt
```
