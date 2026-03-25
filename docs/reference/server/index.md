# exo_server

Web UI and API server for running Exo agents via HTTP, WebSocket, and SSE.

```python
from exo_server import (
    # app — FastAPI application
    ChatRequest,
    ChatResponse,
    create_app,
    register_agent,
    # agents — agent management routes
    AgentInfo,
    WorkspaceFile,
    WorkspaceFileContent,
    # sessions — session management
    AppendMessageRequest,
    CreateSessionRequest,
    Session,
    SessionMessage,
    SessionSummary,
    UpdateSessionRequest,
)
```

**Install:** `pip install exo-server`

---

## Submodules

| Module | Description |
|---|---|
| [`exo_server.app`](app.md) | FastAPI app factory, `/chat` endpoint, agent registry |
| [`exo_server.agents`](agents.md) | Agent management and workspace routes |
| [`exo_server.sessions`](sessions.md) | Session CRUD and message history |
| [`exo_server.streaming`](streaming.md) | WebSocket and SSE streaming endpoints |

---

## Public API summary

| Export | Kind | Source |
|---|---|---|
| `ChatRequest` | Pydantic model | `app` |
| `ChatResponse` | Pydantic model | `app` |
| `create_app` | Function | `app` |
| `register_agent` | Function | `app` |
| `AgentInfo` | Pydantic model | `agents` |
| `WorkspaceFile` | Pydantic model | `agents` |
| `WorkspaceFileContent` | Pydantic model | `agents` |
| `AppendMessageRequest` | Pydantic model | `sessions` |
| `CreateSessionRequest` | Pydantic model | `sessions` |
| `Session` | Pydantic model | `sessions` |
| `SessionMessage` | Pydantic model | `sessions` |
| `SessionSummary` | Pydantic model | `sessions` |
| `UpdateSessionRequest` | Pydantic model | `sessions` |

---

## Quick start

```python
from exo_server import create_app, register_agent
from exo.agent import Agent

# Create agent
agent = Agent(name="helper", model="openai:gpt-4o", instructions="You are helpful.")

# Create and configure app
app = create_app()
register_agent(app, agent, default=True)

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```
