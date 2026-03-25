# exo_server.sessions

Session management API routes. Provides CRUD endpoints for chat sessions. Each session groups a conversation with a specific agent, storing messages exchanged during the interaction.

```python
from exo_server.sessions import (
    AppendMessageRequest,
    CreateSessionRequest,
    Session,
    SessionMessage,
    SessionSummary,
    UpdateSessionRequest,
    session_router,
)
```

---

## SessionMessage

```python
class SessionMessage(BaseModel)
```

A single message within a session.

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `str` | *(required)* | Message role (e.g. `"user"`, `"assistant"`) |
| `content` | `str` | *(required)* | Message text content |
| `timestamp` | `float` | `time.time()` | When the message was created |

---

## Session

```python
class Session(BaseModel)
```

A chat session grouping a conversation with an agent.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | *(auto: 16-char hex)* | Unique session identifier |
| `agent_name` | `str` | `""` | Name of the associated agent |
| `title` | `str` | `""` | Session title |
| `messages` | `list[SessionMessage]` | `[]` | Conversation messages |
| `created_at` | `float` | `time.time()` | Creation timestamp |
| `updated_at` | `float` | `time.time()` | Last update timestamp |

---

## CreateSessionRequest

```python
class CreateSessionRequest(BaseModel)
```

Request body for creating a new session.

| Field | Type | Default | Description |
|---|---|---|---|
| `agent_name` | `str` | `""` | Name of the agent for this session |
| `title` | `str` | `""` | Session title |

---

## UpdateSessionRequest

```python
class UpdateSessionRequest(BaseModel)
```

Request body for updating a session.

| Field | Type | Default | Description |
|---|---|---|---|
| `title` | `str \| None` | `None` | New title (unchanged if `None`) |
| `agent_name` | `str \| None` | `None` | New agent name (unchanged if `None`) |

---

## AppendMessageRequest

```python
class AppendMessageRequest(BaseModel)
```

Request body for appending a message to a session.

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `str` | *(required)* | Message role |
| `content` | `str` | *(required)* | Message content |

---

## SessionSummary

```python
class SessionSummary(BaseModel)
```

Lightweight session info for list responses.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | *(required)* | Session identifier |
| `agent_name` | `str` | *(required)* | Associated agent name |
| `title` | `str` | *(required)* | Session title |
| `message_count` | `int` | *(required)* | Number of messages |
| `created_at` | `float` | *(required)* | Creation timestamp |
| `updated_at` | `float` | *(required)* | Last update timestamp |

---

## session_router

```python
session_router = APIRouter(prefix="/sessions", tags=["sessions"])
```

FastAPI router for session management endpoints. Sessions are stored in-memory on the app state.

### Endpoints

#### POST /sessions

Create a new chat session.

**Request body:** `CreateSessionRequest`

**Response:** `Session` (status 201)

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "helper", "title": "Python help"}'
```

#### GET /sessions

List all sessions, newest first.

**Response:** `list[SessionSummary]`

```bash
curl http://localhost:8000/sessions
```

#### GET /sessions/{session_id}

Retrieve a single session by ID.

| Parameter | Type | Description |
|---|---|---|
| `session_id` | `str` (path) | Session identifier |

**Response:** `Session`

**Errors:** 404 if session not found.

#### PATCH /sessions/{session_id}

Update session metadata (title, agent_name). Only provided fields are updated.

| Parameter | Type | Description |
|---|---|---|
| `session_id` | `str` (path) | Session identifier |

**Request body:** `UpdateSessionRequest`

**Response:** `Session`

**Errors:** 404 if session not found.

```bash
curl -X PATCH http://localhost:8000/sessions/abc123 \
  -H "Content-Type: application/json" \
  -d '{"title": "Updated title"}'
```

#### DELETE /sessions/{session_id}

Delete a session.

| Parameter | Type | Description |
|---|---|---|
| `session_id` | `str` (path) | Session identifier |

**Response:** 204 No Content

**Errors:** 404 if session not found.

#### POST /sessions/{session_id}/messages

Append a message to a session's conversation history.

| Parameter | Type | Description |
|---|---|---|
| `session_id` | `str` (path) | Session identifier |

**Request body:** `AppendMessageRequest`

**Response:** `SessionMessage` (status 201)

**Errors:** 404 if session not found.

```bash
curl -X POST http://localhost:8000/sessions/abc123/messages \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "What is Python?"}'
```

#### GET /sessions/{session_id}/messages

List all messages in a session.

| Parameter | Type | Description |
|---|---|---|
| `session_id` | `str` (path) | Session identifier |

**Response:** `list[SessionMessage]`

**Errors:** 404 if session not found.

### Example

```python
import httpx

async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
    # Create session
    resp = await client.post("/sessions", json={"agent_name": "helper", "title": "Demo"})
    session = resp.json()
    session_id = session["id"]

    # Add messages
    await client.post(f"/sessions/{session_id}/messages", json={"role": "user", "content": "Hello"})
    await client.post(f"/sessions/{session_id}/messages", json={"role": "assistant", "content": "Hi!"})

    # List messages
    resp = await client.get(f"/sessions/{session_id}/messages")
    messages = resp.json()
    print(f"Session has {len(messages)} messages")

    # Delete session
    await client.delete(f"/sessions/{session_id}")
```
