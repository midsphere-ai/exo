# REST API Patterns and Conventions

This document describes the internal API patterns used in Exo Web. All endpoints follow these conventions to ensure consistency.

## Error Response Envelope

All errors use a standardized JSON envelope defined in `exo_web/errors.py`:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": null
  }
}
```

### Standard Error Codes

| HTTP Status | Code | When to Use |
|-------------|------|-------------|
| 400 | `BAD_REQUEST` | Invalid request parameters |
| 401 | `UNAUTHORIZED` | Missing or invalid authentication |
| 403 | `FORBIDDEN` | Insufficient permissions / RBAC violation |
| 404 | `RESOURCE_NOT_FOUND` | Entity does not exist |
| 409 | `CONFLICT` | Duplicate or conflicting state |
| 422 | `VALIDATION_ERROR` | Pydantic request body validation failed |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Unexpected server error |

### Validation Errors

Pydantic validation failures include field-level details:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation error",
    "details": {
      "fields": [
        {
          "field": "email",
          "message": "field required",
          "type": "value_error.missing"
        }
      ]
    }
  }
}
```

Error handlers are registered centrally via `register_error_handlers()` in `app.py`. Do not catch and reformat errors in individual route handlers -- let the global handlers do it.

## Pagination (Cursor-Based)

All list endpoints use cursor-based pagination from `exo_web/pagination.py`.

### Response Shape

```json
{
  "data": [ ... ],
  "pagination": {
    "next_cursor": "base64-encoded-string-or-null",
    "has_more": true,
    "total": 42
  }
}
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cursor` | `str \| None` | `None` | Opaque cursor from a previous response |
| `limit` | `int` | `20` | Items per page (clamped to 1--100) |

### Cursor Encoding

Cursors are base64-encoded `created_at|id` strings. The `paginate()` helper handles encoding and decoding -- never construct cursors manually.

### Usage

```python
from exo_web.pagination import paginate

@router.get("")
async def list_projects(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    async with get_db() as db:
        result = await paginate(
            db,
            table="projects",
            conditions=["user_id = ?"],
            params=[user["id"]],
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        return result.model_dump()
```

### How It Works

1. Fetches `limit + 1` rows to detect whether more pages exist
2. Uses `(created_at < ? OR (created_at = ? AND id < ?))` for deterministic ordering
3. Runs a separate `COUNT(*)` query for `total`
4. Returns `next_cursor` pointing to the last item in the page

## Authentication Patterns

### Session Cookie

The primary auth mechanism is an HTTP-only session cookie (`exo_session`). The `get_current_user` dependency extracts and validates it:

```python
from exo_web.routes.auth import get_current_user

@router.get("")
async def list_things(
    user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    # user has: id, email, role, created_at
    ...
```

Session cookies are set on login with:
- `httponly=True` -- not accessible via JavaScript
- `samesite="lax"` -- CSRF protection
- `path="/"`

### CSRF Protection

Mutating requests (POST, PUT, DELETE) require an `X-CSRF-Token` header. The frontend monkey-patches `window.fetch` in `PageLayout.astro` to include this automatically. The token is fetched from `GET /api/v1/auth/csrf`.

Exempt paths are listed in `middleware/csrf.py` `_EXEMPT_PATHS` and `_EXEMPT_PREFIXES`. Add new exempt routes there when needed (e.g., webhook endpoints).

### API Key Header

CI/CD and external integrations authenticate via the `X-API-Key` header. API keys are hashed before storage and never returned in API responses -- only an `api_key_set: bool` flag.

### WebSocket Auth

WebSocket connections cannot use `Depends()`. Extract the session cookie manually:

```python
session_id = websocket.cookies.get("exo_session")
# Validate session_id against DB before websocket.accept()
```

## RBAC Patterns

Three roles with a strict hierarchy: `viewer < developer < admin`.

### Enforcing Minimum Role

Use the `require_role()` dependency factory:

```python
from exo_web.routes.auth import require_role

@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),
) -> None:
    ...
```

The dependency validates the user's role level against the required minimum and raises `403 FORBIDDEN` if insufficient. It also returns the user dict, so you don't need a separate `get_current_user` dependency.

### Role Hierarchy

```python
_ROLE_HIERARCHY = {"viewer": 0, "developer": 1, "admin": 2}
```

| Role | Level | Typical Access |
|------|-------|---------------|
| `viewer` | 0 | Read-only access |
| `developer` | 1 | Create, update, execute |
| `admin` | 2 | Delete, manage users, audit log, settings |

## Naming Conventions

### URL Patterns

- **Prefix**: `/api/v1/{resource}` -- all routes use the `/api/v1/` prefix
- **Collection names**: plural, kebab-case (`/api/v1/api-keys`, `/api/v1/audit-log`)
- **Resource ID**: `/{resource}/{id}` (UUIDs)
- **Custom actions**: `/{resource}/{id}/{action}` (`/api/v1/workflows/{id}/run`)
- **Health check**: `/api/health` (unversioned)

### Standard CRUD Verbs

| Operation | Method | Path | Status |
|-----------|--------|------|--------|
| List | `GET` | `/api/v1/{resource}` | 200 |
| Create | `POST` | `/api/v1/{resource}` | 201 |
| Get | `GET` | `/api/v1/{resource}/{id}` | 200 |
| Update | `PUT` | `/api/v1/{resource}/{id}` | 200 |
| Delete | `DELETE` | `/api/v1/{resource}/{id}` | 204 |

### Field Names

- **snake_case** for all JSON fields (`created_at`, `user_id`, `model_name`)
- **ISO 8601** for timestamps (`created_at`, `updated_at`, `last_run_at`)
- **Foreign keys**: `{entity}_id` (`project_id`, `user_id`)
- **JSON-stored arrays/objects**: `{name}_json` suffix in DB columns (`tools_json`, `nodes_json`)

### Pydantic Model Names

```
{Entity}Create    -- POST request body
{Entity}Update    -- PUT request body
{Entity}Response  -- API response model
```

All models use `Field(description="...")` for documentation:

```python
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: str = Field("", description="Human-readable description")
```

## Audit Logging

Security-sensitive actions must be audit-logged via `audit_log()` from `services/audit.py`.

### When to Log

- Authentication events: `login`, `logout`
- Resource lifecycle: `create_agent`, `delete_workflow`, `create_deployment`
- Permission changes: `update_role`
- Sensitive operations: `create_api_key`, `delete_provider_key`

### Usage

```python
from exo_web.services.audit import audit_log

# After a successful action:
ip = request.client.host if request.client else None
await audit_log(
    user_id=user["id"],
    action="create_agent",
    entity_type="agent",
    entity_id=agent_id,
    details={"name": body.name},
    ip_address=ip,
)
```

### Error Handling

Audit log writes are fire-and-forget -- failures are logged but never fail the request:

```python
try:
    await audit_log(...)
except Exception:
    _log.exception("Failed to write audit log entry")
```

### Querying

The audit log is queryable at `GET /api/v1/audit-log` (admin only, uses `require_role("admin")`). Supports filtering by `user_id`, `action`, `entity_type`, `date_from`, and `date_to`.

## Testing Patterns

### Test Client Setup

Use `httpx.AsyncClient` with the FastAPI app for route tests:

```python
import pytest
from httpx import ASGITransport, AsyncClient
from exo_web.app import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

### Mocking the Database

Use `unittest.mock.AsyncMock` for aiosqlite connections and a `FakeRow` helper for row objects:

```python
from unittest.mock import AsyncMock, patch

class FakeRow:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()
```

### Test File Naming

Prefix test files uniquely to avoid pytest collection conflicts across workspace packages:

```
test_agent_runtime.py    -- good (specific)
test_agents.py           -- bad (could conflict with exo-core tests)
```

### Mocking Auth

Patch `get_current_user` to bypass authentication in tests:

```python
@patch("exo_web.routes.projects.get_current_user")
async def test_list_projects(mock_auth, client):
    mock_auth.return_value = {"id": "user-1", "email": "test@example.com", "role": "admin"}
    resp = await client.get("/api/v1/projects")
    assert resp.status_code == 200
```

## Migration Conventions

### File Location

`packages/exo-web/src/exo_web/migrations/`

### Naming

Sequential numbering with descriptive name:

```
{NNN}_{description}.sql
```

Examples: `001_create_users.sql`, `008_create_agents.sql`, `061_create_audit_log.sql`

### SQL Rules

Always use `IF NOT EXISTS`:

```sql
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agents_project_id ON agents(project_id);
CREATE INDEX IF NOT EXISTS idx_agents_user_id ON agents(user_id);
```

### Column Conventions

| Pattern | Convention |
|---------|-----------|
| Primary key | `id TEXT PRIMARY KEY` (UUID v4) |
| Timestamps | `TEXT NOT NULL DEFAULT (datetime('now'))` |
| Foreign keys | `REFERENCES {table}(id) ON DELETE CASCADE` |
| JSON storage | `{name}_json TEXT NOT NULL DEFAULT '{}' or '[]'` |
| Optional text | `TEXT NOT NULL DEFAULT ''` |
| Real numbers | `REAL` (e.g., `temperature REAL`) |
| Integers | `INTEGER` (e.g., `max_tokens INTEGER`) |

### Index Rules

- Always index foreign key columns
- Always index columns used in frequent WHERE/ORDER BY clauses
- Use the pattern `idx_{table}_{column}`

### Execution

Migrations run automatically on app startup via the FastAPI lifespan handler. They are tracked in a `_migrations` table -- each file runs exactly once.
