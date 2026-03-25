---
title: API Reference
description: REST API endpoint documentation
section: api-reference
order: 1
---

# API Reference

All API endpoints are prefixed with `/api/v1/`. The interactive API documentation is also available:

- **[Swagger UI](/docs)** — Try endpoints interactively
- **[ReDoc](/redoc)** — Alternative read-friendly docs

## Authentication

Requests require authentication via **session cookie** (browser) or **API key header** (`X-API-Key`) for CI/CD integrations.

### Session-based (browser)

1. `POST /api/v1/auth/login` with `{email, password}` — returns a `Set-Cookie: exo_session=...` header
2. Include the session cookie on all subsequent requests
3. For mutating requests (`POST`/`PUT`/`DELETE`), include the `X-CSRF-Token` header — obtain via `GET /api/v1/auth/csrf`

### API Key (CI/CD)

1. Generate an API key from **Settings > API Keys** (or `POST /api/v1/api-keys`)
2. Pass it as the `X-API-Key` header on CI endpoints (`/api/v1/ci/*`)
3. API key endpoints are CSRF-exempt

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | Log in with email and password |
| POST | `/api/v1/auth/register` | Create a new account |
| GET | `/api/v1/auth/me` | Get current user |
| POST | `/api/v1/auth/logout` | Log out |
| GET | `/api/v1/auth/csrf` | Get CSRF token |

## Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/projects` | List projects |
| POST | `/api/v1/projects` | Create a project |
| GET | `/api/v1/projects/:id` | Get project |
| PUT | `/api/v1/projects/:id` | Update project |
| DELETE | `/api/v1/projects/:id` | Delete project |

## Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/agents` | List agents |
| POST | `/api/v1/agents` | Create an agent |
| GET | `/api/v1/agents/:id` | Get agent |
| PUT | `/api/v1/agents/:id` | Update agent |
| DELETE | `/api/v1/agents/:id` | Delete agent |
| POST | `/api/v1/agents/:id/duplicate` | Duplicate an agent |

## Workflows

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/workflows` | List workflows |
| POST | `/api/v1/workflows` | Create a workflow |
| GET | `/api/v1/workflows/:id` | Get workflow with canvas data |
| PUT | `/api/v1/workflows/:id` | Update workflow and canvas |
| DELETE | `/api/v1/workflows/:id` | Delete workflow |
| POST | `/api/v1/workflows/:id/run` | Execute a workflow |

## Tools

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tools` | List all tools (built-in + custom) |
| POST | `/api/v1/tools` | Create a custom tool |
| GET | `/api/v1/tools/:id` | Get tool |
| PUT | `/api/v1/tools/:id` | Update tool |
| DELETE | `/api/v1/tools/:id` | Delete tool |

## Knowledge Bases

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/knowledge-bases` | List knowledge bases |
| POST | `/api/v1/knowledge-bases` | Create a knowledge base |
| GET | `/api/v1/knowledge-bases/:id` | Get knowledge base |
| DELETE | `/api/v1/knowledge-bases/:id` | Delete knowledge base |
| POST | `/api/v1/knowledge-bases/:id/documents` | Upload a document |

## Deployments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/deployments` | List deployments |
| POST | `/api/v1/deployments` | Create a deployment |
| GET | `/api/v1/deployments/:id` | Get deployment |
| PUT | `/api/v1/deployments/:id` | Update deployment |
| DELETE | `/api/v1/deployments/:id` | Delete deployment |

## Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/runs` | List runs with filtering |
| GET | `/api/v1/runs/:id` | Get run details with trace |
| GET | `/api/v1/monitoring/costs` | Get cost aggregation |
| GET | `/api/v1/monitoring/health` | Get system health metrics |
| GET | `/api/v1/alerts` | List alert rules |
| POST | `/api/v1/alerts` | Create an alert rule |

## Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/conversations` | List conversations |
| GET | `/api/v1/conversations/:id` | Get conversation |
| DELETE | `/api/v1/conversations/:id` | Delete conversation |
| GET | `/api/v1/conversations/:id/messages` | Get conversation messages |

## Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/search?q=query` | Global search across entities |

## Providers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/providers` | List configured providers |
| POST | `/api/v1/providers` | Add a provider |
| PUT | `/api/v1/providers/:id` | Update provider |
| DELETE | `/api/v1/providers/:id` | Delete provider |
| POST | `/api/v1/providers/:id/test` | Test provider connectivity |

## Pagination

List endpoints return paginated results:

```json
{
  "data": [...],
  "pagination": {
    "next_cursor": "abc123",
    "has_more": true,
    "total": 42
  }
}
```

Pass `?cursor=abc123` to fetch the next page.

## Error Format

All API errors return a consistent JSON envelope:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Project not found",
    "details": null
  }
}
```

### Error Codes

| Code | HTTP Status | Description | Example |
|------|-------------|-------------|---------|
| `BAD_REQUEST` | 400 | Invalid request parameters | Missing required query param |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication | Expired session cookie |
| `FORBIDDEN` | 403 | Insufficient permissions | Viewer trying to delete |
| `RESOURCE_NOT_FOUND` | 404 | Entity does not exist | Invalid project ID |
| `CONFLICT` | 409 | Duplicate or conflicting state | Duplicate email on register |
| `VALIDATION_ERROR` | 422 | Request body validation failed | See `details.fields` array |
| `RATE_LIMITED` | 429 | Too many requests | Retry after backoff |
| `INTERNAL_ERROR` | 500 | Unexpected server error | Bug — please report |

### Validation Error Details

When `code` is `VALIDATION_ERROR`, the `details` field contains a `fields` array:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation error",
    "details": {
      "fields": [
        {
          "field": "name",
          "message": "String should have at least 1 character",
          "type": "string_too_short"
        }
      ]
    }
  }
}
```
