# Security

Exo Web ships with layered security defaults: encrypted secrets, session-based auth, RBAC, CSRF protection, rate limiting, input sanitization, and restrictive HTTP headers. This guide covers configuration, hardening, and the threat model for production deployments.

## Secret Key (`EXO_SECRET_KEY`)

The secret key derives the Fernet encryption key used to encrypt provider API keys at rest (via `crypto.py`). If the key changes, all previously encrypted data becomes unreadable.

**Generate a secure key:**

```bash
export EXO_SECRET_KEY=$(openssl rand -hex 32)
```

**Startup validation:** The application logs a warning if the default `change-me-in-production` value is detected. Never deploy with the default.

| Environment Variable | Default | Purpose |
|---|---|---|
| `EXO_SECRET_KEY` | `change-me-in-production` | Derives the Fernet key for API-key encryption |

**Key rotation:** There is no automatic rotation. To rotate, decrypt all stored provider keys with the old secret, update the environment variable, then re-encrypt. A maintenance script or manual DB migration is required.

## CORS

Cross-origin requests are blocked by default (same-origin only). Enable CORS only when your frontend is served from a different origin than the API.

```bash
# Single origin
export EXO_CORS_ORIGINS="https://app.example.com"

# Multiple origins (comma-separated)
export EXO_CORS_ORIGINS="https://app.example.com,https://widget.example.com"
```

When enabled, the middleware allows:

- **Methods:** GET, POST, PUT, DELETE, OPTIONS
- **Headers:** Content-Type, Authorization, X-CSRF-Token, X-API-Key
- **Credentials:** Allowed (`allow_credentials=True`)

**Hardening tips:**

- Never use `*` as an origin — it disables credential support and exposes the API to any domain.
- List only the exact origins that need access.
- If you only serve Exo from a single domain, leave `EXO_CORS_ORIGINS` unset.

## Rate Limiting

An in-memory sliding-window rate limiter protects all API endpoints. Three separate limits apply depending on the endpoint category:

| Environment Variable | Default | Scope | Key |
|---|---|---|---|
| `EXO_RATE_LIMIT_AUTH` | 5/min | `POST /api/v1/auth/login` | Client IP |
| `EXO_RATE_LIMIT_GENERAL` | 60/min | All other `/api/` routes | Session ID (or IP) |
| `EXO_RATE_LIMIT_AGENT` | 10/min | Workflow `/run` and `/debug` | Session ID (or IP) |

The `/api/health` endpoint is exempt.

**Response headers** on every API response:

| Header | Meaning |
|---|---|
| `X-RateLimit-Limit` | Maximum requests in the window |
| `X-RateLimit-Remaining` | Requests remaining |
| `X-RateLimit-Reset` | Seconds until the window resets |
| `Retry-After` | Present only on `429 Too Many Requests` |

**Tuning for production:**

- Increase `EXO_RATE_LIMIT_GENERAL` if legitimate users hit 429s during normal use.
- Keep `EXO_RATE_LIMIT_AUTH` low (5–10) to slow credential-stuffing attacks.
- The sliding window is in-memory and per-process — behind a load balancer with multiple workers, effective limits multiply by the number of processes. Use an external rate limiter (e.g., nginx `limit_req`) for coordinated limiting across workers.

## RBAC (Role-Based Access Control)

Three roles form a strict hierarchy:

| Role | Level | Capabilities |
|---|---|---|
| `viewer` | 0 | Read-only access to all resources |
| `developer` | 1 | Create/modify agents, workflows, templates; execute runs |
| `admin` | 2 | Full access — manage users, provider keys, settings, CI keys |

Roles are enforced via the `require_role()` FastAPI dependency:

```python
from exo_web.routes.auth import require_role
from fastapi import Depends

@router.delete("/dangerous")
async def dangerous_endpoint(user: dict = Depends(require_role("admin"))):
    ...
```

A user with a higher role automatically satisfies checks for lower roles (an admin passes a `require_role("developer")` check).

**Permission matrix:**

| Action | viewer | developer | admin |
|---|---|---|---|
| View agents/workflows | Yes | Yes | Yes |
| Create/edit agents | No | Yes | Yes |
| Execute workflows | No | Yes | Yes |
| Manage provider keys | No | No | Yes |
| Manage users/roles | No | No | Yes |
| Create CI/CD API keys | No | No | Yes |
| View settings | Yes | Yes | Yes |
| Modify settings | No | No | Yes |

## Session Management

Sessions use server-side storage (SQLite `sessions` table) with HttpOnly cookies.

| Property | Value |
|---|---|
| Cookie name | `exo_session` |
| HttpOnly | Yes (not accessible to JavaScript) |
| SameSite | `Lax` (CSRF mitigation) |
| Session ID | `uuid4()` (cryptographically random) |
| CSRF token | `secrets.token_urlsafe(32)` per session |
| Default expiry | 72 hours |

**Configuring expiry:**

```bash
export EXO_SESSION_EXPIRY_HOURS=24   # Shorter sessions for sensitive environments
```

**Expiry enforcement:** Every authenticated request validates `expires_at > datetime('now')`. Expired sessions are rejected with 401.

**Cleanup:** Expired sessions remain in the database until explicitly purged. To clean up:

```sql
DELETE FROM sessions WHERE expires_at < datetime('now');
```

Run this periodically (e.g., daily cron job) to reclaim space.

**Force logout scenarios:**

| Event | Effect |
|---|---|
| User logs out | Current session deleted |
| Password change | All sessions except current are deleted |
| Admin password reset | All sessions for that user are deleted |

## API Key Management (CI/CD)

CI/CD integrations authenticate via API keys in the `X-API-Key` header instead of session cookies.

**Key lifecycle:**

1. **Generate:** An admin calls `POST /api/v1/settings/api-keys` with a label and permissions list. The raw key (prefixed `orb_ci_...`) is returned once.
2. **Store:** Only the SHA-256 hash is persisted in the `api_keys` table. The raw key cannot be recovered.
3. **Use:** CI systems pass the key in the `X-API-Key` header. The server hashes the provided key and looks up the hash.
4. **Track:** `last_used_at` is updated on every successful authentication.
5. **Revoke:** `DELETE /api/v1/settings/api-keys/{key_id}` removes the key immediately.

**Permissions:**

| Permission | Grants |
|---|---|
| `ci:deploy` | Deploy agents and workflows |
| `ci:evaluate` | Trigger evaluation runs |
| `ci:status` | Check deployment status |

Keys are scoped to a subset of these permissions at creation time. The `_require_permission()` dependency checks the key's permission list on each request.

**Best practices:**

- Create separate keys per CI pipeline (one for deploy, one for eval) with minimal permissions.
- Rotate keys periodically — revoke the old key and generate a new one.
- Store keys in your CI system's secret store (e.g., GitHub Actions secrets), never in code.
- Monitor `last_used_at` via `GET /api/v1/settings/api-keys` to detect unused or compromised keys.

## CSP Headers

The `SecurityHeadersMiddleware` injects security headers on every response:

| Header | Value | Purpose |
|---|---|---|
| `Content-Security-Policy` | `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:` | Restricts resource loading to same origin |
| `X-Content-Type-Options` | `nosniff` | Prevents MIME-type sniffing |
| `X-Frame-Options` | `DENY` (default) or `SAMEORIGIN` | Clickjacking protection |

**Customizing for embedding:**

If you need to embed Exo pages in an iframe (e.g., an agent widget), add the path to `_FRAMEABLE_PATHS` in `middleware/security.py`:

```python
_FRAMEABLE_PATHS: set[str] = {"/embed/chat", "/embed/agent"}
```

Paths in this set receive `X-Frame-Options: SAMEORIGIN` instead of `DENY`, allowing same-origin iframes.

**Extending the CSP:**

To allow external resources (e.g., a CDN for fonts or analytics scripts), modify the `_CSP` string in `middleware/security.py`:

```python
_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' https://cdn.example.com",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "img-src 'self' data: https:",
    "font-src 'self' https://fonts.gstatic.com",
])
```

Only add origins you trust. Every addition widens the attack surface.

## Input Sanitization

All user-provided text fields are sanitized before database storage using `sanitize_html()` from `exo_web/sanitize.py`. The function:

1. **Strips all HTML tags** — removes any `<tag>` markup.
2. **Neutralizes dangerous patterns:**
   - `javascript:` and `vbscript:` protocol handlers
   - `data:text/html` URIs
   - Inline event handlers (`onclick=`, `onerror=`, etc.)

**Where it's applied:** Every route that writes user-provided text calls `sanitize_html()` before storage — agent names, descriptions, instructions, workflow metadata, project names, template labels, annotations, API key labels, and all other text fields across 20+ route files.

**What it does NOT do:**

- It does not HTML-encode output. The assumption is that all stored text is plain text rendered in contexts that don't interpret HTML (JSON API responses, Astro templates with automatic escaping).
- It does not validate or sanitize file uploads — see `upload.py` for upload-specific validation (allowed MIME types, size limits).

## CSRF Protection

The CSRF middleware (`middleware/csrf.py`) protects all state-changing requests:

- **Token generation:** A `csrf_token` is created per session (stored in the `sessions` table) using `secrets.token_urlsafe(32)`.
- **Token delivery:** The token is included in the initial page load and available to client-side JavaScript.
- **Token verification:** POST, PUT, and DELETE requests must include the token in the `X-CSRF-Token` header. The middleware compares it against the session's stored token.
- **Auto-injection:** The frontend monkey-patches `window.fetch` in `PageLayout.astro` to automatically add the `X-CSRF-Token` header on all mutation requests. No manual header management is needed in page scripts.

**Exempt paths:** Endpoints that accept non-browser authentication (webhooks, CI/CD) are listed in `_EXEMPT_PATHS` and `_EXEMPT_PREFIXES` in `middleware/csrf.py`.

## Middleware Stack Order

Middleware executes in reverse registration order (last registered = outermost). The registration order in `app.py`:

```
1. CSRFMiddleware          (innermost — runs last)
2. RateLimitMiddleware
3. SecurityHeadersMiddleware (outermost — runs first)
```

This means security headers are applied to every response (including rate-limited 429s and CSRF 403s), rate limiting is checked before CSRF (no point validating CSRF on a rate-limited request), and CSRF validation happens closest to the route handler.

## Production Hardening Checklist

- [ ] Set `EXO_SECRET_KEY` to a random 64-character hex string
- [ ] Set `EXO_SESSION_EXPIRY_HOURS` appropriate to your security requirements
- [ ] Leave `EXO_CORS_ORIGINS` unset (or set to exact production origins only)
- [ ] Deploy behind a reverse proxy (nginx, Caddy) with TLS termination
- [ ] Enable the reverse proxy's rate limiting for coordinated protection across workers
- [ ] Set up a cron job to purge expired sessions periodically
- [ ] Create CI/CD API keys with minimal permissions per pipeline
- [ ] Review `_FRAMEABLE_PATHS` — keep empty unless you need iframe embedding
- [ ] Monitor audit logs for suspicious activity (login failures, role changes)
- [ ] Ensure `EXO_DEBUG` is `false` in production
