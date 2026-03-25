# Architecture Decision Records — Exo Web

This document records the key architectural decisions made for Exo Web and the reasoning behind each choice.

---

## ADR-001: SQLite as the Primary Database

**Status:** Accepted

**Context:** Exo Web needs a database for storing agent configurations, workflow state, conversation history, audit logs, and more. Options considered: PostgreSQL, MySQL, SQLite.

**Decision:** Use SQLite with WAL (Write-Ahead Logging) mode via aiosqlite.

**Rationale:**
- **Zero-ops deployment** — No separate database process to install, configure, or maintain. The database is a single file (`exo.db`).
- **Single-unit architecture** — Exo Web is designed as a single deployable unit (frontend + backend + database). SQLite aligns perfectly with this goal.
- **WAL mode** enables concurrent reads while a write is in progress, providing sufficient concurrency for the expected workload (single-user or small-team usage).
- **Foreign keys** are enabled at connection time (`PRAGMA foreign_keys=ON`) for referential integrity.
- **Performance** — SQLite is faster than network databases for the read-heavy, low-write workload typical of an AI agent platform.

**Trade-offs:**
- No built-in replication — scaling to multiple server instances would require migrating to PostgreSQL.
- Write concurrency is limited to one writer at a time (WAL mode mitigates read concurrency).
- Large-scale deployments (hundreds of concurrent users) would benefit from a client-server database.

---

## ADR-002: FTS5 for Full-Text Search

**Status:** Accepted

**Context:** Users need to search across agents, workflows, and knowledge bases by name and content.

**Decision:** Use SQLite FTS5 (Full-Text Search 5) virtual tables for search functionality.

**Rationale:**
- **No external dependency** — FTS5 is built into SQLite, requiring no Elasticsearch or other search service.
- **Automatic sync** — FTS5 tables are populated alongside the main tables in the same transaction.
- **Good enough** — For the expected data volumes (hundreds to thousands of items), FTS5 provides fast, relevant search results.
- **Term frequency scoring** — Built-in `bm25()` ranking function provides relevance-ordered results.

**Implementation:** Three FTS5 tables (`search_agents`, `search_workflows`, `search_knowledge_bases`) created in migration `042_create_fts5_tables.sql`. Keyword search falls back to `LIKE` queries with term frequency scoring for non-FTS fields.

---

## ADR-003: Single Multiplexed WebSocket

**Status:** Accepted

**Context:** Exo Web has multiple real-time features: agent chat streaming, workflow execution monitoring, log tailing, sandbox output, and notifications. Each could use a separate WebSocket connection.

**Decision:** Use a single multiplexed WebSocket at `/api/v1/ws` with channel-based routing.

**Rationale:**
- **Connection efficiency** — Browsers limit concurrent WebSocket connections per domain (typically 6-30). A single connection avoids exhausting this limit.
- **Simpler auth** — One connection = one authentication handshake. The session cookie is validated once on connect.
- **Lower overhead** — Each WebSocket connection has TCP and TLS overhead. Multiplexing amortizes this across all real-time features.
- **Channel isolation** — Clients subscribe only to channels they need. The server only sends messages to subscribed connections.

**Protocol:**
```json
{"channel": "chat", "type": "token", "payload": {"text": "Hello"}}
```

**Channels:** `system`, `chat`, `execution`, `logs`, `sandbox`, `notifications`

**Trade-off:** The server-side `WebSocketManager` is slightly more complex than separate endpoints, but the code is ~160 lines and well-contained in `websocket.py`.

---

## ADR-004: Astro 5.x + FastAPI Single Deployable Unit

**Status:** Accepted

**Context:** The platform needs both a rich frontend (visual workflow canvas, agent playground) and a Python backend (LLM orchestration via Exo framework).

**Decision:** Bundle Astro 5.x (Node.js SSG) and FastAPI (Python) as a single deployable package. In development, Vite proxies API requests to the FastAPI server. In production, Astro builds static files served alongside the API.

**Rationale:**
- **Unified deployment** — One `docker run` or process manager command starts everything. No separate frontend and backend deployments to coordinate.
- **Astro's strengths** — Static site generation for fast page loads, React islands for interactive components (workflow canvas), minimal JavaScript shipped to the client.
- **FastAPI's strengths** — Native async Python for LLM provider calls, WebSocket support, Pydantic validation, automatic OpenAPI docs.
- **Exo framework integration** — The backend directly imports `exo-core` and `exo-models` as workspace dependencies, enabling native Python agent execution.

**Trade-offs:**
- Dual toolchain (Node.js + Python) increases build complexity.
- The `exo-web` package has both `package.json` and `pyproject.toml`.

---

## ADR-005: Session Cookies over JWT

**Status:** Accepted

**Context:** The authentication system needs to support browser-based access and API key access for CI/CD.

**Decision:** Use server-side sessions with HttpOnly cookies for browser auth, and API key headers for programmatic access.

**Rationale:**
- **Revocability** — Sessions can be immediately invalidated by deleting the row from the `sessions` table. JWTs cannot be revoked until they expire.
- **CSRF protection** — Each session stores a unique CSRF token. The frontend includes it via `X-CSRF-Token` header on mutating requests (auto-injected by PageLayout).
- **No client-side token storage** — HttpOnly cookies prevent XSS-based token theft. JWTs in localStorage are vulnerable to XSS.
- **Session invalidation on password change** — When a user changes their password, all other sessions are deleted immediately.

**API key auth** (`X-API-Key` header) is CSRF-exempt and used for CI/CD integrations where cookies are impractical.

---

## ADR-006: Sequential SQL Migrations

**Status:** Accepted

**Context:** The database schema evolves as features are added. Need a way to apply schema changes reliably.

**Decision:** Use numbered sequential `.sql` files tracked in a `_migrations` table.

**Rationale:**
- **Simplicity** — No migration framework dependency (no Alembic, no Django migrations). Just numbered SQL files.
- **Deterministic** — Migrations are applied in lexicographic order. Each runs exactly once.
- **Automatic** — `run_migrations()` executes on every application startup. No manual migration commands needed.
- **Transparent** — Each migration is a readable SQL file. Easy to review, audit, and understand.

**Convention:** Files named `NNN_description.sql` (e.g., `001_create_users.sql`, `042_create_fts5_tables.sql`).

Currently at 69 migrations covering the full schema.

---

## ADR-007: API Versioning via URL Prefix

**Status:** Accepted

**Context:** API stability is important for frontend compatibility and external integrations.

**Decision:** Version all API routes under `/api/v1/` with `APIVersionRedirectMiddleware` that 301-redirects legacy `/api/*` paths to `/api/v1/*`.

**Rationale:**
- **Explicit versioning** — Clients know exactly which API version they're using.
- **Backward compatibility** — Old paths redirect rather than 404, giving clients time to migrate.
- **Future-proofing** — When v2 is needed, v1 routes continue to work alongside v2.

**Exception:** The health check at `/api/health` is unversioned for monitoring tools that expect a fixed path.

---

## ADR-008: RBAC with Role Hierarchy

**Status:** Accepted

**Context:** Multi-user deployments need access control beyond simple authentication.

**Decision:** Three hierarchical roles: `viewer` (0) < `developer` (1) < `admin` (2), enforced via `require_role()` dependency factory.

**Rationale:**
- **Simple hierarchy** — Three levels cover the common access patterns without the complexity of per-resource permissions.
- **Composable** — `require_role("admin")` is a standard FastAPI `Depends()` that can be added to any route.
- **Additive** — Higher roles inherit all capabilities of lower roles. An admin can do everything a developer can do.

**Implementation:**
```python
_ROLE_HIERARCHY = {"viewer": 0, "developer": 1, "admin": 2}

def require_role(min_role: str):
    async def _check_role(user = Depends(get_current_user)):
        if _ROLE_HIERARCHY[user["role"]] < _ROLE_HIERARCHY[min_role]:
            raise HTTPException(403)
        return user
    return _check_role
```

---

## ADR-009: In-Memory Rate Limiting

**Status:** Accepted

**Context:** API endpoints need protection against abuse, especially authentication and agent execution endpoints.

**Decision:** Use custom in-memory sliding-window rate limiting in `middleware/rate_limit.py`.

**Rationale:**
- **No external dependency** — No Redis or rate-limit service required. Consistent with the zero-ops philosophy.
- **Per-endpoint tuning** — Different limits for auth (5/min), general (60/min), and agent (10/min) endpoints.
- **Low latency** — In-memory checks add negligible overhead compared to network-based solutions.

**Trade-off:** Rate limit state is lost on restart and not shared across multiple server instances. For single-instance deployments, this is acceptable.

---

## ADR-010: Encrypted API Key Storage

**Status:** Accepted

**Context:** LLM provider API keys (OpenAI, Anthropic, etc.) must be stored securely. Users configure these via the UI.

**Decision:** Encrypt API keys with `EXO_SECRET_KEY` before storing in the database. Never expose encrypted or plaintext keys in API responses.

**Rationale:**
- **At-rest encryption** — If the database file is compromised, API keys are not readable without the secret key.
- **API safety** — Response models use `api_key_set: bool` instead of returning the actual key.
- **Key rotation** — Provider keys table supports multiple keys per provider with load balancing.

**Implementation:** `encrypt_api_key()` / `decrypt_api_key()` in `exo_web/crypto.py`.
