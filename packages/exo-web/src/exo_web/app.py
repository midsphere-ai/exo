"""FastAPI application for Exo Web."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from exo_web.config import settings
from exo_web.database import _DB_PATH, run_migrations
from exo_web.errors import register_error_handlers
from exo_web.middleware.csrf import CSRFMiddleware
from exo_web.middleware.rate_limit import RateLimitMiddleware
from exo_web.middleware.security import SecurityHeadersMiddleware
from exo_web.middleware.versioning import APIVersionRedirectMiddleware
from exo_web.routes.agent_templates import router as agent_templates_router
from exo_web.routes.agents import router as agents_router
from exo_web.routes.alerts import router as alerts_router
from exo_web.routes.annotations import router as annotations_router
from exo_web.routes.applications import router as applications_router
from exo_web.routes.approvals import router as approvals_router
from exo_web.routes.artifacts import router as artifacts_router
from exo_web.routes.audit_log import router as audit_log_router
from exo_web.routes.auth import router as auth_router
from exo_web.routes.benchmarks import router as benchmarks_router
from exo_web.routes.checkpoints import router as checkpoints_router
from exo_web.routes.ci import api_keys_router, ci_router
from exo_web.routes.config_versions import router as config_versions_router
from exo_web.routes.context_state import router as context_state_router
from exo_web.routes.conversations import router as conversations_router
from exo_web.routes.costs import router as costs_router
from exo_web.routes.crews import router as crews_router
from exo_web.routes.deployments import deployed_router
from exo_web.routes.deployments import router as deployments_router
from exo_web.routes.evaluations import router as evaluations_router
from exo_web.routes.integrations import router as integrations_router
from exo_web.routes.knowledge_bases import router as knowledge_bases_router
from exo_web.routes.logs import router as logs_router
from exo_web.routes.metrics import router as metrics_router
from exo_web.routes.models import router as models_router
from exo_web.routes.neuron_pipelines import router as neuron_pipelines_router
from exo_web.routes.notifications import router as notifications_router
from exo_web.routes.observability import router as observability_router
from exo_web.routes.plans import router as plans_router
from exo_web.routes.playground import router as playground_router
from exo_web.routes.plugins import router as plugins_router
from exo_web.routes.projects import router as projects_router
from exo_web.routes.prompt_templates import router as prompt_templates_router
from exo_web.routes.provider_keys import router as provider_keys_router
from exo_web.routes.providers import router as providers_router
from exo_web.routes.retention import router as retention_router
from exo_web.routes.run_queue import router as run_queue_router
from exo_web.routes.runs import router as runs_router
from exo_web.routes.sandbox import router as sandbox_router
from exo_web.routes.schedules import router as schedules_router
from exo_web.routes.search import router as search_router
from exo_web.routes.team import router as team_router
from exo_web.routes.threads import router as threads_router
from exo_web.routes.tools import router as tools_router
from exo_web.routes.vector_stores import router as vector_stores_router
from exo_web.routes.webhooks import router as webhooks_router
from exo_web.routes.workflow_runs import router as workflow_runs_router
from exo_web.routes.workflows import router as workflows_router
from exo_web.routes.workspace_export import router as workspace_export_router
from exo_web.routes.workspace_settings import router as workspace_settings_router
from exo_web.websocket import router as ws_router

logger = logging.getLogger("exo_web")


def _validate_startup() -> None:
    """Validate configuration on startup and log warnings for insecure defaults."""
    # Check for default or missing secret key
    if settings.secret_key == "change-me-in-production":
        if not settings.debug:
            raise RuntimeError("EXO_SECRET_KEY must be changed in production")
        logger.warning(
            "Using default secret key — all encrypted data is insecure. Set EXO_SECRET_KEY env var"
        )
    elif not settings.secret_key:
        logger.warning(
            "Secret key is empty — all encrypted data is insecure. Set EXO_SECRET_KEY env var"
        )

    # Validate database path is writable
    db_path = Path(_DB_PATH)
    db_dir = db_path.parent
    if db_dir.exists() and not os.access(db_dir, os.W_OK):
        logger.error("Database directory is not writable: %s", db_dir)
    elif not db_dir.exists():
        logger.error("Database directory does not exist: %s", db_dir)

    # Log startup config summary (no secrets)
    logger.info(
        "Exo startup config: database=%s, debug=%s, session_expiry=%dh, cors_origins=%s",
        _DB_PATH,
        settings.debug,
        settings.session_expiry_hours,
        settings.cors_origins or "(same-origin)",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate config and run migrations on startup."""
    from exo_web.services.cleanup import start_cleanup, stop_cleanup
    from exo_web.services.mcp_health import start_mcp_health, stop_mcp_health
    from exo_web.services.scheduler import start_scheduler, stop_scheduler

    _validate_startup()
    await run_migrations()
    await start_scheduler()
    await start_cleanup()
    await start_mcp_health()
    yield
    await stop_mcp_health()
    await stop_cleanup()
    await stop_scheduler()


app = FastAPI(
    title="Exo Web API",
    description=(
        "Comprehensive AI agent platform with visual workflow canvas, multi-agent "
        "orchestration, real-time monitoring, RAG pipeline, plugin marketplace, and "
        "five application types.\n\n"
        "## Authentication\n\n"
        "Most endpoints require authentication via **session cookie** (browser) or "
        "**API key header** (`X-API-Key`) for CI/CD integrations.\n\n"
        "- **Session cookie**: Obtained via `POST /api/v1/auth/login`\n"
        "- **API key**: Generated in Settings > API Keys, passed as `X-API-Key` header\n"
        "- **CSRF token**: Required for mutating requests from browsers — "
        "obtain via `GET /api/v1/auth/csrf`\n\n"
        "## Error Format\n\n"
        "All errors return a consistent JSON envelope:\n\n"
        "```json\n"
        '{"error": {"code": "ERROR_CODE", "message": "Human-readable message", '
        '"details": null}}\n'
        "```\n\n"
        "| Code | Status | Description |\n"
        "|------|--------|-------------|\n"
        "| `BAD_REQUEST` | 400 | Invalid request parameters |\n"
        "| `UNAUTHORIZED` | 401 | Missing or invalid authentication |\n"
        "| `FORBIDDEN` | 403 | Insufficient permissions |\n"
        "| `RESOURCE_NOT_FOUND` | 404 | Entity does not exist |\n"
        "| `CONFLICT` | 409 | Duplicate or conflicting state |\n"
        "| `VALIDATION_ERROR` | 422 | Request body validation failed |\n"
        "| `RATE_LIMITED` | 429 | Too many requests |\n"
        "| `INTERNAL_ERROR` | 500 | Unexpected server error |\n"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.debug,
    lifespan=lifespan,
)

register_error_handlers(app)

app.add_middleware(CSRFMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(APIVersionRedirectMiddleware)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-CSRF-Token", "X-API-Key"],
    )

app.include_router(agent_templates_router)
app.include_router(agents_router)
app.include_router(alerts_router)
app.include_router(config_versions_router)
app.include_router(annotations_router)
app.include_router(artifacts_router)
app.include_router(audit_log_router)
app.include_router(approvals_router)
app.include_router(ci_router)
app.include_router(api_keys_router)
app.include_router(checkpoints_router)
app.include_router(neuron_pipelines_router)
app.include_router(plans_router)
app.include_router(applications_router)
app.include_router(context_state_router)
app.include_router(conversations_router)
app.include_router(costs_router)
app.include_router(crews_router)
app.include_router(deployments_router)
app.include_router(deployed_router)
app.include_router(benchmarks_router)
app.include_router(evaluations_router)
app.include_router(integrations_router)
app.include_router(knowledge_bases_router)
app.include_router(observability_router)
app.include_router(logs_router)
app.include_router(auth_router)
app.include_router(metrics_router)
app.include_router(models_router)
app.include_router(notifications_router)
app.include_router(playground_router)
app.include_router(plugins_router)
app.include_router(projects_router)
app.include_router(retention_router)
app.include_router(run_queue_router)
app.include_router(runs_router)
app.include_router(sandbox_router)
app.include_router(schedules_router)
app.include_router(search_router)
app.include_router(prompt_templates_router)
app.include_router(provider_keys_router)
app.include_router(providers_router)
app.include_router(team_router)
app.include_router(threads_router)
app.include_router(tools_router)
app.include_router(vector_stores_router)
app.include_router(webhooks_router)
app.include_router(workflow_runs_router)
app.include_router(workflows_router)
app.include_router(workspace_export_router)
app.include_router(workspace_settings_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check with per-agent and per-provider status."""
    from exo_web.database import get_db

    agents: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []

    try:
        async with get_db() as db:
            # Per-agent health: recent error rate and latency from runs
            cursor = await db.execute(
                """
                SELECT
                    a.id, a.name,
                    COUNT(r.id) AS total_runs,
                    SUM(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END) AS failed_runs,
                    AVG(
                        CASE WHEN r.end_time IS NOT NULL AND r.start_time IS NOT NULL
                        THEN (julianday(r.end_time) - julianday(r.start_time)) * 86400000
                        END
                    ) AS avg_latency_ms
                FROM agents a
                LEFT JOIN runs r ON r.agent_id = a.id
                    AND r.created_at >= datetime('now', '-1 hour')
                GROUP BY a.id, a.name
                ORDER BY a.name
                """
            )
            for row in await cursor.fetchall():
                r = dict(row)
                total = r["total_runs"] or 0
                failed = r["failed_runs"] or 0
                error_rate = round(failed / total * 100, 2) if total > 0 else 0.0
                agents.append(
                    {
                        "id": r["id"],
                        "name": r["name"],
                        "status": "degraded" if error_rate > 50 else "healthy",
                        "error_rate": error_rate,
                        "avg_latency_ms": round(r["avg_latency_ms"] or 0, 2),
                        "recent_runs": total,
                    }
                )

            # Per-provider health: check if keys are configured
            cursor = await db.execute(
                """
                SELECT p.id, p.name, p.provider_type,
                    COUNT(pk.id) AS key_count
                FROM providers p
                LEFT JOIN provider_keys pk ON pk.provider_id = p.id
                GROUP BY p.id, p.name, p.provider_type
                ORDER BY p.name
                """
            )
            for row in await cursor.fetchall():
                r = dict(row)
                providers.append(
                    {
                        "id": r["id"],
                        "name": r["name"],
                        "provider_type": r["provider_type"],
                        "status": "configured" if r["key_count"] > 0 else "no_keys",
                        "key_count": r["key_count"],
                    }
                )

    except Exception:
        return {"status": "degraded", "agents": [], "providers": []}

    return {
        "status": "ok",
        "agents": agents,
        "providers": providers,
    }
