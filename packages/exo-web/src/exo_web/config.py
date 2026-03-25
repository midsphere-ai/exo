"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings, loaded from environment variables."""

    database_url: str = os.getenv("EXO_DATABASE_URL", "sqlite+aiosqlite:///exo.db")
    secret_key: str = os.getenv(
        "EXO_SECRET_KEY", "change-me-in-production"
    )  # Override via EXO_SECRET_KEY — MUST change in production
    debug: bool = os.getenv("EXO_DEBUG", "false").lower() in ("true", "1", "yes")
    session_expiry_hours: int = int(os.getenv("EXO_SESSION_EXPIRY_HOURS", "72"))
    rate_limit_auth: int = int(os.getenv("EXO_RATE_LIMIT_AUTH", "5"))
    rate_limit_general: int = int(os.getenv("EXO_RATE_LIMIT_GENERAL", "200"))
    rate_limit_agent: int = int(os.getenv("EXO_RATE_LIMIT_AGENT", "10"))
    max_upload_mb: float = float(os.getenv("EXO_MAX_UPLOAD_MB", "50"))
    upload_dir: str = os.getenv("EXO_UPLOAD_DIR", "data/uploads/")
    artifact_dir: str = os.getenv("EXO_ARTIFACT_DIR", "data/artifacts/")
    cleanup_interval_hours: int = int(os.getenv("EXO_CLEANUP_INTERVAL_HOURS", "6"))
    cors_origins: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        raw = os.getenv("EXO_CORS_ORIGINS", "")
        self.cors_origins = [o.strip() for o in raw.split(",") if o.strip()] if raw else []


settings = Settings()
