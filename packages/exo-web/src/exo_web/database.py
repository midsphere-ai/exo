"""Database connection management and migration runner."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from exo_web.config import settings

# Extract the file path from the database URL.
# Supports "sqlite+aiosqlite:///path" and plain "path.db" formats.
_DB_URL = settings.database_url
_DB_PATH = _DB_URL.split("///", 1)[-1] if _DB_URL.startswith("sqlite") else _DB_URL

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Yield an aiosqlite connection as an async context manager.

    Used by most of the codebase via ``async with get_db() as db:``.
    For FastAPI dependencies, use :func:`get_db_dep` instead.
    """
    db = await aiosqlite.connect(_DB_PATH)
    db.row_factory = aiosqlite.Row
    try:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        yield db
    finally:
        await db.close()


async def get_db_dep() -> AsyncIterator[aiosqlite.Connection]:
    """FastAPI-compatible dependency that yields an aiosqlite connection.

    FastAPI expects a bare async generator, not an async context manager.
    """
    async with get_db() as db:
        yield db


async def run_migrations() -> list[str]:
    """Run all pending migrations in order.

    Migrations are sequential .sql files in the migrations/ directory,
    named like 001_create_users.sql, 002_create_projects.sql.

    Returns the list of newly applied migration filenames.
    """
    applied: list[str] = []

    async with get_db() as db:
        # Ensure the migrations tracking table exists.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS _migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await db.commit()

        # Get already-applied migrations.
        cursor = await db.execute("SELECT name FROM _migrations ORDER BY id")
        rows = await cursor.fetchall()
        already_applied = {row[0] for row in rows}

        # Find and sort migration files.
        if not MIGRATIONS_DIR.is_dir():
            return applied

        migration_files = sorted(f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql"))

        for filename in migration_files:
            if filename in already_applied:
                continue

            sql = (MIGRATIONS_DIR / filename).read_text()
            await db.executescript(sql)
            await db.execute("INSERT INTO _migrations (name) VALUES (?)", (filename,))
            await db.commit()
            applied.append(filename)

    return applied
