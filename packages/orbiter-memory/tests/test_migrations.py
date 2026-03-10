"""Tests for the memory migration system."""

from __future__ import annotations

import pytest

aiosqlite = pytest.importorskip("aiosqlite")

from orbiter.memory.backends.sqlite import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    SQLiteMemoryStore,
)
from orbiter.memory.migrations import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    Migration,
    MigrationError,
    MigrationRegistry,
    run_migrations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store():
    """Fresh in-memory SQLite store."""
    async with SQLiteMemoryStore(":memory:") as s:
        yield s


def _noop_up(db):
    """No-op migration for testing."""


async def _noop_async_up(db):
    """Async no-op migration."""


# ---------------------------------------------------------------------------
# Migration dataclass
# ---------------------------------------------------------------------------


class TestMigrationDataclass:
    def test_frozen(self) -> None:
        m = Migration(version=1, description="test", up=_noop_async_up)
        with pytest.raises(AttributeError):
            m.version = 2  # type: ignore[misc]

    def test_defaults(self) -> None:
        m = Migration(version=1, description="test", up=_noop_async_up)
        assert m.down is None

    def test_with_down(self) -> None:
        m = Migration(version=1, description="test", up=_noop_async_up, down=_noop_async_up)
        assert m.down is not None


# ---------------------------------------------------------------------------
# MigrationRegistry
# ---------------------------------------------------------------------------


class TestMigrationRegistry:
    def test_register_and_list(self) -> None:
        reg = MigrationRegistry()
        m1 = Migration(version=1, description="first", up=_noop_async_up)
        m2 = Migration(version=2, description="second", up=_noop_async_up)
        reg.register(m1)
        reg.register(m2)
        assert reg.all == [m1, m2]

    def test_register_out_of_order_sorts(self) -> None:
        reg = MigrationRegistry()
        m3 = Migration(version=3, description="third", up=_noop_async_up)
        m1 = Migration(version=1, description="first", up=_noop_async_up)
        reg.register(m3)
        reg.register(m1)
        assert [m.version for m in reg.all] == [1, 3]

    def test_duplicate_version_raises(self) -> None:
        reg = MigrationRegistry()
        m1 = Migration(version=1, description="first", up=_noop_async_up)
        reg.register(m1)
        dup = Migration(version=1, description="duplicate", up=_noop_async_up)
        with pytest.raises(MigrationError, match="Duplicate"):
            reg.register(dup)

    def test_list_pending_from_zero(self) -> None:
        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="a", up=_noop_async_up))
        reg.register(Migration(version=2, description="b", up=_noop_async_up))
        pending = reg.list_pending(0)
        assert [m.version for m in pending] == [1, 2]

    def test_list_pending_skips_applied(self) -> None:
        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="a", up=_noop_async_up))
        reg.register(Migration(version=2, description="b", up=_noop_async_up))
        reg.register(Migration(version=3, description="c", up=_noop_async_up))
        pending = reg.list_pending(2)
        assert [m.version for m in pending] == [3]

    def test_list_pending_none_remaining(self) -> None:
        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="a", up=_noop_async_up))
        assert reg.list_pending(1) == []

    def test_empty_registry(self) -> None:
        reg = MigrationRegistry()
        assert reg.all == []
        assert reg.list_pending(0) == []


# ---------------------------------------------------------------------------
# run_migrations — SQLite
# ---------------------------------------------------------------------------


class TestRunMigrationsSQLite:
    async def test_no_pending_returns_zero(self, store: SQLiteMemoryStore) -> None:
        reg = MigrationRegistry()
        count = await run_migrations(store, reg)
        assert count == 0

    async def test_applies_single_migration(self, store: SQLiteMemoryStore) -> None:
        executed = []

        async def add_column(db):
            await db.execute(
                "ALTER TABLE memory_items ADD COLUMN tags TEXT DEFAULT ''"
            )
            executed.append(1)

        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="add tags", up=add_column))

        count = await run_migrations(store, reg)
        assert count == 1
        assert len(executed) == 1

        # Verify _migrations table was updated
        db = store._ensure_init()
        cursor = await db.execute("SELECT version, description FROM _migrations")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["version"] == 1
        assert rows[0]["description"] == "add tags"

    async def test_applies_multiple_in_order(self, store: SQLiteMemoryStore) -> None:
        order = []

        async def up_v1(db):
            order.append(1)

        async def up_v2(db):
            order.append(2)

        async def up_v3(db):
            order.append(3)

        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="first", up=up_v1))
        reg.register(Migration(version=2, description="second", up=up_v2))
        reg.register(Migration(version=3, description="third", up=up_v3))

        count = await run_migrations(store, reg)
        assert count == 3
        assert order == [1, 2, 3]

    async def test_skips_already_applied(self, store: SQLiteMemoryStore) -> None:
        executed = []

        async def up_v1(db):
            executed.append(1)

        async def up_v2(db):
            executed.append(2)

        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="first", up=up_v1))
        reg.register(Migration(version=2, description="second", up=up_v2))

        # Run once — both applied
        count = await run_migrations(store, reg)
        assert count == 2
        assert executed == [1, 2]

        # Run again — nothing new
        executed.clear()
        count = await run_migrations(store, reg)
        assert count == 0
        assert executed == []

    async def test_incremental_migration(self, store: SQLiteMemoryStore) -> None:
        """Register v1 and run, then register v2 and run — only v2 should execute."""
        executed = []

        async def up_v1(db):
            executed.append(1)

        async def up_v2(db):
            executed.append(2)

        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="first", up=up_v1))
        await run_migrations(store, reg)
        assert executed == [1]

        # Add v2 and run again
        executed.clear()
        reg.register(Migration(version=2, description="second", up=up_v2))
        count = await run_migrations(store, reg)
        assert count == 1
        assert executed == [2]

    async def test_error_stops_remaining(self, store: SQLiteMemoryStore) -> None:
        executed = []

        async def up_v1(db):
            executed.append(1)

        async def up_v2(db):
            msg = "intentional failure"
            raise RuntimeError(msg)

        async def up_v3(db):
            executed.append(3)

        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="ok", up=up_v1))
        reg.register(Migration(version=2, description="fails", up=up_v2))
        reg.register(Migration(version=3, description="never", up=up_v3))

        with pytest.raises(MigrationError, match="v2 failed"):
            await run_migrations(store, reg)

        # v1 was applied, v2 and v3 were not
        assert executed == [1]

        db = store._ensure_init()
        cursor = await db.execute("SELECT version FROM _migrations ORDER BY version")
        rows = await cursor.fetchall()
        assert [r["version"] for r in rows] == [1]

    async def test_migrations_table_tracks_applied_at(self, store: SQLiteMemoryStore) -> None:
        async def up_v1(db):
            pass

        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="test", up=up_v1))
        await run_migrations(store, reg)

        db = store._ensure_init()
        cursor = await db.execute("SELECT applied_at FROM _migrations WHERE version = 1")
        row = await cursor.fetchone()
        assert row is not None
        assert row["applied_at"] is not None  # has a timestamp


class TestRunMigrationsEdgeCases:
    async def test_unsupported_store_type(self) -> None:
        reg = MigrationRegistry()
        with pytest.raises(MigrationError, match="Unsupported"):
            await run_migrations(object(), reg)

    async def test_real_schema_change(self, store: SQLiteMemoryStore) -> None:
        """Verify a real ALTER TABLE migration actually changes the schema."""

        async def add_priority(db):
            await db.execute(
                "ALTER TABLE memory_items ADD COLUMN priority INTEGER DEFAULT 0"
            )

        reg = MigrationRegistry()
        reg.register(Migration(version=1, description="add priority", up=add_priority))
        await run_migrations(store, reg)

        # Verify column exists by inserting with it
        db = store._ensure_init()
        await db.execute(
            "INSERT INTO memory_items "
            "(id, content, memory_type, status, metadata, extra_json, "
            "created_at, updated_at, priority) "
            "VALUES ('t1', 'test', 'human', 'accepted', '{}', '{}', "
            "'2024-01-01', '2024-01-01', 5)"
        )
        await db.commit()

        cursor = await db.execute("SELECT priority FROM memory_items WHERE id = 't1'")
        row = await cursor.fetchone()
        assert row["priority"] == 5
