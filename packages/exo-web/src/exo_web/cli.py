"""CLI commands for exo-web administration."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

import bcrypt

from exo_web.database import MIGRATIONS_DIR, get_db, run_migrations


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


async def _create_user(
    email: str, password: str, *, admin: bool = False, role: str = "developer"
) -> None:
    """Create a user record in the database."""
    # Ensure migrations are up to date.
    await run_migrations()

    user_id = str(uuid.uuid4())
    password_hash = _hash_password(password)

    # --admin flag implies admin role for backward compatibility.
    if admin:
        role = "admin"

    async with get_db() as db:
        # Check if email already exists.
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (email,))
        if await cursor.fetchone():
            print(f"Error: a user with email '{email}' already exists.", file=sys.stderr)
            sys.exit(1)

        await db.execute(
            "INSERT INTO users (id, email, password_hash, is_admin, role) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, password_hash, int(admin), role),
        )
        await db.commit()

    print(f"User created: {email} (id: {user_id}, role: {role})")


async def _migrate(*, status: bool = False) -> None:
    """Run pending migrations or show migration status."""
    if status:
        await _show_migration_status()
    else:
        await _run_migrations()


async def _show_migration_status() -> None:
    """Show applied and pending migrations."""
    # Collect all migration files.
    all_migrations: list[str] = []
    if MIGRATIONS_DIR.is_dir():
        all_migrations = sorted(f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql"))

    if not all_migrations:
        print("No migration files found.")
        return

    # Get applied migrations from DB.
    applied: set[str] = set()
    async with get_db() as db:
        # Check if _migrations table exists.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
        )
        if await cursor.fetchone():
            cursor = await db.execute("SELECT name FROM _migrations ORDER BY id")
            rows = await cursor.fetchall()
            applied = {row[0] for row in rows}

    pending_count = 0
    applied_count = 0
    for name in all_migrations:
        if name in applied:
            print(f"  [applied]  {name}")
            applied_count += 1
        else:
            print(f"  [pending]  {name}")
            pending_count += 1

    print(f"\n{applied_count} applied, {pending_count} pending")


async def _run_migrations() -> None:
    """Run all pending migrations, printing each as it's applied."""
    applied = await run_migrations()
    if applied:
        for name in applied:
            print(f"  Applied: {name}")
        print(f"\n{len(applied)} migration(s) applied successfully.")
    else:
        print("No pending migrations.")


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(prog="exo-web", description="Exo Web CLI")
    subparsers = parser.add_subparsers(dest="command")

    create_user_parser = subparsers.add_parser("create-user", help="Create a new user account")
    create_user_parser.add_argument("--email", required=True, help="User email address")
    create_user_parser.add_argument("--password", required=True, help="User password")
    create_user_parser.add_argument(
        "--admin", action="store_true", default=False, help="Grant admin privileges"
    )
    create_user_parser.add_argument(
        "--role",
        choices=["admin", "developer", "viewer"],
        default="developer",
        help="User role (default: developer)",
    )

    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Show migration status instead of running migrations",
    )

    args = parser.parse_args()

    if args.command == "create-user":
        asyncio.run(_create_user(args.email, args.password, admin=args.admin, role=args.role))
    elif args.command == "migrate":
        try:
            asyncio.run(_migrate(status=args.status))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
