"""Audit log service — records security-sensitive actions."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from exo_web.database import get_db

_log = logging.getLogger(__name__)


async def audit_log(
    user_id: str,
    action: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    """Record an audit log entry.

    Parameters
    ----------
    user_id:
        The user performing the action.
    action:
        Action name (e.g. ``login``, ``create_agent``, ``update_role``).
    entity_type:
        Type of entity affected (e.g. ``agent``, ``workflow``, ``user``).
    entity_id:
        ID of the affected entity.
    details:
        Optional dict of extra context serialized as JSON.
    ip_address:
        Client IP address if available.
    """
    entry_id = str(uuid.uuid4())
    details_json = json.dumps(details) if details else None

    try:
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO audit_log (id, user_id, action, entity_type, entity_id, details_json, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, user_id, action, entity_type, entity_id, details_json, ip_address),
            )
            await db.commit()
    except Exception:
        _log.exception("Failed to write audit log entry: action=%s", action)
