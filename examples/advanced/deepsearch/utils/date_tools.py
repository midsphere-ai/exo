"""Date detection and formatting utilities."""
from __future__ import annotations
from datetime import datetime


def format_date_based_on_type(dt: datetime, fmt_type: str = "full") -> str:
    """Format a datetime based on the type of formatting needed."""
    if fmt_type == "full":
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    elif fmt_type == "date":
        return dt.strftime("%Y-%m-%d")
    elif fmt_type == "short":
        return dt.strftime("%b %d, %Y")
    return dt.isoformat()


def format_date_range(query: dict) -> str | None:
    """Format a date range based on tbs parameter."""
    tbs = query.get("tbs", "")
    if not tbs:
        return None

    now = datetime.utcnow()
    ranges = {
        "qdr:h": "past hour",
        "qdr:d": "past 24 hours",
        "qdr:w": "past week",
        "qdr:m": "past month",
        "qdr:y": "past year",
    }
    return ranges.get(tbs)
