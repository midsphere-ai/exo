"""Focus mode helpers — simplified now that classifier handles intent detection."""

from __future__ import annotations


def sources_for_focus(focus: str) -> list[str]:
    """Map a focus mode name to enabled source types."""
    mapping = {
        "web": ["web"],
        "academic": ["web", "academic"],
        "reddit": ["web", "discussions"],
        "all": ["web", "academic", "discussions"],
    }
    return mapping.get(focus, ["web"])
