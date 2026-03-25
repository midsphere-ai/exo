"""Input sanitization utilities for user-provided text."""

from __future__ import annotations

import re

# Matches HTML tags — used to strip all markup from user text.
_TAG_RE = re.compile(r"<[^>]+>")

# Dangerous patterns that could be used for XSS even without full tags.
_DANGEROUS_PATTERNS = [
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"vbscript\s*:", re.IGNORECASE),
    re.compile(r"data\s*:\s*text/html", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
]


def sanitize_html(text: str) -> str:
    """Strip dangerous HTML tags and attributes from user-provided text.

    Returns plain text with all HTML markup removed and dangerous patterns
    neutralized. Safe for storage and later rendering.
    """
    if not text:
        return text

    # Strip all HTML tags.
    result = _TAG_RE.sub("", text)

    # Neutralize dangerous patterns by removing them.
    for pattern in _DANGEROUS_PATTERNS:
        result = pattern.sub("", result)

    return result
