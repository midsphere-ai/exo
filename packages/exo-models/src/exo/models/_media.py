"""Shared multimodal content block conversion helpers for Google providers.

Both the Gemini and Vertex providers use the same Google genai API format
for multimodal content, so this module provides a single implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from exo.types import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    TextBlock,
    VideoBlock,
)

_log = logging.getLogger(__name__)


def content_blocks_to_google(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """Convert a list of ContentBlock objects to Google genai API parts.

    Args:
        blocks: List of ContentBlock objects.

    Returns:
        List of Google-format content part dicts suitable for the
        ``parts`` field in a ``generate_content()`` call.
    """
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({"text": block.text})
        elif isinstance(block, ImageURLBlock):
            url = block.url
            if url.startswith("data:"):
                # data: URI — parse out media_type and base64 data
                # Format: data:<media_type>;base64,<data>
                header, _, data = url.partition(",")
                media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
                parts.append({"inline_data": {"mime_type": media_type, "data": data}})
            else:
                # https:// or gs:// URL — use file_data
                mime_type = _guess_mime_from_url(url)
                parts.append({"file_data": {"file_uri": url, "mime_type": mime_type}})
        elif isinstance(block, ImageDataBlock):
            parts.append({"inline_data": {"mime_type": block.media_type, "data": block.data}})
        elif isinstance(block, AudioBlock):
            mime_type = f"audio/{block.format}"
            parts.append({"inline_data": {"mime_type": mime_type, "data": block.data}})
        elif isinstance(block, VideoBlock):
            if block.url:
                parts.append(
                    {
                        "file_data": {
                            "file_uri": block.url,
                            "mime_type": block.media_type,
                        }
                    }
                )
            elif block.data:
                parts.append({"inline_data": {"mime_type": block.media_type, "data": block.data}})
            else:
                _log.warning("VideoBlock has neither url nor data; skipping")
        elif isinstance(block, DocumentBlock):
            parts.append({"inline_data": {"mime_type": block.media_type, "data": block.data}})
    return parts


def _guess_mime_from_url(url: str) -> str:
    """Guess a MIME type from a URL's file extension.

    Args:
        url: A URL string.

    Returns:
        A MIME type string, defaulting to ``"application/octet-stream"``.
    """
    lower = url.lower().split("?")[0]
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith(".mp3"):
        return "audio/mp3"
    if lower.endswith(".wav"):
        return "audio/wav"
    if lower.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"
