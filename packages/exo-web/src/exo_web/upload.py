"""Shared file upload infrastructure.

Provides consistent upload handling for document ingestion, artifact management,
CSV imports, and any other file upload use case.
"""

from __future__ import annotations

import logging
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile

from exo_web.config import settings

logger = logging.getLogger(__name__)

# Default allowed MIME types grouped by extension
_MIME_MAP: dict[str, set[str]] = {
    "txt": {"text/plain"},
    "csv": {"text/csv", "text/plain", "application/csv"},
    "json": {"application/json", "text/plain"},
    "pdf": {"application/pdf"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    "md": {"text/markdown", "text/plain"},
    "png": {"image/png"},
    "jpg": {"image/jpeg"},
    "jpeg": {"image/jpeg"},
    "gif": {"image/gif"},
    "svg": {"image/svg+xml"},
    "zip": {"application/zip", "application/x-zip-compressed"},
}


@dataclass
class UploadResult:
    """Result of a successful file upload."""

    path: Path
    original_filename: str
    stored_filename: str
    size_bytes: int
    extension: str
    content_type: str | None


def _get_upload_dir() -> Path:
    """Return the upload directory, creating it if needed."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _validate_extension(filename: str, allowed_types: set[str]) -> str:
    """Validate file extension against allowed types. Returns the extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not ext:
        raise HTTPException(status_code=422, detail="File must have an extension")
    if ext not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(allowed_types))}",
        )
    return ext


def _validate_mime_type(content_type: str | None, ext: str) -> None:
    """Validate MIME type matches the extension if we have a mapping for it."""
    if ext not in _MIME_MAP:
        return  # No mapping — skip MIME check
    if not content_type:
        return  # No content type provided — skip
    if content_type not in _MIME_MAP[ext]:
        raise HTTPException(
            status_code=422,
            detail=f"MIME type '{content_type}' does not match extension '.{ext}'",
        )


def _generate_stored_filename(original_filename: str) -> str:
    """Generate a unique filename using UUID prefix to prevent collisions."""
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
    prefix = uuid.uuid4().hex[:12]
    safe_name = (
        original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    )
    # Keep only safe chars in the base name
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)[:100]
    return f"{prefix}_{safe_name}.{ext}" if ext else f"{prefix}_{safe_name}"


async def handle_upload(
    file: UploadFile,
    allowed_types: set[str],
    max_size_mb: float | None = None,
) -> UploadResult:
    """Validate and save an uploaded file.

    Args:
        file: FastAPI UploadFile instance.
        allowed_types: Set of allowed file extensions (e.g. {"pdf", "csv", "docx"}).
        max_size_mb: Maximum file size in MB. Defaults to EXO_MAX_UPLOAD_MB setting.

    Returns:
        UploadResult with path and metadata.

    Raises:
        HTTPException: On validation failure (422) or file too large (413).
    """
    if max_size_mb is None:
        max_size_mb = settings.max_upload_mb

    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=422, detail="Filename is required")

    # Validate extension
    ext = _validate_extension(file.filename, allowed_types)

    # Validate MIME type
    _validate_mime_type(file.content_type, ext)

    # Read content and validate size
    content = await file.read()
    max_bytes = int(max_size_mb * 1024 * 1024)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {max_size_mb}MB)")
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="File is empty")

    # Generate unique filename and save
    upload_dir = _get_upload_dir()
    stored_filename = _generate_stored_filename(file.filename)
    file_path = upload_dir / stored_filename

    try:
        file_path.write_bytes(content)
    except Exception:
        # Cleanup on write failure
        if file_path.exists():
            file_path.unlink()
        logger.exception("Failed to save uploaded file %s", file.filename)
        raise HTTPException(  # noqa: B904
            status_code=500, detail="Failed to save uploaded file"
        )

    # Resolve content type
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0]

    return UploadResult(
        path=file_path,
        original_filename=file.filename,
        stored_filename=stored_filename,
        size_bytes=len(content),
        extension=ext,
        content_type=content_type,
    )


def cleanup_upload(path: Path) -> None:
    """Remove an uploaded file. Safe to call if file doesn't exist."""
    try:
        if path.exists():
            path.unlink()
            logger.debug("Cleaned up upload: %s", path)
    except OSError:
        logger.warning("Failed to clean up upload: %s", path, exc_info=True)
