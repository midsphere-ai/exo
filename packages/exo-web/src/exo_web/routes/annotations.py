"""Annotation REST API — cached Q&A responses with similarity matching."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html
from exo_web.upload import cleanup_upload, handle_upload

router = APIRouter(prefix="/api/v1/annotations", tags=["annotations"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnnotationCreate(BaseModel):
    query: str = Field(min_length=1, description="Query")
    original_response: str | None = Field(None, description="Original response")
    improved_response: str = Field(min_length=1, description="Improved response")
    similarity_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Similarity threshold"
    )


class AnnotationUpdate(BaseModel):
    query: str | None = Field(None, description="Query")
    original_response: str | None = Field(None, description="Original response")
    improved_response: str | None = Field(None, description="Improved response")
    similarity_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Similarity threshold"
    )


class AnnotationResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    user_id: str = Field(description="Owning user identifier")
    query: str = Field(description="Query")
    original_response: str | None = Field(description="Original response")
    improved_response: str = Field(description="Improved response")
    similarity_threshold: float = Field(description="Similarity threshold")
    usage_count: int = Field(description="Number of times used")
    cost_saved: float = Field(description="Cost saved")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class AnnotationMatchRequest(BaseModel):
    query: str = Field(min_length=1, description="Query")


class AnnotationMatchResponse(BaseModel):
    matched: bool = Field(description="Matched")
    annotation: AnnotationResponse | None = Field(None, description="Annotation")
    similarity_score: float | None = Field(None, description="Similarity score")


class ImportResult(BaseModel):
    imported: int = Field(description="Imported")
    skipped: int = Field(description="Skipped")
    errors: list[str] = Field(description="Errors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings using SequenceMatcher.

    This provides a quick token-overlap similarity suitable for annotation
    matching without requiring external embedding models.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Endpoints — define /search, /import, /match BEFORE /{id} routes
# ---------------------------------------------------------------------------


@router.get("")
async def list_annotations(
    search: str | None = None,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[AnnotationResponse]:
    """List annotations with optional full-text search."""
    async with get_db() as db:
        if search:
            cursor = await db.execute(
                """
                SELECT a.* FROM annotations a
                JOIN annotations_fts f ON a.rowid = f.rowid
                WHERE a.user_id = ? AND annotations_fts MATCH ?
                ORDER BY a.created_at DESC
                """,
                (user["id"], search),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM annotations WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
    return [AnnotationResponse(**dict(r)) for r in rows]


@router.post("/import", response_model=ImportResult)
async def import_annotations(
    file: UploadFile,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> ImportResult:
    """Bulk import annotations from a CSV with query,response columns."""
    result = await handle_upload(file, allowed_types={"csv"}, max_size_mb=10)
    try:
        csv_text = result.path.read_text(encoding="utf-8")
    except Exception:
        cleanup_upload(result.path)
        raise HTTPException(status_code=422, detail="Failed to read CSV file")  # noqa: B904

    cleanup_upload(result.path)

    reader = csv.DictReader(io.StringIO(csv_text))
    if (
        not reader.fieldnames
        or "query" not in reader.fieldnames
        or "response" not in reader.fieldnames
    ):
        raise HTTPException(
            status_code=422,
            detail="CSV must have 'query' and 'response' columns",
        )

    imported = 0
    skipped = 0
    errors: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        for i, row in enumerate(reader, start=2):  # row 1 is header
            query = (row.get("query") or "").strip()
            response = (row.get("response") or "").strip()
            if not query or not response:
                skipped += 1
                errors.append(f"Row {i}: empty query or response")
                continue

            annotation_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO annotations (id, user_id, query, improved_response, similarity_threshold, usage_count, cost_saved, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0.8, 0, 0.0, ?, ?)
                """,
                (
                    annotation_id,
                    user["id"],
                    sanitize_html(query),
                    sanitize_html(response),
                    now,
                    now,
                ),
            )
            imported += 1

        await db.commit()

    return ImportResult(imported=imported, skipped=skipped, errors=errors)


@router.post("/match", response_model=AnnotationMatchResponse)
async def match_annotation(
    body: AnnotationMatchRequest,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Find a matching annotation for a query using similarity scoring.

    When a match is found above the annotation's threshold, returns the
    improved_response — allowing it to be served instead of an LLM call.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM annotations WHERE user_id = ? ORDER BY usage_count DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()

        best_match: dict[str, Any] | None = None
        best_score = 0.0

        for row in rows:
            annotation = dict(row)
            score = _similarity(body.query, annotation["query"])
            if score >= annotation["similarity_threshold"] and score > best_score:
                best_match = annotation
                best_score = score

        if best_match:
            # Increment usage_count
            await db.execute(
                "UPDATE annotations SET usage_count = usage_count + 1 WHERE id = ?",
                (best_match["id"],),
            )
            await db.commit()
            return {
                "matched": True,
                "annotation": AnnotationResponse(**best_match),
                "similarity_score": best_score,
            }

    return {"matched": False, "annotation": None, "similarity_score": None}


@router.post("", response_model=AnnotationResponse, status_code=201)
async def create_annotation(
    body: AnnotationCreate,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> AnnotationResponse:
    """Create a new annotation."""
    annotation_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO annotations (id, user_id, query, original_response, improved_response,
                                     similarity_threshold, usage_count, cost_saved, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0.0, ?, ?)
            """,
            (
                annotation_id,
                user["id"],
                sanitize_html(body.query),
                sanitize_html(body.original_response) if body.original_response else None,
                sanitize_html(body.improved_response),
                body.similarity_threshold,
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,))
        row = await cursor.fetchone()
    return AnnotationResponse(**dict(row))  # type: ignore[arg-type]


@router.get("/{annotation_id}", response_model=AnnotationResponse)
async def get_annotation(
    annotation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> AnnotationResponse:
    """Get a single annotation by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM annotations WHERE id = ? AND user_id = ?",
            (annotation_id, user["id"]),
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return AnnotationResponse(**dict(row))


@router.put("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(
    annotation_id: str,
    body: AnnotationUpdate,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> AnnotationResponse:
    """Update an annotation."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM annotations WHERE id = ? AND user_id = ?",
            (annotation_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Annotation not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=422, detail="No fields to update")

        # Sanitize text fields
        for field in ("query", "original_response", "improved_response"):
            if field in updates and updates[field] is not None:
                updates[field] = sanitize_html(updates[field])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*updates.values(), annotation_id]

        await db.execute(
            f"UPDATE annotations SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,))
        row = await cursor.fetchone()
    return AnnotationResponse(**dict(row))  # type: ignore[arg-type]


@router.delete("/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an annotation."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM annotations WHERE id = ? AND user_id = ?",
            (annotation_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Annotation not found")

        await db.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        await db.commit()
