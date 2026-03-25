"""Standardized error response format for the Exo Web API.

All API errors return:
    {"error": {"code": "<ERROR_CODE>", "message": "<human-readable>", "details": <object|null>}}
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Map HTTP status codes to standard error codes
_STATUS_CODE_MAP: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "RESOURCE_NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
}


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict | list | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
            }
        },
    )


async def _http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    code = _STATUS_CODE_MAP.get(exc.status_code, "INTERNAL_ERROR")
    message = str(exc.detail) if exc.detail else code
    return _error_response(exc.status_code, code, message)


async def _validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    field_errors = []
    for err in exc.errors():
        loc = err.get("loc", ())
        # Skip the first element if it's "body"/"query"/"path" prefix
        parts = loc[1:] if loc else loc
        field_path = ".".join(str(part) for part in parts)
        field_errors.append(
            {
                "field": field_path or None,
                "message": err.get("msg", ""),
                "type": err.get("type", ""),
            }
        )
    return _error_response(
        422,
        "VALIDATION_ERROR",
        "Validation error",
        {"fields": field_errors},
    )


async def _generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return _error_response(500, "INTERNAL_ERROR", "Internal server error")


def register_error_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI app."""
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_exception_handler)  # type: ignore[arg-type]
