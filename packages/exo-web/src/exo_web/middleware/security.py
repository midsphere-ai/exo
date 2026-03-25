"""Security headers middleware — CSP, X-Content-Type-Options, X-Frame-Options."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Paths that may be embedded in iframes (widget routes).
_FRAMEABLE_PATHS: set[str] = set()

_CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
    ]
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers on every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"

        if request.url.path in _FRAMEABLE_PATHS:
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
        else:
            response.headers["X-Frame-Options"] = "DENY"

        return response
