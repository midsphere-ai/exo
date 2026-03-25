"""API versioning redirect middleware.

Redirects legacy ``/api/...`` paths to ``/api/v1/...`` with 301 Moved Permanently.
Infrastructure endpoints like ``/api/health`` are excluded from redirection.
"""

from __future__ import annotations

import re

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

# Paths that should NOT be redirected (infra endpoints without versioning).
_NO_REDIRECT = {"/api/health"}

# Match /api/<something> but NOT /api/v1/<something> and NOT /api/health.
_LEGACY_API_RE = re.compile(r"^/api/(?!v1/|health)")


class APIVersionRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect legacy /api/... requests to /api/v1/... with 301."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        if path in _NO_REDIRECT:
            return await call_next(request)

        if _LEGACY_API_RE.match(path):
            new_path = "/api/v1/" + path[len("/api/") :]
            # Preserve query string.
            query = str(request.url.query)
            url = new_path + ("?" + query if query else "")
            return RedirectResponse(url=url, status_code=301)

        return await call_next(request)
