"""CORS for the open API: any origin may read /v1, while the rest of the API keeps its allow-list."""

from __future__ import annotations

from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


class OpenApiCORSMiddleware:
    """Answers CORS for paths under ``prefix`` with "any origin, GET only"; other paths pass through.

    Added outside the site's own CORSMiddleware, so /v1 requests never reach that allow-list.
    """

    def __init__(self, app: ASGIApp, prefix: str) -> None:
        self.app = app
        self.prefix = prefix.rstrip("/")
        self.open = CORSMiddleware(
            app,
            allow_origins=["*"],
            allow_methods=["GET", "HEAD", "OPTIONS"],
            allow_headers=["X-API-Key", "If-None-Match"],
            expose_headers=["ETag"],
            max_age=86400,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if path == self.prefix or path.startswith(self.prefix + "/"):
            await self.open(scope, receive, send)
        else:
            await self.app(scope, receive, send)
