"""Access guards for admin and legacy endpoints."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from cumulus.settings import get_settings


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """Allow the request only with the configured admin key.

    With no key configured the endpoint is hidden (404) rather than open, unless the deployment
    explicitly opts in with ``CUMULUS_ALLOW_UNAUTHENTICATED_ADMIN`` (local dev, tests).
    """
    settings = get_settings()
    expected = settings.admin_api_key
    if not expected:
        if settings.allow_unauthenticated_admin:
            return
        raise HTTPException(status_code=404, detail="Not Found")
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def require_legacy_endpoints() -> None:
    """Hide the retired random-forest/ERA5 endpoints unless CUMULUS_ENABLE_LEGACY_ENDPOINTS is on."""
    if not get_settings().enable_legacy_endpoints:
        raise HTTPException(status_code=404, detail="Not Found")
