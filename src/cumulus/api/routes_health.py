"""Health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from cumulus.api.security import require_admin_key
from cumulus.schemas import HealthResponse
from cumulus.services.preflight_service import build_preflight_report
from cumulus.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    """Cheap liveness probe: no file I/O and no server paths in the response."""
    return HealthResponse(status="ok", project_name=get_settings().project_name)


@router.get("/health/details", response_model=HealthResponse, dependencies=[Depends(require_admin_key)])
def healthcheck_details() -> HealthResponse:
    """Full preflight report (data sources, resolved paths). Admin only: it exposes server paths."""
    settings = get_settings()
    return HealthResponse(status="ok", project_name=settings.project_name, **build_preflight_report(settings))
