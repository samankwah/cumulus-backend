"""Health endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from cumulus.schemas import HealthResponse
from cumulus.services.preflight_service import build_preflight_report
from cumulus.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", project_name=settings.project_name, **build_preflight_report(settings))
