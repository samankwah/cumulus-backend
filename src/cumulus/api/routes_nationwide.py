"""Nationwide forecast and advisory endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from cumulus.schemas import (
    NationwideGeographySummaryResponse,
    NationwideLocationPageResponse,
    NationwideRunResponse,
)
from cumulus.services.nationwide_service import (
    generate_nationwide_run,
    get_active_run_manifest,
    get_geography_summary,
    list_active_locations,
)
from cumulus.settings import get_settings

router = APIRouter(tags=["nationwide"])


@router.post("/nationwide/generate", response_model=NationwideRunResponse)
def nationwide_generate_endpoint(
    horizon_days: int | None = Query(default=None, ge=1),
    forecast_source: str | None = Query(default=None),
) -> NationwideRunResponse:
    manifest = generate_nationwide_run(get_settings(), horizon_days=horizon_days, forecast_source=forecast_source)
    return NationwideRunResponse(**manifest)


@router.get("/nationwide/run/active", response_model=NationwideRunResponse)
def nationwide_active_run_endpoint(
    forecast_source: str | None = Query(default=None),
) -> NationwideRunResponse:
    return NationwideRunResponse(**get_active_run_manifest(get_settings(), forecast_source=forecast_source))


@router.get("/nationwide/locations", response_model=NationwideLocationPageResponse)
def nationwide_locations_endpoint(
    page: int = Query(default=1, ge=1),
    page_size: int | None = Query(default=None, ge=1),
    region: str | None = Query(default=None),
    district: str | None = Query(default=None),
    forecast_source: str | None = Query(default=None),
) -> NationwideLocationPageResponse:
    payload = list_active_locations(
        get_settings(),
        forecast_source=forecast_source,
        page=page,
        page_size=page_size,
        region=region,
        district=district,
    )
    return NationwideLocationPageResponse(**payload)


@router.get("/nationwide/regions/{region_name}", response_model=NationwideGeographySummaryResponse)
def nationwide_region_endpoint(
    region_name: str,
    forecast_source: str | None = Query(default=None),
) -> NationwideGeographySummaryResponse:
    try:
        summary = get_geography_summary(get_settings(), "region", region_name, forecast_source=forecast_source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return NationwideGeographySummaryResponse(**summary)


@router.get("/nationwide/districts/{district_name}", response_model=NationwideGeographySummaryResponse)
def nationwide_district_endpoint(
    district_name: str,
    forecast_source: str | None = Query(default=None),
) -> NationwideGeographySummaryResponse:
    try:
        summary = get_geography_summary(get_settings(), "district", district_name, forecast_source=forecast_source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return NationwideGeographySummaryResponse(**summary)
