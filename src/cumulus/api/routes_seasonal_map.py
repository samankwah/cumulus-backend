"""Seasonal advisory map endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from cumulus.schemas import (
    SeasonalMapOptionsResponse,
    SeasonalMapProductResponse,
    SeasonalMapRefreshResponse,
    SeasonalMapRunResponse,
)
from cumulus.services.seasonal_map_service import (
    generate_seasonal_map_product,
    get_active_seasonal_map_product,
    get_seasonal_map_options,
    list_supported_season_profiles,
    refresh_seasonal_map_products,
)
from cumulus.settings import get_settings

router = APIRouter(tags=["seasonal-map"])


@router.post("/seasonal-map/generate", response_model=SeasonalMapRunResponse)
def seasonal_map_generate_endpoint(
    theme: str = Query(...),
    season_profile: str = Query(...),
    mode: str = Query(...),
    subseason: str | None = Query(default=None),
    forecast_source: str | None = Query(default=None),
) -> SeasonalMapRunResponse:
    payload = generate_seasonal_map_product(
        get_settings(),
        theme,
        season_profile,
        mode=mode,
        subseason=subseason,
        forecast_source=forecast_source,
    )
    return SeasonalMapRunResponse(**payload)


@router.get("/seasonal-map/active", response_model=SeasonalMapProductResponse)
def seasonal_map_active_endpoint(
    theme: str = Query(...),
    season_profile: str = Query(...),
    mode: str = Query(...),
    subseason: str | None = Query(default=None),
    forecast_source: str | None = Query(default=None),
) -> SeasonalMapProductResponse:
    payload = get_active_seasonal_map_product(
        get_settings(),
        theme,
        season_profile,
        mode=mode,
        subseason=subseason,
        forecast_source=forecast_source,
    )
    return SeasonalMapProductResponse(**payload)


@router.post("/seasonal-map/refresh", response_model=SeasonalMapRefreshResponse)
def seasonal_map_refresh_endpoint(
    theme: str | None = Query(default=None),
    season_profile: str | None = Query(default=None),
    mode: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
    forecast_source: str | None = Query(default=None),
) -> SeasonalMapRefreshResponse:
    payload = refresh_seasonal_map_products(
        get_settings(),
        theme=theme,
        season_profile=season_profile,
        mode=mode,
        subseason=subseason,
        forecast_source=forecast_source,
    )
    return SeasonalMapRefreshResponse(**payload)


@router.get("/seasonal-map/profiles", response_model=list[str])
def seasonal_map_profiles_endpoint() -> list[str]:
    return list_supported_season_profiles(get_settings())


@router.get("/seasonal-map/options", response_model=SeasonalMapOptionsResponse)
def seasonal_map_options_endpoint() -> SeasonalMapOptionsResponse:
    return SeasonalMapOptionsResponse(**get_seasonal_map_options(get_settings()))
