"""Forecast endpoints."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from cumulus.api.errors import CumulusServiceError
from cumulus.schemas import (
    ForecastDeterministicProductResponse,
    ForecastDeterministicSampleResponse,
    ForecastProductRefreshResponse,
    ForecastProbabilityProductResponse,
    ForecastProbabilitySampleResponse,
    ForecastRasterMetadataResponse,
    ForecastRasterSampleResponse,
    ForecastRequest,
    ForecastResponse,
    ForecastThemeOptionResponse,
    PointRequest,
    PredictResponse,
)
from cumulus.services.forecast_product_service import (
    get_active_deterministic_product,
    get_active_probability_product,
    get_deterministic_preview_path,
    get_probability_preview_path,
    list_supported_product_themes,
    refresh_forecast_products,
    render_deterministic_tile,
    render_probability_tile,
    sample_active_deterministic_product,
    sample_active_probability_product,
)
from cumulus.services.forecast_raster_service import (
    get_forecast_raster_metadata,
    render_forecast_raster_tile,
    sample_forecast_raster,
)
from cumulus.services.forecast_service import generate_forecast
from cumulus.services.prediction_service import build_predict_response, predict_for_point
from cumulus.settings import get_settings

router = APIRouter(tags=["forecast"])


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PointRequest) -> PredictResponse:
    result = predict_for_point(get_settings(), request)
    return build_predict_response(result)


@router.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(request: ForecastRequest) -> ForecastResponse:
    try:
        results, metadata = generate_forecast(
            get_settings(),
            locations=request.locations,
            forecast_source=request.forecast_source.name,
            forecast_path=request.forecast_source.path,
            forecast_engine=request.forecast_source.engine,
            source_run_id=request.forecast_source.source_run_id,
            variables=request.forecast_source.variables,
            horizon_days=request.horizon_days,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CumulusServiceError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForecastResponse(
        model_version=str(metadata["model_version"]),
        generated_at=datetime.now(UTC),
        spatial_resolution_km=metadata.get("spatial_resolution_km"),
        forecast_source=metadata.get("forecast_source"),
        data_origin=metadata.get("data_origin"),
        source_run_id=metadata.get("source_run_id"),
        calibration_version=metadata.get("calibration_version"),
        seasonal_refresh=metadata.get("seasonal_refresh"),
        results=results,
    )


@router.get("/forecast/raster", response_model=ForecastRasterMetadataResponse)
def forecast_raster_metadata_endpoint(
    variable: str = Query(default="rainfall_daily_mm"),
    horizon_day: int = Query(default=1, ge=1),
    forecast_source: str | None = Query(default=None),
) -> ForecastRasterMetadataResponse:
    payload = get_forecast_raster_metadata(
        get_settings(),
        variable=variable,
        horizon_day=horizon_day,
        forecast_source=forecast_source,
    )
    return ForecastRasterMetadataResponse(**payload)


@router.get("/forecast/raster/tiles/{z}/{x}/{y}.png")
def forecast_raster_tile_endpoint(
    z: int,
    x: int,
    y: int,
    variable: str = Query(default="rainfall_daily_mm"),
    horizon_day: int = Query(default=1, ge=1),
    forecast_source: str | None = Query(default=None),
) -> Response:
    png_bytes = render_forecast_raster_tile(
        get_settings(),
        z=z,
        x=x,
        y=y,
        variable=variable,
        horizon_day=horizon_day,
        forecast_source=forecast_source,
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/forecast/raster/sample", response_model=ForecastRasterSampleResponse)
def forecast_raster_sample_endpoint(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    variable: str = Query(default="rainfall_daily_mm"),
    horizon_day: int = Query(default=1, ge=1),
    forecast_source: str | None = Query(default=None),
) -> ForecastRasterSampleResponse:
    payload = sample_forecast_raster(
        get_settings(),
        latitude=latitude,
        longitude=longitude,
        variable=variable,
        horizon_day=horizon_day,
        forecast_source=forecast_source,
    )
    return ForecastRasterSampleResponse(**payload)


@router.post("/forecast/products/refresh", response_model=ForecastProductRefreshResponse)
def forecast_product_refresh_endpoint(
    theme: str | None = Query(default=None),
) -> ForecastProductRefreshResponse:
    payload = refresh_forecast_products(get_settings(), theme=theme)
    return ForecastProductRefreshResponse(**payload)


@router.get("/forecast/products/options", response_model=list[ForecastThemeOptionResponse])
def forecast_product_options_endpoint() -> list[ForecastThemeOptionResponse]:
    return [ForecastThemeOptionResponse(**item) for item in list_supported_product_themes(get_settings())]


@router.get("/forecast/probability/active", response_model=ForecastProbabilityProductResponse)
def forecast_probability_active_endpoint(
    request: Request,
    theme: str = Query(...),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> ForecastProbabilityProductResponse:
    payload = get_active_probability_product(
        get_settings(),
        theme=theme,
        season_profile=season_profile,
        subseason=subseason,
        api_base_url=str(request.base_url).rstrip("/"),
    )
    return ForecastProbabilityProductResponse(**payload)


@router.get("/forecast/deterministic/active", response_model=ForecastDeterministicProductResponse)
def forecast_deterministic_active_endpoint(
    request: Request,
    theme: str = Query(...),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> ForecastDeterministicProductResponse:
    payload = get_active_deterministic_product(
        get_settings(),
        theme=theme,
        season_profile=season_profile,
        subseason=subseason,
        api_base_url=str(request.base_url).rstrip("/"),
    )
    return ForecastDeterministicProductResponse(**payload)


@router.get("/forecast/probability/sample", response_model=ForecastProbabilitySampleResponse)
def forecast_probability_sample_endpoint(
    theme: str = Query(...),
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> ForecastProbabilitySampleResponse:
    payload = sample_active_probability_product(
        get_settings(),
        theme=theme,
        latitude=latitude,
        longitude=longitude,
        season_profile=season_profile,
        subseason=subseason,
    )
    return ForecastProbabilitySampleResponse(**payload)


@router.get("/forecast/deterministic/sample", response_model=ForecastDeterministicSampleResponse)
def forecast_deterministic_sample_endpoint(
    theme: str = Query(...),
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> ForecastDeterministicSampleResponse:
    payload = sample_active_deterministic_product(
        get_settings(),
        theme=theme,
        latitude=latitude,
        longitude=longitude,
        season_profile=season_profile,
        subseason=subseason,
    )
    return ForecastDeterministicSampleResponse(**payload)


@router.get("/forecast/probability/tiles/{z}/{x}/{y}.png")
def forecast_probability_tile_endpoint(
    z: int,
    x: int,
    y: int,
    theme: str = Query(...),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> Response:
    png_bytes = render_probability_tile(
        get_settings(),
        theme=theme,
        z=z,
        x=x,
        y=y,
        season_profile=season_profile,
        subseason=subseason,
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/forecast/deterministic/tiles/{z}/{x}/{y}.png")
def forecast_deterministic_tile_endpoint(
    z: int,
    x: int,
    y: int,
    theme: str = Query(...),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> Response:
    png_bytes = render_deterministic_tile(
        get_settings(),
        theme=theme,
        z=z,
        x=x,
        y=y,
        season_profile=season_profile,
        subseason=subseason,
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/forecast/probability/preview.png")
def forecast_probability_preview_endpoint(
    theme: str = Query(...),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> FileResponse:
    settings = get_settings()
    target = get_probability_preview_path(
        settings,
        theme=theme,
        season_profile=season_profile,
        subseason=subseason,
    )
    return FileResponse(target, media_type="image/png")


@router.get("/forecast/deterministic/preview.png")
def forecast_deterministic_preview_endpoint(
    theme: str = Query(...),
    season_profile: str | None = Query(default=None),
    subseason: str | None = Query(default=None),
) -> FileResponse:
    settings = get_settings()
    target = get_deterministic_preview_path(
        settings,
        theme=theme,
        season_profile=season_profile,
        subseason=subseason,
    )
    return FileResponse(target, media_type="image/png")
