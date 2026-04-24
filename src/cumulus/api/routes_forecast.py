"""Forecast endpoints."""

from __future__ import annotations

from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException

from cumulus.schemas import ForecastRequest, ForecastResponse, PointRequest, PredictResponse
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
        results=results,
    )
