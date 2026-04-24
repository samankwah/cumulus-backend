"""Public point prediction orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path

import pandas as pd

from cumulus.api.errors import (
    ForecastSourceNotConfiguredError,
    ForecastSourceReadError,
    InferenceExecutionError,
    InvalidCoordinatesError,
    InvalidHorizonError,
    ModelArtifactsNotAvailableError,
)
from cumulus.data.location_index import find_location_metadata
from cumulus.data.extractors import extract_locations
from cumulus.frontend_contract.serializers import serialize_daily_forecast
from cumulus.modeling.predictor import predict_dataframe
from cumulus.schemas import AgroCharacteristicsResponse, PointAdvisoryResponse, PointRequest, PredictResponse
from cumulus.services.agro_service import build_agro_characteristics
from cumulus.services.nationwide_service import find_active_point_record
from cumulus.services.source_resolution import (
    resolve_calibration_version,
    resolve_forecast_source,
    open_source_dataset,
)
from cumulus.settings import Settings


logger = logging.getLogger(__name__)


@dataclass
class PointPredictionResult:
    location_id: str
    latitude: float
    longitude: float
    model_version: str
    calibration_version: str
    generated_at: datetime
    horizon_days: int
    forecast_source: str
    data_origin: str
    source_run_id: str
    spatial_resolution_km: float
    forecast_frame: pd.DataFrame
    agro_characteristics: AgroCharacteristicsResponse | None = None
    precomputed_advisory: PointAdvisoryResponse | None = None


def predict_for_point(settings: Settings, request: PointRequest) -> PointPredictionResult:
    horizon_days = request.horizon_days or settings.forecast_horizon_days
    _validate_request(settings, request.latitude, request.longitude, horizon_days)
    cached_result = _build_result_from_active_artifacts(settings, request, horizon_days, request.forecast_source)
    if cached_result is not None:
        logger.info(
            "predict.cache_hit location_id=%s latitude=%.4f longitude=%.4f horizon_days=%s",
            cached_result.location_id,
            request.latitude,
            request.longitude,
            horizon_days,
        )
        return cached_result
    resolved_source = resolve_forecast_source(settings, request.forecast_source)

    location_id = request.location_id or _build_location_id(request.latitude, request.longitude)
    logger.info(
        "predict.request_start location_id=%s latitude=%.4f longitude=%.4f horizon_days=%s",
        location_id,
        request.latitude,
        request.longitude,
        horizon_days,
    )
    logger.info(
        "predict.source_resolved location_id=%s forecast_source=%s forecast_path=%s variables=%s",
        location_id,
        resolved_source.source_id,
        resolved_source.path,
        resolved_source.variables,
    )

    try:
        ds = open_source_dataset(settings, resolved_source)
    except FileNotFoundError as exc:
        raise ForecastSourceReadError(f"Configured forecast source was not found: {resolved_source.path}") from exc
    except Exception as exc:
        raise ForecastSourceReadError(f"Failed to read configured forecast source: {resolved_source.path}") from exc

    locations_df = pd.DataFrame(
        [
            {
                "location_id": location_id,
                "latitude": request.latitude,
                "longitude": request.longitude,
            }
        ]
    )

    try:
        extracted = extract_locations(ds, locations_df, list(ds.data_vars))
        extracted = extracted.sort_values("time").head(horizon_days).reset_index(drop=True)
    except Exception as exc:
        raise ForecastSourceReadError(
            f"Failed to extract a forecast point for latitude={request.latitude} longitude={request.longitude}."
        ) from exc
    if extracted.empty:
        raise ForecastSourceReadError(
            f"No forecast rows were available for latitude={request.latitude} longitude={request.longitude}."
        )

    try:
        predicted, metadata = predict_dataframe(extracted, settings, forecast_source=resolved_source.source_id)
    except FileNotFoundError as exc:
        raise ModelArtifactsNotAvailableError(
            f"Active model artifacts are unavailable under {Path(settings.model_artifact_dir)}."
        ) from exc
    except Exception as exc:
        raise InferenceExecutionError("Prediction failed while running the active downscaling model.") from exc

    logger.info(
        "predict.success location_id=%s rows=%s model_version=%s",
        location_id,
        len(predicted),
        metadata.get("model_version"),
    )
    location_metadata = find_location_metadata(
        settings.config_dir / "locations.yaml",
        location_id=request.location_id,
        latitude=request.latitude,
        longitude=request.longitude,
        tolerance_degrees=settings.nationwide.known_location_tolerance_degrees,
    )
    calibration_version = resolve_calibration_version(
        resolved_source,
        metadata,
        agro_ecological_zone=location_metadata.get("agro_ecological_zone") if location_metadata else None,
    )
    predicted = predicted.sort_values("time").reset_index(drop=True)
    agro_characteristics = build_agro_characteristics(predicted, settings)
    return PointPredictionResult(
        location_id=location_id,
        latitude=request.latitude,
        longitude=request.longitude,
        model_version=str(metadata["model_version"]),
        calibration_version=calibration_version,
        generated_at=datetime.now(UTC),
        horizon_days=horizon_days,
        forecast_source=resolved_source.source_id,
        data_origin=resolved_source.data_origin,
        source_run_id=resolved_source.source_run_id,
        spatial_resolution_km=settings.data_pipeline.target_resolution_km,
        forecast_frame=predicted,
        agro_characteristics=agro_characteristics,
        precomputed_advisory=None,
    )


def build_predict_response(result: PointPredictionResult) -> PredictResponse:
    return PredictResponse(
        location_id=result.location_id,
        latitude=result.latitude,
        longitude=result.longitude,
        model_version=result.model_version,
        calibration_version=result.calibration_version,
        generated_at=result.generated_at,
        horizon_days=result.horizon_days,
        forecast_source=result.forecast_source,
        data_origin=result.data_origin,
        source_run_id=result.source_run_id,
        spatial_resolution_km=result.spatial_resolution_km,
        daily_forecast=serialize_daily_forecast(
            result.forecast_frame[[column for column in ["time", "rainfall_raw_mm", "rainfall_corrected_mm", "temp_c"] if column in result.forecast_frame.columns]]
        ),
        agro_characteristics=result.agro_characteristics,
    )


def _validate_request(settings: Settings, latitude: float, longitude: float, horizon_days: int) -> None:
    bounds = settings.data_pipeline.ghana_bounds
    if not (bounds.latitude_min <= latitude <= bounds.latitude_max):
        raise InvalidCoordinatesError(
            f"Latitude {latitude} is outside the supported range {bounds.latitude_min} to {bounds.latitude_max}."
        )
    if not (bounds.longitude_min <= longitude <= bounds.longitude_max):
        raise InvalidCoordinatesError(
            f"Longitude {longitude} is outside the supported range {bounds.longitude_min} to {bounds.longitude_max}."
        )
    if horizon_days < 1 or horizon_days > settings.max_forecast_horizon_days:
        raise InvalidHorizonError(
            f"horizon_days must be between 1 and {settings.max_forecast_horizon_days}."
        )


def _build_location_id(latitude: float, longitude: float) -> str:
    return f"point_{latitude:.4f}_{longitude:.4f}".replace("-", "m").replace(".", "p")


def _build_result_from_active_artifacts(
    settings: Settings,
    request: PointRequest,
    horizon_days: int,
    forecast_source: str | None,
) -> PointPredictionResult | None:
    record = find_active_point_record(
        settings,
        location_id=request.location_id,
        latitude=request.latitude,
        longitude=request.longitude,
        forecast_source=forecast_source,
    )
    if record is None:
        return None

    forecast_rows = record.get("forecast_frame_rows", [])
    if len(forecast_rows) < horizon_days:
        return None

    frame = pd.DataFrame(forecast_rows[:horizon_days])
    if frame.empty:
        return None
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    precomputed_advisory = None
    if int(record.get("horizon_days", 0)) == horizon_days and "point_advisory" in record:
        precomputed_advisory = PointAdvisoryResponse(**record["point_advisory"])
    agro_characteristics = None
    if "agro_characteristics" in record:
        agro_characteristics = AgroCharacteristicsResponse(**record["agro_characteristics"])

    return PointPredictionResult(
        location_id=str(record["location_id"]),
        latitude=float(record["latitude"]),
        longitude=float(record["longitude"]),
        model_version=str(record["model_version"]),
        calibration_version=str(record["calibration_version"]),
        generated_at=datetime.fromisoformat(str(record["generated_at"])),
        horizon_days=horizon_days,
        forecast_source=str(record.get("forecast_source") or forecast_source or "configured"),
        data_origin=str(record.get("data_origin") or "nationwide_artifact_cache"),
        source_run_id=str(record.get("source_run_id") or f"{forecast_source or 'configured'}-run"),
        spatial_resolution_km=float(record.get("spatial_resolution_km") or settings.data_pipeline.target_resolution_km),
        forecast_frame=frame.sort_values("time").reset_index(drop=True),
        agro_characteristics=agro_characteristics,
        precomputed_advisory=precomputed_advisory,
    )
