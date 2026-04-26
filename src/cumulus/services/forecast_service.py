"""Forecast service orchestration."""

from __future__ import annotations

import pandas as pd

from cumulus.api.errors import SeasonalMapRefreshFailedError
from cumulus.advisory.rules import build_advisory
from cumulus.data.extractors import extract_locations
from cumulus.frontend_contract.serializers import serialize_advisory, serialize_daily_forecast
from cumulus.modeling.predictor import predict_dataframe
from cumulus.schemas import ForecastResultResponse, LocationRequest
from cumulus.services.agro_service import build_agro_characteristics
from cumulus.services.seasonal_map_service import refresh_seasonal_map_products
from cumulus.services.source_resolution import open_source_dataset, resolve_calibration_version, resolve_forecast_source
from cumulus.settings import Settings


def generate_forecast(
    settings: Settings,
    locations: list[LocationRequest],
    forecast_source: str | None,
    forecast_path: str | None,
    forecast_engine: str | None,
    source_run_id: str | None,
    variables: list[str] | None,
    horizon_days: int,
) -> tuple[list[ForecastResultResponse], dict[str, object]]:
    resolved_source = resolve_forecast_source(
        settings,
        forecast_source,
        override_path=forecast_path,
        override_engine=forecast_engine,
        override_variables=variables,
        override_source_run_id=source_run_id,
    )
    ds = open_source_dataset(settings, resolved_source)
    available_variables = [resolved_source.variable_aliases.get(variable, variable) for variable in resolved_source.variables]
    available_variables = [variable for variable in available_variables if variable in ds.data_vars]
    locations_df = pd.DataFrame(
        [{"location_id": item.location_id, "latitude": item.lat, "longitude": item.lon} for item in locations]
    )
    extracted = extract_locations(ds, locations_df, available_variables)
    extracted = extracted.sort_values(["location_id", "time"]).groupby("location_id").head(horizon_days).reset_index(drop=True)
    predicted, metadata = predict_dataframe(extracted, settings, forecast_source=resolved_source.source_id)

    results: list[ForecastResultResponse] = []
    for location_id, group in predicted.groupby("location_id"):
        advisory_payload = build_advisory(group.sort_values("time"), settings.advisory)
        first = group.iloc[0]
        agro_characteristics = build_agro_characteristics(group.sort_values("time"), settings)
        calibration_version = resolve_calibration_version(resolved_source, metadata)
        results.append(
            ForecastResultResponse(
                location_id=location_id,
                lat=float(first["requested_latitude"]),
                lon=float(first["requested_longitude"]),
                spatial_resolution_km=settings.data_pipeline.target_resolution_km,
                forecast_source=resolved_source.source_id,
                data_origin=resolved_source.data_origin,
                source_run_id=resolved_source.source_run_id,
                model_version=str(metadata["model_version"]),
                calibration_version=calibration_version,
                generated_at=pd.Timestamp.now("UTC").to_pydatetime(),
                daily_forecast=serialize_daily_forecast(
                    group[[column for column in ["time", "rainfall_raw_mm", "rainfall_corrected_mm", "temp_c"] if column in group.columns]]
                ),
                agro_characteristics=agro_characteristics,
                advisory=serialize_advisory(advisory_payload),
            )
        )
    seasonal_refresh = refresh_seasonal_map_products(
        settings,
        forecast_source=resolved_source.source_id,
        resolved_source_override=resolved_source,
    )
    if seasonal_refresh["failed_count"]:
        raise SeasonalMapRefreshFailedError(
            "Forecast generation completed, but seasonal map refresh failed for "
            f"{seasonal_refresh['failed_count']} combination(s)."
        )
    return results, {
        **metadata,
        "forecast_source": resolved_source.source_id,
        "data_origin": resolved_source.data_origin,
        "source_run_id": resolved_source.source_run_id,
        "calibration_version": resolve_calibration_version(resolved_source, metadata),
        "spatial_resolution_km": settings.data_pipeline.target_resolution_km,
        "seasonal_refresh": seasonal_refresh,
    }
