"""Training service orchestration."""

from __future__ import annotations

import pandas as pd

from cumulus.data.extractors import extract_locations, extract_station_points
from cumulus.data.loaders import open_dataset, subset_ghana_bbox
from cumulus.data.location_index import load_locations
from cumulus.data.station_data import load_station_observations
from cumulus.modeling.trainer import train_baseline_model
from cumulus.preprocessing.alignment import align_forecast_and_station_daily, build_ml_ready_dataset
from cumulus.services.source_resolution import resolve_forecast_source
from cumulus.settings import Settings


def build_aligned_daily_dataframe(
    settings: Settings,
    forecast_path: str,
    station_path: str,
    locations_path: str | None = None,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    stations = load_station_observations(station_path)
    locations = load_locations(locations_path or settings.config_dir / "locations.yaml")
    station_locations = stations.rename(columns={"station_id": "location_id"})[["location_id", "latitude", "longitude"]].drop_duplicates()
    ds = open_dataset(
        forecast_path,
        settings.rainfall_variable_aliases,
        variables=variables or ["tp", "t2m", "u10", "v10"],
        chunks=settings.data_pipeline.xarray_chunks or None,
    )
    bounded = subset_ghana_bbox(ds, settings.data_pipeline.ghana_bounds.model_dump())
    requested_variables = [name for name in ["precip_mm", "temp_c", "u10", "v10"] if name in bounded.data_vars]
    extracted = extract_station_points(bounded, station_locations.rename(columns={"location_id": "station_id"}), requested_variables)
    if not locations.empty:
        explicit_points = locations[~locations["location_id"].astype(str).isin(extracted["location_id"].astype(str))]
        explicit_points = extract_locations(bounded, explicit_points, requested_variables) if not explicit_points.empty else pd.DataFrame()
    else:
        explicit_points = pd.DataFrame()
    if not explicit_points.empty:
        explicit_points["station_id"] = pd.NA
        extracted = pd.concat([extracted, explicit_points], ignore_index=True, sort=False)
    extracted["location_id"] = extracted["location_id"].astype(str)
    stations["location_id"] = stations["station_id"].astype(str)
    return align_forecast_and_station_daily(
        extracted,
        stations,
        location_key="location_id",
        daily_frequency=settings.data_pipeline.daily_frequency,
    )


def build_ml_ready_dataframe(
    settings: Settings,
    forecast_path: str,
    station_path: str,
    locations_path: str | None = None,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    aligned = build_aligned_daily_dataframe(
        settings,
        forecast_path=forecast_path,
        station_path=station_path,
        locations_path=locations_path,
        variables=variables,
    )
    return build_ml_ready_dataset(
        aligned,
        target_column=settings.data_pipeline.station_target_column,
        location_key="location_id",
    )


def build_merged_training_dataframe(
    settings: Settings,
    forecast_path: str,
    station_path: str,
    locations_path: str | None = None,
) -> pd.DataFrame:
    return build_ml_ready_dataframe(
        settings,
        forecast_path=forecast_path,
        station_path=station_path,
        locations_path=locations_path,
    )


def train_from_inputs(
    settings: Settings,
    merged_dataset_path: str | None = None,
    station_path: str | None = None,
    forecast_path: str | None = None,
    forecast_source: str | None = None,
) -> dict[str, object]:
    if merged_dataset_path:
        merged = _read_merged_dataset(merged_dataset_path)
        merged["time"] = pd.to_datetime(merged["time"], utc=True)
    else:
        resolved_source = None
        if forecast_path is None:
            resolved_source = resolve_forecast_source(settings, forecast_source or settings.default_forecast_source)
            forecast_path = str(resolved_source.path)
            forecast_source = resolved_source.source_id
        elif forecast_source is None:
            forecast_source = settings.default_forecast_source
        if station_path is None:
            if settings.default_station_path is None:
                raise ValueError(
                    "Training requires station observations. Set --station-path or configure CUMULUS_DEFAULT_STATION_PATH."
                )
            station_path = str(settings.default_station_path)
        if not pd.io.common.file_exists(station_path):
            raise ValueError(f"Station observations were not found: {station_path}")
        if not forecast_path or not station_path:
            raise ValueError(
                "Training requires either merged_dataset_path or a resolved forecast source plus station observations."
            )
        merged = build_merged_training_dataframe(settings, forecast_path=forecast_path, station_path=station_path)
    return train_baseline_model(merged, settings, forecast_source=forecast_source or settings.default_forecast_source)


def _read_merged_dataset(path: str) -> pd.DataFrame:
    lower_path = str(path).lower()
    if lower_path.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)
