"""Forecast source resolution and source-specific preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from cumulus.api.errors import ForecastSourceNotConfiguredError
from cumulus.data.source_manifests import classify_data_origin
from cumulus.data.loaders import open_dataset, subset_ghana_bbox
from cumulus.settings import Settings


@dataclass(frozen=True)
class ResolvedForecastSource:
    source_id: str
    path: Path
    engine: str | None
    variables: list[str]
    variable_aliases: dict[str, str]
    source_run_id: str
    national_calibration_version: str
    regional_calibration_versions: dict[str, str]
    manifest_path: Path | None
    data_origin: str


def normalize_forecast_source_id(source_id: str | None) -> str | None:
    if source_id is None:
        return None
    return str(source_id).strip().lower().replace(" ", "_")


def resolve_forecast_source(
    settings: Settings,
    requested_source: str | None = None,
    *,
    override_path: str | Path | None = None,
    override_engine: str | None = None,
    override_variables: list[str] | None = None,
    override_source_run_id: str | None = None,
) -> ResolvedForecastSource:
    source_id = normalize_forecast_source_id(requested_source) or _default_source_id(settings)
    configured = dict(settings.forecast_sources.get(source_id, {})) if source_id else {}
    if not configured and override_path is None and source_id is not None and source_id != "configured":
        raise ForecastSourceNotConfiguredError(
            f"Forecast source '{source_id}' is not configured. Configure an ERA5 or GFS source path first."
        )

    raw_source_path = override_path or configured.get("path")
    if raw_source_path is None and requested_source is None and settings.upstream_forecast_path is not None:
        source_id = "configured"
        configured = {}
        raw_source_path = settings.upstream_forecast_path
        override_engine = override_engine or settings.upstream_forecast_engine
        override_variables = override_variables or settings.default_forecast_variables
    if raw_source_path is None:
        raise ForecastSourceNotConfiguredError(
            "No upstream forecast source is configured. Set an ERA5/GFS source or CUMULUS_UPSTREAM_FORECAST_PATH."
        )
    source_path = Path(raw_source_path)

    variables = list(override_variables or configured.get("variables") or settings.default_forecast_variables)
    variable_aliases = {
        **settings.rainfall_variable_aliases,
        **dict(configured.get("variable_aliases") or {}),
    }
    source_run_id = str(
        override_source_run_id
        or configured.get("source_run_id")
        or f"{source_id or 'configured'}-{source_path.stem}"
    )
    return ResolvedForecastSource(
        source_id=source_id or "configured",
        path=source_path,
        engine=override_engine or configured.get("engine"),
        variables=variables,
        variable_aliases=variable_aliases,
        source_run_id=source_run_id,
        national_calibration_version=str(configured.get("national_calibration_version") or "national_backbone_v1"),
        regional_calibration_versions={
            str(key): str(value)
            for key, value in dict(configured.get("regional_calibration_versions") or {}).items()
        },
        manifest_path=Path(configured["manifest_path"]) if configured.get("manifest_path") else None,
        data_origin=str(configured.get("data_origin") or classify_data_origin(source_path)),
    )


def open_source_dataset(settings: Settings, source: ResolvedForecastSource) -> xr.Dataset:
    dataset = open_dataset(
        source.path,
        variable_aliases=source.variable_aliases,
        variables=source.variables,
        chunks=settings.data_pipeline.xarray_chunks,
        engine=source.engine,
    )
    return preprocess_source_dataset(settings, source, dataset)


def preprocess_source_dataset(settings: Settings, source: ResolvedForecastSource, dataset: xr.Dataset) -> xr.Dataset:
    if source.source_id == "era5":
        return _prepare_era5_dataset(settings, dataset)
    if source.source_id == "gfs":
        return _prepare_gfs_dataset(settings, dataset)
    return _prepare_generic_dataset(settings, dataset)


def resolve_calibration_version(
    source: ResolvedForecastSource,
    metadata: dict[str, Any],
    agro_ecological_zone: str | None = None,
) -> str:
    if agro_ecological_zone:
        for zone_name, version in source.regional_calibration_versions.items():
            if zone_name.strip().lower() == agro_ecological_zone.strip().lower():
                return version
    bias_method = metadata.get("selected_bias_method") or metadata.get("bias_method")
    if bias_method:
        return f"{source.national_calibration_version}:{bias_method}"
    return source.national_calibration_version


def _prepare_era5_dataset(settings: Settings, dataset: xr.Dataset) -> xr.Dataset:
    bounded = subset_ghana_bbox(dataset, settings.data_pipeline.ghana_bounds.model_dump(), list(dataset.data_vars))
    return bounded.sortby("time")


def _prepare_gfs_dataset(settings: Settings, dataset: xr.Dataset) -> xr.Dataset:
    bounded = subset_ghana_bbox(dataset, settings.data_pipeline.ghana_bounds.model_dump(), list(dataset.data_vars))
    return bounded.sortby("time")


def _prepare_generic_dataset(settings: Settings, dataset: xr.Dataset) -> xr.Dataset:
    return subset_ghana_bbox(dataset, settings.data_pipeline.ghana_bounds.model_dump(), list(dataset.data_vars))


def _default_source_id(settings: Settings) -> str | None:
    default_source = normalize_forecast_source_id(settings.default_forecast_source)
    if (
        default_source
        and default_source in settings.forecast_sources
        and settings.forecast_sources[default_source].path is not None
    ):
        return default_source
    for source_id, config in settings.forecast_sources.items():
        if config.path is not None:
            return source_id
    if settings.upstream_forecast_path is not None:
        return "configured"
    return default_source
