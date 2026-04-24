"""Preflight reporting for raw forecast data and model readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cumulus.services.source_resolution import normalize_forecast_source_id, resolve_forecast_source
from cumulus.settings import Settings
from cumulus.utils.io import read_json


def build_preflight_report(settings: Settings) -> dict[str, Any]:
    requested_source = normalize_forecast_source_id(settings.default_forecast_source)
    active_source_id: str | None = requested_source
    active_forecast_path: str | None = None
    active_data_origin: str | None = None
    source_resolution_error: str | None = None

    try:
        active_source = resolve_forecast_source(settings, requested_source)
        active_source_id = active_source.source_id
        active_forecast_path = str(active_source.path)
        active_data_origin = active_source.data_origin
    except Exception as exc:  # pragma: no cover - surfaced through report payload
        source_resolution_error = str(exc)

    source_reports = {
        source_id: _build_source_report(settings, source_id, active_source_id == source_id)
        for source_id in ("era5", "gfs")
    }
    effective_point_origin = None
    if active_source_id and source_reports.get(active_source_id, {}).get("nationwide_artifacts_available"):
        effective_point_origin = "nationwide_artifact_cache"
    else:
        effective_point_origin = active_data_origin

    station_path = settings.default_station_path
    return {
        "default_forecast_source": settings.default_forecast_source,
        "active_forecast_source": active_source_id,
        "active_forecast_path": active_forecast_path,
        "active_data_origin": active_data_origin,
        "point_request_data_origin": effective_point_origin,
        "source_resolution_error": source_resolution_error,
        "station_path": str(station_path) if station_path is not None else None,
        "station_path_exists": bool(station_path and station_path.exists()),
        "data_sources": source_reports,
    }


def _build_source_report(settings: Settings, source_id: str, is_active: bool) -> dict[str, Any]:
    config = settings.forecast_sources.get(source_id)
    path = Path(config.path) if config and config.path else None
    manifest_path = Path(config.manifest_path) if config and config.manifest_path else settings.raw_data_dir / source_id / "manifest.json"
    model_registry = settings.model_artifact_dir / f"active_model_{source_id}.json"
    nationwide_pointer = settings.nationwide.artifact_dir / f"active_run_{source_id}.json"
    active_model = _load_active_model_safe(settings.model_artifact_dir, source_id)
    manifest = _read_json_safe(manifest_path)

    return {
        "is_active_source": is_active,
        "configured": config is not None,
        "path": str(path) if path is not None else None,
        "path_exists": bool(path and path.exists()),
        "engine": config.engine if config else None,
        "variables": list(config.variables) if config else [],
        "source_run_id": config.source_run_id if config else None,
        "data_origin": config.data_origin if config else "missing",
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_refresh_timestamp": manifest.get("refresh_timestamp") if isinstance(manifest, dict) else None,
        "manifest_temporal_coverage": manifest.get("temporal_coverage") if isinstance(manifest, dict) else {},
        "model_artifacts_available": active_model is not None,
        "active_model_registry_path": str(model_registry),
        "active_model_version": active_model.get("model_version") if active_model else None,
        "nationwide_artifacts_available": nationwide_pointer.exists(),
        "nationwide_pointer_path": str(nationwide_pointer),
    }


def _load_active_model_safe(registry_dir: Path, source_id: str) -> dict[str, Any] | None:
    registry_path = registry_dir / f"active_model_{source_id}.json"
    if not registry_path.exists():
        return None
    payload = read_json(registry_path)
    return payload if isinstance(payload, dict) else None


def _read_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None
