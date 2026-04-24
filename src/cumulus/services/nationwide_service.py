"""Nationwide batch generation and artifact-backed serving."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cumulus.advisory.farmer_rules import build_farmer_advisory
from cumulus.api.errors import ForecastSourceNotConfiguredError, NationwideArtifactsNotAvailableError
from cumulus.data.extractors import extract_locations
from cumulus.data.location_index import load_serving_locations
from cumulus.frontend_contract.serializers import serialize_daily_forecast, serialize_farmer_advisory
from cumulus.modeling.predictor import predict_dataframe
from cumulus.schemas import PointAdvisoryResponse
from cumulus.services.agro_service import build_agro_characteristics
from cumulus.services.source_resolution import (
    normalize_forecast_source_id,
    open_source_dataset,
    resolve_calibration_version,
    resolve_forecast_source,
)
from cumulus.settings import Settings
from cumulus.utils.io import ensure_directory, read_json, write_json


logger = logging.getLogger(__name__)
ACTIVE_RUN_FILENAME = "active_run.json"


def generate_nationwide_run(
    settings: Settings,
    horizon_days: int | None = None,
    *,
    forecast_source: str | None = None,
) -> dict[str, Any]:
    """Generate a nationwide serving run from one forecast extraction pass."""

    resolved_source = resolve_forecast_source(settings, forecast_source)

    run_horizon = horizon_days or settings.forecast_horizon_days
    locations = load_serving_locations(settings.config_dir / "locations.yaml")
    if locations.empty:
        raise ValueError("No serving locations are configured in configs/locations.yaml.")

    logger.info(
        "nationwide.generate_start locations=%s horizon_days=%s forecast_path=%s",
        len(locations),
        run_horizon,
        resolved_source.path,
    )
    ds = open_source_dataset(settings, resolved_source)
    extracted = extract_locations(ds, locations[["location_id", "latitude", "longitude"]], list(ds.data_vars))
    extracted = (
        extracted.sort_values(["location_id", "time"])
        .groupby("location_id", group_keys=False)
        .head(run_horizon)
        .reset_index(drop=True)
    )
    predicted, metadata = predict_dataframe(extracted, settings, forecast_source=resolved_source.source_id)
    predicted = predicted.merge(
        locations,
        on="location_id",
        how="left",
        suffixes=("", "_meta"),
    )

    generated_at = datetime.now(UTC)
    manifest_calibration_version = resolve_calibration_version(resolved_source, metadata)
    point_records: list[dict[str, Any]] = []
    failed_locations: list[str] = []
    for location in locations.to_dict(orient="records"):
        location_id = str(location["location_id"])
        group = predicted[predicted["location_id"].astype(str) == location_id].sort_values("time").reset_index(drop=True)
        if group.empty:
            failed_locations.append(location_id)
            continue
        point_records.append(
            _build_point_record(
                group=group,
                location=location,
                source_run_id=resolved_source.source_run_id,
                model_version=str(metadata["model_version"]),
                calibration_version=resolve_calibration_version(
                    resolved_source,
                    metadata,
                    agro_ecological_zone=location.get("agro_ecological_zone"),
                ),
                model_strategy=settings.nationwide.model_strategy,
                generated_at=generated_at,
                forecast_source=resolved_source.source_id,
                data_origin=resolved_source.data_origin,
                horizon_days=run_horizon,
                settings=settings,
            )
        )

    region_summaries = _build_geography_summaries(
        point_records,
        geography_key="region",
        model_strategy=settings.nationwide.model_strategy,
        note_template="Regional summary aggregated from {count} configured locations in {name}.",
    )
    district_summaries = _build_geography_summaries(
        point_records,
        geography_key="district",
        model_strategy=settings.nationwide.model_strategy,
        note_template="District summary aggregated from {count} configured locations in {name}.",
    )

    run_id = f"nationwide_{resolved_source.source_id}_{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_directory(settings.nationwide.artifact_dir / resolved_source.source_id / "runs" / run_id)
    points_path = run_dir / "points.json"
    regions_path = run_dir / "regions.json"
    districts_path = run_dir / "districts.json"
    manifest_path = run_dir / "manifest.json"
    active_pointer_path = settings.nationwide.artifact_dir / f"active_run_{resolved_source.source_id}.json"

    write_json(points_path, point_records)
    write_json(regions_path, region_summaries)
    write_json(districts_path, district_summaries)
    manifest = {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "horizon_days": run_horizon,
        "source_run_id": resolved_source.source_run_id,
        "spatial_resolution_km": settings.data_pipeline.target_resolution_km,
        "model_version": str(metadata["model_version"]),
        "calibration_version": manifest_calibration_version,
        "model_strategy": settings.nationwide.model_strategy,
        "forecast_source": resolved_source.source_id,
        "data_origin": resolved_source.data_origin,
        "location_count": int(len(locations)),
        "available_location_count": int(len(point_records)),
        "failed_location_count": int(len(failed_locations)),
        "failed_locations": failed_locations,
        "region_count": int(len(region_summaries)),
        "district_count": int(len(district_summaries)),
        "run_dir": str(run_dir),
        "points_path": str(points_path),
        "regions_path": str(regions_path),
        "districts_path": str(districts_path),
    }
    write_json(manifest_path, manifest)
    write_json(
        active_pointer_path,
        {
            "run_id": run_id,
            "manifest_path": str(manifest_path),
        },
    )
    if _is_default_artifact_source(settings, resolved_source.source_id):
        write_json(
            settings.nationwide.artifact_dir / ACTIVE_RUN_FILENAME,
            {
                "run_id": run_id,
                "manifest_path": str(manifest_path),
            },
        )
    clear_nationwide_cache()
    logger.info(
        "nationwide.generate_success run_id=%s locations=%s failed=%s",
        run_id,
        len(point_records),
        len(failed_locations),
    )
    return manifest


def get_active_run_manifest(settings: Settings, *, forecast_source: str | None = None) -> dict[str, Any]:
    try:
        pointer = read_json(_active_run_pointer_path(settings, forecast_source))
    except FileNotFoundError as exc:
        raise NationwideArtifactsNotAvailableError(
            f"No active nationwide run is available under {settings.nationwide.artifact_dir}."
        ) from exc
    manifest_path = pointer.get("manifest_path")
    if not manifest_path:
        raise NationwideArtifactsNotAvailableError("Nationwide active run metadata is incomplete.")
    return _read_json_cached(str(Path(manifest_path).resolve()))


def list_active_locations(
    settings: Settings,
    *,
    forecast_source: str | None = None,
    page: int = 1,
    page_size: int | None = None,
    region: str | None = None,
    district: str | None = None,
) -> dict[str, Any]:
    manifest = get_active_run_manifest(settings, forecast_source=forecast_source)
    points = _load_active_points(settings, forecast_source=forecast_source)
    filtered = [
        item
        for item in points
        if _matches_filter(item.get("region"), region) and _matches_filter(item.get("district"), district)
    ]
    filtered.sort(key=lambda item: str(item.get("location_id")))
    resolved_page_size = max(1, min(page_size or settings.nationwide.default_page_size, settings.nationwide.max_page_size))
    start = max(0, (page - 1) * resolved_page_size)
    end = start + resolved_page_size
    return {
        "run_id": manifest["run_id"],
        "generated_at": manifest["generated_at"],
        "page": page,
        "page_size": resolved_page_size,
        "total_locations": len(filtered),
        "items": filtered[start:end],
    }


def get_geography_summary(
    settings: Settings,
    geography_type: str,
    geography_name: str,
    *,
    forecast_source: str | None = None,
) -> dict[str, Any]:
    manifest = get_active_run_manifest(settings, forecast_source=forecast_source)
    summaries = (
        _load_active_regions(settings, forecast_source=forecast_source)
        if geography_type == "region"
        else _load_active_districts(settings, forecast_source=forecast_source)
    )
    for summary in summaries:
        if str(summary.get("geography_name", "")).lower() == geography_name.strip().lower():
            return summary
    raise FileNotFoundError(f"No active {geography_type} summary was found for {geography_name}.")


def find_active_point_record(
    settings: Settings,
    *,
    location_id: str | None,
    latitude: float,
    longitude: float,
    forecast_source: str | None = None,
) -> dict[str, Any] | None:
    try:
        points = _load_active_points(settings, forecast_source=forecast_source)
    except NationwideArtifactsNotAvailableError:
        return None
    tolerance = settings.nationwide.known_location_tolerance_degrees
    if location_id:
        for point in points:
            if str(point.get("location_id")) == str(location_id):
                return point
    for point in points:
        if (
            abs(float(point.get("latitude", 999.0)) - latitude) <= tolerance
            and abs(float(point.get("longitude", 999.0)) - longitude) <= tolerance
        ):
            return point
    return None


def clear_nationwide_cache() -> None:
    _read_json_cached.cache_clear()


@lru_cache(maxsize=32)
def _read_json_cached(path: str) -> Any:
    return read_json(Path(path))


def _load_active_points(settings: Settings, *, forecast_source: str | None = None) -> list[dict[str, Any]]:
    manifest = get_active_run_manifest(settings, forecast_source=forecast_source)
    return _read_json_cached(str(Path(manifest["points_path"]).resolve()))


def _load_active_regions(settings: Settings, *, forecast_source: str | None = None) -> list[dict[str, Any]]:
    manifest = get_active_run_manifest(settings, forecast_source=forecast_source)
    return _read_json_cached(str(Path(manifest["regions_path"]).resolve()))


def _load_active_districts(settings: Settings, *, forecast_source: str | None = None) -> list[dict[str, Any]]:
    manifest = get_active_run_manifest(settings, forecast_source=forecast_source)
    return _read_json_cached(str(Path(manifest["districts_path"]).resolve()))


def _build_point_record(
    *,
    group: pd.DataFrame,
    location: dict[str, Any],
    source_run_id: str,
    model_version: str,
    calibration_version: str,
    model_strategy: str,
    generated_at: datetime,
    forecast_source: str,
    data_origin: str,
    horizon_days: int,
    settings: Settings,
) -> dict[str, Any]:
    agro_characteristics = build_agro_characteristics(group, settings)
    advisory_payload = build_farmer_advisory(
        group[["time", "rainfall_corrected_mm", "temp_c"]].copy(),
        settings.advisory,
    )
    advisory_payload["location_id"] = str(location["location_id"])
    serialized_farmer_advisory = serialize_farmer_advisory(advisory_payload)
    point_advisory = PointAdvisoryResponse(
        location_id=str(location["location_id"]),
        latitude=float(location["latitude"]),
        longitude=float(location["longitude"]),
        forecast_source=forecast_source,
        data_origin="nationwide_artifact_cache",
        source_run_id=source_run_id,
        spatial_resolution_km=settings.data_pipeline.target_resolution_km,
        model_version=model_version,
        calibration_version=calibration_version,
        generated_at=generated_at,
        agro_characteristics=agro_characteristics,
        planting_recommendation=serialized_farmer_advisory.planting_recommendation,
        dry_spell_alert=serialized_farmer_advisory.dry_spell_alert,
        irrigation_advice=serialized_farmer_advisory.irrigation_advice,
    )
    forecast_frame_rows = [
        {
            "time": row["time"].isoformat(),
            "rainfall_raw_mm": float(row["rainfall_raw_mm"]),
            "rainfall_corrected_mm": float(row["rainfall_corrected_mm"]),
            "temp_c": float(row.get("temp_c", 0.0)),
            "precip_mm": float(row.get("precip_mm", 0.0)),
        }
        for _, row in group.iterrows()
    ]
    return {
        "location_id": str(location["location_id"]),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
        "region": location.get("region"),
        "district": location.get("district"),
        "agro_ecological_zone": location.get("agro_ecological_zone"),
        "is_serving_location": bool(location.get("is_serving_location", True)),
        "forecast_source": forecast_source,
        "data_origin": "nationwide_artifact_cache",
        "source_data_origin": data_origin,
        "source_run_id": source_run_id,
        "spatial_resolution_km": settings.data_pipeline.target_resolution_km,
        "model_version": model_version,
        "calibration_version": calibration_version,
        "model_strategy": model_strategy,
        "generated_at": generated_at.isoformat(),
        "horizon_days": horizon_days,
        "daily_forecast": [
            item.model_dump(mode="json")
            for item in serialize_daily_forecast(
                group[[column for column in ["time", "rainfall_raw_mm", "rainfall_corrected_mm", "temp_c"] if column in group.columns]]
            )
        ],
        "forecast_frame_rows": forecast_frame_rows,
        "agro_characteristics": agro_characteristics.model_dump(mode="json"),
        "point_advisory": point_advisory.model_dump(mode="json"),
    }


def _build_geography_summaries(
    point_records: list[dict[str, Any]],
    *,
    geography_key: str,
    model_strategy: str,
    note_template: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in point_records:
        geography_name = record.get(geography_key)
        if not geography_name:
            continue
        grouped.setdefault(str(geography_name), []).append(record)

    summaries: list[dict[str, Any]] = []
    for geography_name, records in sorted(grouped.items()):
        summaries.append(
            {
                "geography_type": geography_key,
                "geography_name": geography_name,
                "generated_at": records[0]["generated_at"],
                "forecast_source": records[0]["forecast_source"],
                "data_origin": "nationwide_artifact_cache",
                "source_run_id": records[0]["source_run_id"],
                "spatial_resolution_km": records[0]["spatial_resolution_km"],
                "model_version": records[0]["model_version"],
                "calibration_version": records[0]["calibration_version"],
                "model_strategy": model_strategy,
                "horizon_days": int(records[0]["horizon_days"]),
                "location_count": len(records),
                "coverage_count": len(records),
                "daily_forecast": _aggregate_daily_forecast(records),
                "planting_recommendation": _aggregate_advisory_kind(records, "planting"),
                "dry_spell_alert": _aggregate_advisory_kind(records, "dry_spell"),
                "irrigation_advice": _aggregate_advisory_kind(records, "irrigation"),
                "note": note_template.format(name=geography_name, count=len(records)),
            }
        )
    return summaries


def _aggregate_daily_forecast(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for day in record["daily_forecast"]:
            rows.append(day)
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby("date", as_index=False)[
            [column for column in ["rainfall_raw_mm", "rainfall_corrected_mm", "temperature_c"] if column in frame.columns]
        ]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return [
        {
            "date": row["date"],
            "rainfall_raw_mm": float(round(row["rainfall_raw_mm"], 3)),
            "rainfall_corrected_mm": float(round(row["rainfall_corrected_mm"], 3)),
            "temperature_c": float(round(row["temperature_c"], 3)) if "temperature_c" in grouped.columns else None,
            "horizon_day": index + 1,
        }
        for index, (_, row) in enumerate(grouped.iterrows())
    ]


def _aggregate_advisory_kind(records: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    card_key = {
        "planting": "planting_recommendation",
        "dry_spell": "dry_spell_alert",
        "irrigation": "irrigation_advice",
    }[kind]
    cards = [record["point_advisory"][card_key] for record in records]
    buckets = [_severity_for_card(kind, card) for card in cards]
    alert_count = sum(bucket == "high" for bucket in buckets)
    watch_count = sum(bucket == "moderate" for bucket in buckets)
    highest_bucket = _highest_severity_bucket(buckets)
    source_card = cards[buckets.index(highest_bucket)] if highest_bucket in buckets else cards[0]
    headline = source_card["headline"]
    if kind == "planting":
        headline = (
            "Delay planting across part of the geography"
            if highest_bucket == "high"
            else "Planting signal is mixed"
            if highest_bucket == "moderate"
            else "Planting conditions look favorable"
        )
    elif kind == "dry_spell":
        headline = (
            "Dry spell warning present"
            if highest_bucket == "high"
            else "Dry spell watch present"
            if highest_bucket == "moderate"
            else "Dry spell risk is limited"
        )
    elif kind == "irrigation":
        headline = (
            "Irrigation support likely needed"
            if highest_bucket == "high"
            else "Monitor soil moisture"
            if highest_bucket == "moderate"
            else "No irrigation pressure"
        )
    return {
        **source_card,
        "headline": headline,
        "severity_bucket": highest_bucket,
        "available_location_count": len(records),
        "alert_count": alert_count,
        "watch_count": watch_count,
        "reason": f"Aggregate {kind.replace('_', ' ')} signal built from {len(records)} configured locations.",
    }


def _severity_for_card(kind: str, card: dict[str, Any]) -> str:
    level = str(card.get("level", "")).lower()
    if kind == "planting":
        if level == "plant_now":
            return "low"
        if "delay" in level:
            return "high"
        return "moderate"
    if kind == "dry_spell":
        if level == "warning":
            return "high"
        if level == "watch":
            return "moderate"
        return "low"
    if level == "irrigate_if_possible":
        return "high"
    if level == "monitor_soil_moisture":
        return "moderate"
    return "low"


def _highest_severity_bucket(buckets: list[str]) -> str:
    if "high" in buckets:
        return "high"
    if "moderate" in buckets:
        return "moderate"
    return "low"


def _matches_filter(value: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    if value is None:
        return False
    return str(value).strip().lower() == expected.strip().lower()


def _active_run_pointer_path(settings: Settings, forecast_source: str | None) -> Path:
    normalized_source = normalize_forecast_source_id(forecast_source) or normalize_forecast_source_id(
        settings.default_forecast_source
    )
    if normalized_source:
        source_pointer = settings.nationwide.artifact_dir / f"active_run_{normalized_source}.json"
        if source_pointer.exists():
            return source_pointer
    source_pointers = sorted(settings.nationwide.artifact_dir.glob("active_run_*.json"))
    if len(source_pointers) == 1:
        return source_pointers[0]
    legacy_pointer = settings.nationwide.artifact_dir / ACTIVE_RUN_FILENAME
    if legacy_pointer.exists():
        return legacy_pointer
    if normalized_source:
        return settings.nationwide.artifact_dir / f"active_run_{normalized_source}.json"
    return legacy_pointer


def _is_default_artifact_source(settings: Settings, forecast_source: str) -> bool:
    default_source = normalize_forecast_source_id(settings.default_forecast_source)
    if default_source is None and len(settings.forecast_sources) == 1:
        default_source = next(iter(settings.forecast_sources))
    if default_source is not None:
        configured_default = settings.forecast_sources.get(default_source)
        if configured_default is not None and configured_default.path is None and forecast_source == "configured":
            return True
    return normalize_forecast_source_id(forecast_source) == default_source
