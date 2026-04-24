"""Helpers for raw forecast source manifests and conventional data layout."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from cumulus.utils.io import read_json, write_json


RAW_SOURCE_DIRS = ("era5", "gfs", "stations")
DEFAULT_SOURCE_VARIABLES = {
    "era5": ["tp", "u10", "v10", "t2m"],
    "gfs": ["tp", "u10", "v10", "t2m"],
}
SAMPLE_FILE_HINTS = ("sample_forecast_smoke", "sample_forecast", "smoke")


def ensure_raw_data_layout(raw_data_dir: Path) -> None:
    for source_id in RAW_SOURCE_DIRS:
        (raw_data_dir / source_id).mkdir(parents=True, exist_ok=True)


def source_manifest_path(raw_data_dir: Path, source_id: str) -> Path:
    return raw_data_dir / _normalize_source_id(source_id) / "manifest.json"


def load_source_manifest(raw_data_dir: Path, source_id: str) -> dict[str, Any] | None:
    manifest_path = source_manifest_path(raw_data_dir, source_id)
    if not manifest_path.exists():
        return None
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in source manifest: {manifest_path}")
    return payload


def resolve_manifest_dataset_path(raw_data_dir: Path, source_id: str, manifest: dict[str, Any]) -> Path | None:
    manifest_path = source_manifest_path(raw_data_dir, source_id)
    candidate = (
        manifest.get("dataset_path")
        or manifest.get("primary_dataset_path")
        or manifest.get("combined_dataset_path")
    )
    if isinstance(candidate, str) and candidate.strip():
        return _resolve_manifest_value_path(manifest_path.parent, candidate)

    dataset_paths = manifest.get("dataset_paths")
    if isinstance(dataset_paths, list):
        for item in dataset_paths:
            if isinstance(item, str) and item.strip():
                return _resolve_manifest_value_path(manifest_path.parent, item)
    return None


def resolve_manifest_dataset_paths(raw_data_dir: Path, source_id: str, manifest: dict[str, Any]) -> list[Path]:
    manifest_path = source_manifest_path(raw_data_dir, source_id)
    resolved: list[Path] = []
    for key in ("dataset_path", "primary_dataset_path", "combined_dataset_path"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            resolved.append(_resolve_manifest_value_path(manifest_path.parent, value))
    dataset_paths = manifest.get("dataset_paths")
    if isinstance(dataset_paths, list):
        for item in dataset_paths:
            if isinstance(item, str) and item.strip():
                resolved.append(_resolve_manifest_value_path(manifest_path.parent, item))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def infer_dataset_engine(path: str | Path | None) -> str | None:
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    if suffix in {".grib", ".grb", ".grb2"}:
        return "cfgrib"
    if suffix in {".nc", ".nc4", ".cdf"}:
        return "scipy"
    return None


def classify_data_origin(path: str | Path | None) -> str:
    if path is None:
        return "missing"
    lowered = str(path).lower()
    if any(hint in lowered for hint in SAMPLE_FILE_HINTS):
        return "fallback_sample_data"
    return "downloaded_real_source_data"


def discover_default_station_path(
    raw_data_dir: Path,
    project_root: Path,
    configured_path: str | Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    configured = Path(configured_path) if configured_path is not None else None
    if configured is not None and (configured.is_absolute() or len(configured.parts) > 1):
        candidates.append(configured)
    candidates.extend(
        [
            raw_data_dir / "stations" / "Rainfall_data.xlsx",
            project_root / "Rainfall_data.xlsx",
        ]
    )
    if configured is not None and configured not in candidates:
        candidates.append(configured)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def build_source_manifest(
    *,
    source_id: str,
    dataset_paths: list[Path],
    manifest_dir: Path,
    variables: list[str],
    engine: str | None,
    source_metadata: dict[str, Any],
    temporal_coverage: dict[str, Any] | None = None,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    normalized_source = _normalize_source_id(source_id)
    primary_path = dataset_paths[0] if dataset_paths else None
    refreshed_at = datetime.now(UTC).isoformat()
    return {
        "source_id": normalized_source,
        "dataset_path": _to_manifest_relative_path(manifest_dir, primary_path) if primary_path else None,
        "dataset_paths": [_to_manifest_relative_path(manifest_dir, path) for path in dataset_paths],
        "engine": engine or infer_dataset_engine(primary_path),
        "variables": variables or DEFAULT_SOURCE_VARIABLES.get(normalized_source, []),
        "temporal_coverage": temporal_coverage or {},
        "source_metadata": source_metadata,
        "refresh_timestamp": refreshed_at,
        "source_run_id": source_run_id or f"{normalized_source}-{refreshed_at.replace(':', '').replace('-', '')}",
    }


def write_source_manifest(raw_data_dir: Path, source_id: str, manifest: dict[str, Any]) -> Path:
    path = source_manifest_path(raw_data_dir, source_id)
    write_json(path, manifest)
    return path


def inspect_temporal_coverage(path: Path, *, engine: str | None = None) -> dict[str, Any]:
    dataset = xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
    try:
        if "time" not in dataset.coords:
            return {}
        time_index = pd.to_datetime(dataset["time"].values, utc=True)
        if len(time_index) == 0:
            return {}
        coverage: dict[str, Any] = {
            "start": time_index.min().isoformat(),
            "end": time_index.max().isoformat(),
            "count": int(len(time_index)),
        }
        if len(time_index) > 1:
            coverage["step_hours"] = float((time_index[1] - time_index[0]).total_seconds() / 3600.0)
        return coverage
    finally:
        dataset.close()


def discover_source_config(raw_data_dir: Path, source_id: str) -> dict[str, Any]:
    normalized_source = _normalize_source_id(source_id)
    manifest = load_source_manifest(raw_data_dir, normalized_source)
    if manifest:
        path = resolve_manifest_dataset_path(raw_data_dir, normalized_source, manifest)
        if path is not None:
            return {
                "path": path,
                "engine": manifest.get("engine") or infer_dataset_engine(path),
                "variables": manifest.get("variables") or DEFAULT_SOURCE_VARIABLES.get(normalized_source, []),
                "source_run_id": manifest.get("source_run_id"),
                "manifest_path": source_manifest_path(raw_data_dir, normalized_source),
                "data_origin": classify_data_origin(path),
            }

    source_dir = raw_data_dir / normalized_source
    candidates = sorted(
        [
            *source_dir.glob("*.nc"),
            *source_dir.glob("*.nc4"),
            *source_dir.glob("*.grib"),
            *source_dir.glob("*.grb"),
            *source_dir.glob("*.grb2"),
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}
    path = candidates[0]
    return {
        "path": path,
        "engine": infer_dataset_engine(path),
        "variables": DEFAULT_SOURCE_VARIABLES.get(normalized_source, []),
        "source_run_id": f"{normalized_source}-{path.stem}",
        "manifest_path": None,
        "data_origin": classify_data_origin(path),
    }


def _normalize_source_id(source_id: str) -> str:
    return str(source_id).strip().lower().replace(" ", "_")


def _resolve_manifest_value_path(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_dir / path).resolve()


def _to_manifest_relative_path(manifest_dir: Path, path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(manifest_dir.resolve()))
    except ValueError:
        return str(resolved_path)
