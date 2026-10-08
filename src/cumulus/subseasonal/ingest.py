"""Turn a raw IFS-UNet run (one NetCDF4 file per lead day) into a compact serving artifact.

Usage::

    python -m cumulus.subseasonal.ingest --source local --path ../Unet --latest
    python -m cumulus.subseasonal.ingest --source azure --latest      # needs CUMULUS_SUBSEASONAL__AZURE_SAS_URL

The raw files are HDF5-based NetCDF4, Africa-wide, uncompressed float32 (~53 MB per run) and
carry a few tiny negative values from the UNet. The artifact written here is:

* clipped at 0 mm and subset to Ghana + 0.5 degrees, latitude ascending;
* NetCDF3 classic (readable by the scipy engine already used in production, no HDF5 libs
  needed at serve time), int16 packed at 0.1 mm (~0.45 MB per run);
* described by a ``manifest.json`` and promoted through ``<source>/active.json``.

Ingest runs offline (locally or in CI). The API only ever reads artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import xarray as xr

from cumulus.settings import Settings, get_settings
from cumulus.subseasonal import geomask
from cumulus.subseasonal.sources import AzureBlobSasSource, LocalFolderSource, RunSource, SourceError

ARTIFACT_FILE_NAME = "rainfall.nc"
MANIFEST_FILE_NAME = "manifest.json"
ACTIVE_FILE_NAME = "active.json"
PACK_SCALE = 0.1


class IngestError(RuntimeError):
    """Raised when a run is incomplete or inconsistent and must not be published."""


@dataclass(frozen=True)
class IngestResult:
    run_id: str
    init_time: datetime
    lead_days: int
    artifact_path: Path
    manifest_path: Path
    promoted: bool
    removed_runs: tuple[str, ...]


def run_id_for(source_id: str, init_time: datetime) -> str:
    return f"{source_id}_{init_time.strftime('%Y%m%d%H')}"


def source_dir(settings: Settings) -> Path:
    return Path(settings.subseasonal.artifact_dir) / settings.subseasonal.source_id


def build_source(settings: Settings, kind: str, path: Path | None) -> RunSource:
    config = settings.subseasonal
    if kind == "auto":
        kind = "azure" if config.azure_sas_url else "local"
    if kind == "azure":
        sas_url = config.azure_sas_url or os.environ.get("IFS_UNET_SAS_URL")
        if not sas_url:
            raise SourceError("Set CUMULUS_SUBSEASONAL__AZURE_SAS_URL (container SAS URL with Read + List).")
        return AzureBlobSasSource(sas_url, prefix=config.azure_prefix)
    root = path or config.local_source_dir
    if root is None:
        raise SourceError("Pass --path or set CUMULUS_SUBSEASONAL__LOCAL_SOURCE_DIR for a local ingest.")
    return LocalFolderSource(Path(root))


def ingest_run(
    settings: Settings,
    source: RunSource,
    init_date: date | None = None,
    *,
    promote: bool = True,
) -> IngestResult:
    runs = source.list_runs()
    if not runs:
        raise IngestError(f"No runs found at {source.label}.")
    target_date = init_date or runs[-1]
    if target_date not in runs:
        raise IngestError(f"Run {target_date.isoformat()} not found at {source.label}.")

    raw_cache = Path(settings.subseasonal.raw_cache_dir or Path(settings.raw_data_dir) / settings.subseasonal.source_id)
    paths = source.fetch_run(target_date, raw_cache)
    if not paths:
        raise IngestError(f"Run {target_date.isoformat()} has no lead files.")

    dataset, details = load_raw_run(paths, settings)
    run_id = run_id_for(settings.subseasonal.source_id, details["init_time"])
    out_dir = source_dir(settings) / run_id
    artifact_path, manifest_path = write_artifact(dataset, details, out_dir, settings, run_id=run_id, source_label=source.label)

    removed: tuple[str, ...] = ()
    promoted = False
    if promote:
        promoted = promote_run(settings, run_id)
        removed = prune_runs(settings)
    # Warm the district raster cache so a fresh deploy never has to build it.
    geomask.load_area_index(
        Path(settings.seasonal_map.district_geojson_path),
        dataset["latitude"].values,
        dataset["longitude"].values,
        Path(settings.subseasonal.artifact_dir),
    )
    return IngestResult(
        run_id=run_id,
        init_time=details["init_time"],
        lead_days=int(dataset.sizes["lead_day"]),
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        promoted=promoted,
        removed_runs=removed,
    )


def load_raw_run(paths: list[Path], settings: Settings) -> tuple[xr.Dataset, dict[str, Any]]:
    """Open, validate and stack per-lead files into a clipped Ghana (lead_day, lat, lon) dataset."""
    try:
        import h5netcdf  # noqa: F401
        import h5py  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise IngestError("Reading raw IFS-UNet files needs h5netcdf and h5py: pip install -e .[ingest]") from exc

    config = settings.subseasonal
    bounds = config.bounds
    fields: list[np.ndarray] = []
    leads: list[int] = []
    init_times: set[datetime] = set()
    reference_axes: tuple[np.ndarray, np.ndarray] | None = None
    checksums: list[dict[str, Any]] = []
    variable_name: str | None = None

    for path in sorted(paths):
        with xr.open_dataset(path, engine="h5netcdf") as raw:
            name = next((alias for alias in config.variable_aliases if alias in raw.data_vars), None)
            if name is None:
                raise IngestError(f"{path.name}: none of {config.variable_aliases} found (has {list(raw.data_vars)}).")
            variable_name = variable_name or name
            raw = _normalize_coords(raw)
            init_time = _init_time(raw, path)
            lead_hours = _lead_hours(raw, init_time, path)
            field = raw[name]
            if "time" in field.dims:
                if field.sizes["time"] != 1:
                    raise IngestError(f"{path.name}: expected a single time step, found {field.sizes['time']}.")
                field = field.isel(time=0)
            extra_dims = set(field.dims) - {"latitude", "longitude"}
            if extra_dims:
                raise IngestError(f"{path.name}: unexpected dimensions {sorted(extra_dims)} on {name}.")
            field = field.sortby("latitude").sortby("longitude")
            field = field.sel(
                latitude=slice(bounds.latitude_min - 1e-6, bounds.latitude_max + 1e-6),
                longitude=slice(bounds.longitude_min - 1e-6, bounds.longitude_max + 1e-6),
            )
            if field.sizes["latitude"] < 2 or field.sizes["longitude"] < 2:
                raise IngestError(f"{path.name}: the file does not cover the Ghana window.")
            axes = (np.round(field["latitude"].values.astype(float), 4), np.round(field["longitude"].values.astype(float), 4))
            if reference_axes is None:
                reference_axes = axes
            elif not (np.array_equal(axes[0], reference_axes[0]) and np.array_equal(axes[1], reference_axes[1])):
                raise IngestError(f"{path.name}: grid differs from the other lead files.")
            fields.append(np.asarray(field.values, dtype=np.float32))
            leads.append(lead_hours)
            init_times.add(init_time)
            checksums.append({"file": path.name, "sha1": _sha1(path)})

    if len(init_times) != 1:
        raise IngestError(f"Lead files mix init times: {sorted(t.isoformat() for t in init_times)}.")
    if any(hours % 24 for hours in leads):
        raise IngestError(f"Lead times must be whole days; got hours {leads}.")
    lead_days = [hours // 24 for hours in leads]
    if len(set(lead_days)) != len(lead_days):
        raise IngestError("Duplicate lead days in run.")
    expected = list(range(1, len(lead_days) + 1))
    if lead_days != expected:
        missing = sorted(set(range(1, max(lead_days) + 1)) - set(lead_days))
        raise IngestError(f"Lead days must be contiguous from day 1; missing {missing[:10]}{'...' if len(missing) > 10 else ''}.")
    if len(lead_days) < config.min_lead_days:
        raise IngestError(f"Run has {len(lead_days)} lead days; at least {config.min_lead_days} are required.")

    stacked = np.stack(fields, axis=0)
    negative_fraction = float(np.mean(stacked < 0))
    stacked = np.clip(stacked, 0.0, None)
    # Round to the packing precision now so stats and the stored artifact agree exactly.
    stacked = np.round(stacked / PACK_SCALE) * PACK_SCALE
    assert reference_axes is not None
    init_time = next(iter(init_times))
    dataset = xr.Dataset(
        {"precip": (("lead_day", "latitude", "longitude"), stacked.astype(np.float32))},
        coords={
            "lead_day": np.asarray(lead_days, dtype=np.int32),
            "latitude": reference_axes[0].astype(np.float32),
            "longitude": reference_axes[1].astype(np.float32),
        },
    )
    details = {
        "init_time": init_time,
        "variable": variable_name,
        "negative_fraction_clipped": round(negative_fraction, 4),
        "max_mm": round(float(np.nanmax(stacked)), 1),
        "checksums": checksums,
        "expected_lead_days": 46,
    }
    return dataset, details


def write_artifact(
    dataset: xr.Dataset,
    details: dict[str, Any],
    out_dir: Path,
    settings: Settings,
    *,
    run_id: str,
    source_label: str,
) -> tuple[Path, Path]:
    config = settings.subseasonal
    init_time: datetime = details["init_time"]
    lead_days = [int(day) for day in dataset["lead_day"].values]
    out = dataset.copy()
    out["precip"].attrs.update(
        {
            "long_name": "24-hour accumulated precipitation",
            "units": "mm",
            "cell_methods": "time: sum (interval: 24 hours)",
            "comment": "Lead day d accumulates from init + (d-1)*24h to init + d*24h. Negative UNet values clipped to 0.",
        }
    )
    out["lead_day"].attrs.update({"long_name": "forecast lead day", "units": "1"})
    out["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    out["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})
    out.attrs.update(
        {
            "title": f"{config.model_label} sub-seasonal daily rainfall for Ghana",
            "source": config.source_label,
            "init_time": init_time.isoformat(),
            "run_id": run_id,
            "Conventions": "CF-1.8",
        }
    )
    encoding = {"precip": {"dtype": "int16", "scale_factor": PACK_SCALE, "add_offset": 0.0, "_FillValue": np.int16(-32768)}}

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{run_id}-"))
    try:
        artifact_path = staging / ARTIFACT_FILE_NAME
        out.to_netcdf(artifact_path, engine="scipy", format="NETCDF3_64BIT", encoding=encoding)
        latitudes = dataset["latitude"].values.astype(float)
        longitudes = dataset["longitude"].values.astype(float)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "source_id": config.source_id,
            "source_label": config.source_label,
            "model_label": config.model_label,
            "ensemble": "deterministic",
            "variable": "precip",
            "raw_variable": details["variable"],
            "unit": "mm",
            "init_time": init_time.isoformat(),
            "lead_days": lead_days,
            "expected_lead_days": details["expected_lead_days"],
            "first_rain_day": init_time.date().isoformat(),
            "last_rain_day": (init_time + timedelta(days=lead_days[-1] - 1)).date().isoformat(),
            "grid": {
                "shape": [len(latitudes), len(longitudes)],
                "resolution_degrees": round(float(np.median(np.diff(latitudes))), 4),
            },
            "bounds": {
                "latitude_min": float(latitudes.min()),
                "latitude_max": float(latitudes.max()),
                "longitude_min": float(longitudes.min()),
                "longitude_max": float(longitudes.max()),
            },
            "quality": {
                "negative_fraction_clipped": details["negative_fraction_clipped"],
                "max_mm": details["max_mm"],
            },
            "artifact": ARTIFACT_FILE_NAME,
            "ingested_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "ingested_from": source_label,
            "source_files": details["checksums"],
        }
        (staging / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(staging, out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out_dir / ARTIFACT_FILE_NAME, out_dir / MANIFEST_FILE_NAME


def list_run_ids(settings: Settings) -> list[str]:
    root = source_dir(settings)
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir() and (child / MANIFEST_FILE_NAME).exists())


def promote_run(settings: Settings, run_id: str) -> bool:
    """Point active.json at ``run_id`` unless a newer run is already active."""
    active_path = source_dir(settings) / ACTIVE_FILE_NAME
    current = None
    if active_path.exists():
        try:
            current = json.loads(active_path.read_text(encoding="utf-8")).get("run_id")
        except (OSError, json.JSONDecodeError):
            current = None
    if current and current > run_id and current in list_run_ids(settings):
        return False
    payload = {"run_id": run_id, "promoted_at": datetime.now(UTC).replace(microsecond=0).isoformat()}
    temporary = active_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, active_path)
    return True


def prune_runs(settings: Settings) -> tuple[str, ...]:
    keep = max(int(settings.subseasonal.retention_runs), 1)
    run_ids = list_run_ids(settings)
    active_path = source_dir(settings) / ACTIVE_FILE_NAME
    active = json.loads(active_path.read_text(encoding="utf-8")).get("run_id") if active_path.exists() else None
    removable = [run_id for run_id in run_ids[:-keep] if run_id != active]
    for run_id in removable:
        shutil.rmtree(source_dir(settings) / run_id, ignore_errors=True)
    return tuple(removable)


def _normalize_coords(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    for name in dataset.dims:
        lowered = str(name).lower()
        if lowered in {"lat", "y"}:
            rename[name] = "latitude"
        elif lowered in {"lon", "x"}:
            rename[name] = "longitude"
    return dataset.rename(rename) if rename else dataset


def _init_time(dataset: xr.Dataset, path: Path) -> datetime:
    if "init_time" in dataset.variables:
        value = np.asarray(dataset["init_time"].values).reshape(-1)[0]
        return _to_datetime(value)
    from cumulus.subseasonal.sources import parse_lead_file

    parsed = parse_lead_file(path.name)
    if parsed is None:
        raise IngestError(f"{path.name}: no init_time variable and the file name does not encode one.")
    return datetime(parsed.init_date.year, parsed.init_date.month, parsed.init_date.day, parsed.init_hour, tzinfo=UTC)


def _lead_hours(dataset: xr.Dataset, init_time: datetime, path: Path) -> int:
    if "lead_time" in dataset.variables:
        value = np.asarray(dataset["lead_time"].values).reshape(-1)[0]
        if isinstance(value, np.timedelta64):
            return int(value / np.timedelta64(1, "h"))
        return int(round(float(value) * 24))  # numeric lead_time is expressed in days upstream
    if "time" in dataset.variables:
        valid = _to_datetime(np.asarray(dataset["time"].values).reshape(-1)[0])
        return int(round((valid - init_time).total_seconds() / 3600))
    from cumulus.subseasonal.sources import parse_lead_file

    parsed = parse_lead_file(path.name)
    if parsed is None:
        raise IngestError(f"{path.name}: cannot determine lead time.")
    return parsed.lead_hours


def _to_datetime(value: Any) -> datetime:
    stamp = np.datetime64(value, "s").astype(datetime)
    return stamp.replace(tzinfo=UTC)


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest an IFS-UNet run into a compact Ghana artifact.")
    parser.add_argument("--source", choices=["auto", "local", "azure"], default="auto")
    parser.add_argument("--path", type=Path, help="Local folder holding YYYY-MM-DD run folders.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-date", type=date.fromisoformat, help="Run to ingest (YYYY-MM-DD). Default: latest.")
    group.add_argument("--latest", action="store_true", help="Ingest the newest run (default).")
    parser.add_argument("--list", action="store_true", help="List available runs and exit.")
    parser.add_argument("--no-promote", action="store_true", help="Write the artifact without making it active.")
    args = parser.parse_args(argv)

    settings = get_settings()
    try:
        source = build_source(settings, args.source, args.path)
        if args.list:
            for run in source.list_runs():
                print(run.isoformat())
            return 0
        result = ingest_run(settings, source, args.init_date, promote=not args.no_promote)
    except (SourceError, IngestError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    size_kb = result.artifact_path.stat().st_size / 1024
    print(
        f"{result.run_id}: {result.lead_days} lead days -> {result.artifact_path} ({size_kb:.0f} KB)"
        f"{' [active]' if result.promoted else ''}"
        f"{' pruned ' + ', '.join(result.removed_runs) if result.removed_runs else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
