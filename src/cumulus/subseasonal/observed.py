"""Observed daily rainfall (IMERG late run) for the elapsed days of a forecast.

A forecast is often read a week or two after it was issued. Its first days have then already
happened, and showing what the model *predicted* for them as if it were still uncertain is
misleading. Wherever observations exist, those days are replaced by what was observed, in
every ensemble member, so everything derived from them agrees: elapsed days become certain (0 or
100 %), a dry spell that began in observed days and runs on into the forecast is recognised as
one spell, and onset can be found in rain that has already fallen.

Upstream preprocesses the half-hourly IMERG late-run files into daily 00-24 UTC totals on a
0.25 degree grid (Azure container ``italynorthdata/data``, folder ``imerg-lr/processed/0p25/data``).
``python -m cumulus.subseasonal.observed`` keeps a rolling, Ghana-sized artifact of them::

    data/artifacts/observed/imerg/imerg_daily.nc     (time, latitude, longitude) int16 at 0.1 mm
    data/artifacts/observed/imerg/manifest.json      dates, coverage, source files

The IMERG day D (00-24 UTC) is the forecast's rain day D (lead day ``D - init + 1``; see
``metrics.rain_day``), so the two line up without any shift.

Upstream's file naming and variable name are not confirmed yet: any ``.nc`` file is read, the
variable is the first of ``variable_aliases`` found, and the date comes from the ``time``
coordinate (its start bound when there is one) or else a date in the file name.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import xarray as xr

from cumulus.settings import ObservedRainConfig, Settings, SubseasonalBoundsConfig, get_settings

ARTIFACT_FILE_NAME = "imerg_daily.nc"
MANIFEST_FILE_NAME = "manifest.json"
PACK_SCALE = 0.1
# Observations are kept for a wider window than the forecast grid, so interpolation onto the
# forecast's edge cells always has neighbours.
MARGIN_DEGREES = 0.5
# The part of the window whose coverage decides whether a day counts as observed (Ghana itself).
GHANA_BOX = {"latitude": (4.5, 11.25), "longitude": (-3.3, 1.25)}
DATE_IN_NAME = re.compile(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")


class ObservedError(RuntimeError):
    """Raised when observed rainfall cannot be read or ingested."""


# --------------------------------------------------------------------------- serving


@dataclass(frozen=True)
class ObservedDaily:
    """The observed-rain artifact: daily totals on its own (coarser) grid."""

    dates: tuple[date, ...]
    latitudes: np.ndarray  # ascending
    longitudes: np.ndarray  # ascending
    values: np.ndarray  # (time, lat, lon) mm, NaN where missing
    coverage: dict[date, float]
    source_label: str
    version: str
    synthetic: bool = False

    @property
    def last_date(self) -> date | None:
        return self.dates[-1] if self.dates else None


@dataclass(frozen=True)
class RunObservations:
    """Observations lined up with one forecast run's days and grid.

    ``values`` holds the run's leading observed days (lead day 1 onwards, contiguous): elapsed
    days the forecast is overwritten with. ``lookback`` holds observed days just before the run's
    first day, used only to tell whether a spell was already under way when the run started.
    """

    values: np.ndarray  # (observed_days, lat, lon), NaN where a cell has no value
    lookback: np.ndarray | None  # (lookback_days, lat, lon) or None
    observed_through: date | None
    source_label: str
    version: str
    synthetic: bool = False

    @property
    def days(self) -> int:
        return int(self.values.shape[0])


def version(config: ObservedRainConfig) -> str:
    """Identifies the observed artifact's content, for cache keys; "" when there is none."""
    if not config.enabled:
        return ""
    manifest_path = Path(config.artifact_dir) / MANIFEST_FILE_NAME
    try:
        return f"{manifest_path.stat().st_mtime_ns}"
    except OSError:
        return ""


def load(config: ObservedRainConfig) -> ObservedDaily | None:
    current = version(config)
    if not current:
        return None
    return _load_cached(str(config.artifact_dir), current)


@lru_cache(maxsize=2)
def _load_cached(artifact_dir: str, current_version: str) -> ObservedDaily | None:
    directory = Path(artifact_dir)
    try:
        manifest = json.loads((directory / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
        with xr.open_dataset(directory / manifest.get("artifact", ARTIFACT_FILE_NAME), engine="scipy") as dataset:
            dataset = dataset.load()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    dates = tuple(_to_date(value) for value in dataset["time"].values)
    coverage = {date.fromisoformat(key): float(value) for key, value in (manifest.get("coverage") or {}).items()}
    return ObservedDaily(
        dates=dates,
        latitudes=np.asarray(dataset["latitude"].values, dtype=float),
        longitudes=np.asarray(dataset["longitude"].values, dtype=float),
        values=np.asarray(dataset["precip"].values, dtype=np.float32),
        coverage=coverage,
        source_label=str(manifest.get("source_label") or "IMERG late run"),
        version=f"{manifest.get('last_date')}:{manifest.get('sha1', '')[:12]}:{current_version}",
        synthetic=bool(manifest.get("synthetic")),
    )


def for_run(
    config: ObservedRainConfig,
    first_day: date,
    day_count: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> RunObservations | None:
    """The run's elapsed observed days on its grid, or None when nothing is observed yet."""
    observed = load(config)
    if observed is None or not observed.dates:
        return None
    return _for_run_cached(
        observed.version,
        str(config.artifact_dir),
        first_day,
        int(day_count),
        tuple(np.round(latitudes, 5)),
        tuple(np.round(longitudes, 5)),
        float(config.min_coverage),
        int(config.lookback_days),
    )


@lru_cache(maxsize=16)
def _for_run_cached(
    observed_version: str,
    artifact_dir: str,
    first_day: date,
    day_count: int,
    latitudes: tuple[float, ...],
    longitudes: tuple[float, ...],
    min_coverage: float,
    lookback_days: int,
) -> RunObservations | None:
    observed = load(ObservedRainConfig(artifact_dir=Path(artifact_dir)))
    if observed is None:
        return None
    usable = {
        day: position
        for position, day in enumerate(observed.dates)
        if observed.coverage.get(day, 1.0) >= min_coverage
    }
    # Leading run days that are observed, without a gap: elapsed days become observations up to
    # the first day that has none yet (IMERG runs a day or two behind).
    leading: list[int] = []
    for offset in range(day_count):
        position = usable.get(first_day + timedelta(days=offset))
        if position is None:
            break
        leading.append(position)
    before: list[int] = []
    for offset in range(1, lookback_days + 1):
        position = usable.get(first_day - timedelta(days=offset))
        if position is None:
            break
        before.append(position)
    before.reverse()
    if not leading and not before:
        return None
    grid_lat = np.asarray(latitudes, dtype=float)
    grid_lon = np.asarray(longitudes, dtype=float)
    values = _regrid(observed, leading, grid_lat, grid_lon)
    lookback = _regrid(observed, before, grid_lat, grid_lon) if before else None
    return RunObservations(
        values=values,
        lookback=lookback,
        observed_through=(first_day + timedelta(days=len(leading) - 1)) if leading else None,
        source_label=observed.source_label,
        version=observed.version,
        synthetic=observed.synthetic,
    )


def _regrid(observed: ObservedDaily, positions: list[int], latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Observed days onto the forecast grid: bilinear, with the nearest value at the window's edge."""
    if not positions:
        return np.empty((0, latitudes.size, longitudes.size), dtype=np.float32)
    source = xr.DataArray(
        observed.values[positions],
        dims=("time", "latitude", "longitude"),
        coords={"latitude": observed.latitudes, "longitude": observed.longitudes},
    )
    linear = source.interp(latitude=latitudes, longitude=longitudes, method="linear")
    nearest = source.interp(latitude=latitudes, longitude=longitudes, method="nearest", kwargs={"fill_value": None})
    values = np.where(np.isfinite(linear.values), linear.values, nearest.values)
    return np.clip(values, 0.0, None).astype(np.float32)


def apply(
    members: np.ndarray | None,
    daily: np.ndarray,
    observations: RunObservations | None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Overwrite the run's elapsed observed days with the observations, in every member.

    Returns (members, daily, forecast) where ``forecast`` keeps the members' mean as forecast for
    the overwritten days (so the drawer can say what had been expected), or None when nothing
    was overwritten. Cells without an observed value keep the forecast.
    """
    if observations is None or observations.days == 0:
        return members, daily, None
    days = min(observations.days, daily.shape[0])
    observed = observations.values[:days]
    has_value = np.isfinite(observed)
    forecast = daily[:days].copy()
    if members is not None:
        members = members.copy()
        members[:, :days] = np.where(has_value[None], observed[None], members[:, :days])
        daily = np.nanmean(members, axis=0).astype(np.float32)
    else:
        daily = daily.copy()
        daily[:days] = np.where(has_value, observed, daily[:days])
    return members, daily, forecast


# --------------------------------------------------------------------------- ingest


@dataclass(frozen=True)
class ObservedFile:
    name: str  # relative to the source root
    size: int | None


class _LocalObservedSource:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.label = f"local:{self.root.name}"

    def list_files(self) -> list[ObservedFile]:
        if not self.root.is_dir():
            raise ObservedError(f"Observed-rain folder does not exist: {self.root}")
        return sorted(
            (ObservedFile(path.relative_to(self.root).as_posix(), path.stat().st_size) for path in self.root.rglob("*.nc")),
            key=lambda item: item.name,
        )

    def fetch(self, item: ObservedFile, cache_dir: Path) -> Path:
        return self.root / item.name


class _AzureObservedSource:
    def __init__(self, sas_url: str, prefix: str):
        from cumulus.subseasonal.sources import AzureBlobSasSource

        self._blob = AzureBlobSasSource(sas_url, prefix=prefix)
        self.label = self._blob.label

    def list_files(self) -> list[ObservedFile]:
        return sorted(
            (ObservedFile(name, size) for name, size in self._blob.list_blobs() if name.endswith(".nc")),
            key=lambda item: item.name,
        )

    def fetch(self, item: ObservedFile, cache_dir: Path) -> Path:
        target = cache_dir / item.name
        if not (target.exists() and item.size is not None and target.stat().st_size == item.size):
            self._blob.download(item.name, target)
        return target


def build_source(config: ObservedRainConfig, kind: str, path: Path | None):
    if kind == "auto":
        kind = "azure" if (config.azure_sas_url or os.environ.get("IMERG_SAS_URL")) else "local"
    if kind == "azure":
        sas_url = config.azure_sas_url or os.environ.get("IMERG_SAS_URL")
        if not sas_url:
            raise ObservedError("Set CUMULUS_SUBSEASONAL__OBSERVED__AZURE_SAS_URL (or IMERG_SAS_URL): a container SAS URL with Read + List.")
        return _AzureObservedSource(sas_url, config.azure_prefix)
    root = path or config.local_source_dir
    if root is None:
        raise ObservedError("Pass --path or set CUMULUS_SUBSEASONAL__OBSERVED__LOCAL_SOURCE_DIR for a local ingest.")
    return _LocalObservedSource(Path(root))


@dataclass(frozen=True)
class ObservedIngestResult:
    artifact_path: Path
    first_date: date | None
    last_date: date | None
    days: int
    new_dates: tuple[date, ...]
    unchanged: bool = False


def ingest(settings: Settings, source: Any, *, today: date | None = None, force: bool = False) -> ObservedIngestResult:
    """Fold any new daily files into the rolling artifact; a no-op when nothing is new."""
    config = settings.subseasonal.observed
    bounds = settings.subseasonal.bounds
    directory = Path(config.artifact_dir)
    manifest_path = directory / MANIFEST_FILE_NAME
    previous: dict[str, Any] = {}
    if manifest_path.exists() and not force:
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    known = {entry["file"]: entry.get("size") for entry in previous.get("source_files") or []}
    today = today or datetime.now(UTC).date()
    oldest = today - timedelta(days=int(config.keep_days))

    files = source.list_files()
    # Skip files already folded in and, where the name carries a date, files outside the window.
    pending = [
        item
        for item in files
        if known.get(item.name, -1) != item.size and not ((named := _date_in_name(item.name)) and named < oldest)
    ]
    existing = load(config) if previous else None
    days: dict[date, np.ndarray] = {}
    grid: tuple[np.ndarray, np.ndarray] | None = None
    if existing is not None:
        grid = (existing.latitudes, existing.longitudes)
        days = {day: existing.values[position] for position, day in enumerate(existing.dates)}
    if not pending and existing is not None:
        return ObservedIngestResult(directory / ARTIFACT_FILE_NAME, existing.dates[0], existing.last_date, len(existing.dates), (), unchanged=True)

    cache_dir = Path(config.raw_cache_dir or Path(settings.raw_data_dir) / "imerg")
    new_dates: list[date] = []
    sources = dict(known)
    for item in pending:
        path = source.fetch(item, cache_dir)
        for day, field, axes in _read_observed_file(path, item.name, config, bounds):
            if grid is not None and not (np.array_equal(axes[0], grid[0]) and np.array_equal(axes[1], grid[1])):
                # Upstream changed its grid: start again from what is being read now.
                days, grid = {}, None
            grid = grid or axes
            days[day] = field
            new_dates.append(day)
        sources[item.name] = item.size
    if grid is None or not days:
        raise ObservedError(f"No observed daily rainfall found at {source.label}.")

    kept = sorted(day for day in days if oldest <= day < today)
    if not kept:
        raise ObservedError(f"None of the observed days at {source.label} fall in the last {config.keep_days} days.")
    values = np.stack([days[day] for day in kept]).astype(np.float32)
    values = np.round(values / PACK_SCALE) * PACK_SCALE
    coverage = {day.isoformat(): round(_coverage(values[position], grid), 4) for position, day in enumerate(kept)}
    artifact = _write(directory, kept, grid, values, coverage, sources, config, source_label=source.label)
    return ObservedIngestResult(artifact, kept[0], kept[-1], len(kept), tuple(sorted(set(new_dates) & set(kept))))


def _read_observed_file(
    path: Path, name: str, config: ObservedRainConfig, bounds: SubseasonalBoundsConfig
) -> list[tuple[date, np.ndarray, tuple[np.ndarray, np.ndarray]]]:
    """Every day in one file as (date, (lat, lon) mm clipped to the window, axes)."""
    dataset = _open(path)
    try:
        variable = next((alias for alias in config.variable_aliases if alias in dataset.data_vars), None)
        if variable is None:
            raise ObservedError(f"{name}: none of {config.variable_aliases} found (has {list(dataset.data_vars)}).")
        dataset = _normalize(dataset)
        field = dataset[variable]
        factor = _to_mm_per_day(str(field.attrs.get("units", "mm")))
        stamps = _dates(dataset, field, name)
        if "time" not in field.dims:
            field = field.expand_dims("time")
        extra = set(field.dims) - {"time", "latitude", "longitude"}
        if extra:
            raise ObservedError(f"{name}: unexpected dimensions {sorted(extra)} on {variable}.")
        field = field.transpose("time", "latitude", "longitude").sortby("latitude").sortby("longitude")
        field = field.sel(
            latitude=slice(bounds.latitude_min - MARGIN_DEGREES, bounds.latitude_max + MARGIN_DEGREES),
            longitude=slice(bounds.longitude_min - MARGIN_DEGREES, bounds.longitude_max + MARGIN_DEGREES),
        )
        if field.sizes["latitude"] < 2 or field.sizes["longitude"] < 2:
            raise ObservedError(f"{name}: the file does not cover Ghana.")
        axes = (np.round(field["latitude"].values.astype(float), 4), np.round(field["longitude"].values.astype(float), 4))
        raw = np.asarray(field.values, dtype=np.float64) * factor
        raw = np.where(np.isfinite(raw) & (raw > -1e-6), raw, np.nan)  # fill values (-9999.9) and negatives
        return [(stamps[position], np.clip(raw[position], 0.0, None), axes) for position in range(raw.shape[0])]
    finally:
        dataset.close()


def _open(path: Path) -> xr.Dataset:
    errors = []
    for engine in ("h5netcdf", "scipy", None):
        try:
            return xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
        except Exception as exc:  # noqa: BLE001 - try the next reader
            errors.append(f"{engine}: {exc}")
    raise ObservedError(f"{path.name}: cannot be opened ({'; '.join(errors)[:300]}).")


def _normalize(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    for name in list(dataset.dims) + list(dataset.coords):
        lowered = str(name).lower()
        if lowered in {"lat", "y"}:
            rename[name] = "latitude"
        elif lowered in {"lon", "x"}:
            rename[name] = "longitude"
    return dataset.rename(rename) if rename else dataset


def _dates(dataset: xr.Dataset, field: xr.DataArray, name: str) -> list[date]:
    if "time" in field.dims or "time" in dataset.coords:
        bounds_name = dataset["time"].attrs.get("bounds")
        if bounds_name and bounds_name in dataset.variables:
            starts = np.asarray(dataset[bounds_name].values)[..., 0].reshape(-1)
        else:
            starts = np.asarray(dataset["time"].values).reshape(-1)
        try:
            return [_to_date(value) for value in starts]
        except (TypeError, ValueError):
            pass
    named = _date_in_name(name)
    if named is None:
        raise ObservedError(f"{name}: no time coordinate and no date in the file name.")
    return [named]


def _to_mm_per_day(units: str) -> float:
    lowered = units.lower().replace(" ", "")
    if "kgm-2s-1" in lowered or "kg/m2/s" in lowered or lowered.endswith("s-1"):
        return 86400.0
    if lowered in {"m", "metre", "meter", "meters"}:
        return 1000.0
    if ("hr" in lowered or "h-1" in lowered or "/h" in lowered) and "day" not in lowered and "d-1" not in lowered:
        return 24.0
    return 1.0


def _to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day") and not isinstance(value, np.datetime64):
        return date(int(value.year), int(value.month), int(value.day))
    stamp = np.datetime64(value, "D").astype(object)
    return stamp if isinstance(stamp, date) else date.fromisoformat(str(stamp))


def _date_in_name(name: str) -> date | None:
    match = DATE_IN_NAME.search(Path(name).name)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _coverage(field: np.ndarray, grid: tuple[np.ndarray, np.ndarray]) -> float:
    latitudes, longitudes = grid
    rows = (latitudes >= GHANA_BOX["latitude"][0]) & (latitudes <= GHANA_BOX["latitude"][1])
    cols = (longitudes >= GHANA_BOX["longitude"][0]) & (longitudes <= GHANA_BOX["longitude"][1])
    window = field[np.ix_(rows, cols)]
    return float(np.isfinite(window).mean()) if window.size else 0.0


def _write(
    directory: Path,
    dates: list[date],
    grid: tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    coverage: dict[str, float],
    sources: dict[str, int | None],
    config: ObservedRainConfig,
    *,
    source_label: str,
    synthetic: bool = False,
) -> Path:
    dataset = xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), values)},
        coords={
            "time": np.asarray([np.datetime64(day.isoformat(), "ns") for day in dates]),
            "latitude": grid[0].astype(np.float32),
            "longitude": grid[1].astype(np.float32),
        },
        attrs={
            "title": "Observed daily rainfall for Ghana",
            "source": config.source_label,
            "comment": "Daily 00-24 UTC totals; the date is the day the rain fell.",
            "Conventions": "CF-1.8",
        },
    )
    dataset["precip"].attrs.update({"long_name": "observed 24-hour precipitation", "units": "mm"})
    encoding = {
        "precip": {"dtype": "int16", "scale_factor": PACK_SCALE, "add_offset": 0.0, "_FillValue": np.int16(-32768)},
        "time": {"units": "days since 1970-01-01", "dtype": "int32"},
    }
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}-"))
    try:
        artifact = staging / ARTIFACT_FILE_NAME
        dataset.to_netcdf(artifact, engine="scipy", format="NETCDF3_64BIT", encoding=encoding)
        digest = hashlib.sha1(artifact.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "source_label": config.source_label,
            "ingested_from": source_label,
            "ingested_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "artifact": ARTIFACT_FILE_NAME,
            "unit": "mm",
            "first_date": dates[0].isoformat(),
            "last_date": dates[-1].isoformat(),
            "days": len(dates),
            "grid": {
                "shape": [int(grid[0].size), int(grid[1].size)],
                "resolution_degrees": round(float(np.median(np.diff(grid[0]))), 4) if grid[0].size > 1 else None,
            },
            "coverage": coverage,
            "sha1": digest,
            "synthetic": synthetic,
            "source_files": [{"file": name, "size": size} for name, size in sorted(sources.items())],
        }
        (staging / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        backup = None
        if directory.exists():
            backup = directory.with_name(f".{directory.name}-old")
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(directory, backup)
        os.replace(staging, directory)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory / ARTIFACT_FILE_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fold new IMERG late-run daily files into the observed-rain artifact.")
    parser.add_argument("--source", choices=["auto", "local", "azure"], default="auto")
    parser.add_argument("--path", type=Path, help="Local folder holding the daily .nc files (searched recursively).")
    parser.add_argument("--force", action="store_true", help="Rebuild from every file rather than adding new ones.")
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        source = build_source(settings.subseasonal.observed, args.source, args.path)
        result = ingest(settings, source, force=args.force)
    except ObservedError as exc:
        print(f"observed ingest failed: {exc}", file=sys.stderr)
        return 1
    if result.unchanged:
        print(f"Observed rain up to {result.last_date}: nothing new.")
    else:
        added = f", new: {', '.join(day.isoformat() for day in result.new_dates)}" if result.new_dates else ""
        print(f"Observed rain {result.first_date} to {result.last_date} ({result.days} days) -> {result.artifact_path}{added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
