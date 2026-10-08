"""Serve sub-seasonal (46-day) IFS-UNet rainfall: runs, map layers, tiles, point and area series.

Everything is computed from the compact NetCDF3 artifact written by
``cumulus.subseasonal.ingest``. A run is loaded once per process (a few MB in memory) and every
derived field (weekly totals, outlook metrics, area weights) is cached alongside it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from urllib.parse import urlencode

import numpy as np
import xarray as xr

from cumulus.api.errors import (
    InvalidCoordinatesError,
    InvalidSubseasonalLayerError,
    SubseasonalAreaNotFoundError,
    SubseasonalRunNotAvailableError,
    SubseasonalRunNotFoundError,
)
from cumulus.settings import Settings
from cumulus.subseasonal import geomask, legends, metrics
from cumulus.subseasonal.ingest import ACTIVE_FILE_NAME, MANIFEST_FILE_NAME, list_run_ids, source_dir
from cumulus.utils.tiles import TILE_SIZE, encode_png, tile_pixel_latitudes, tile_pixel_longitudes

Layer = Literal["rainfall", "rainy_days", "dry_spell_days", "wet_spell_days"]
Aggregation = Literal["daily", "weekly", "total"]
AreaLevel = Literal["region", "district"]

LAYERS: dict[str, dict[str, Any]] = {
    "rainfall": {
        "label": "Rainfall",
        "description": "Forecast rainfall accumulation.",
        "aggregations": ["daily", "weekly", "total"],
    },
    # Indicators default to the whole window ("total" first) for callers that omit the period.
    "rainy_days": {
        "label": "Rainy days",
        "description": "Days with at least {wet} mm of rain.",
        "aggregations": ["total", "daily", "weekly"],
    },
    "dry_spell_days": {
        "label": "Dry-spell days",
        "description": "Days falling inside dry spells (≥{dry} consecutive days below {wet} mm).",
        "aggregations": ["total", "daily", "weekly"],
    },
    "wet_spell_days": {
        "label": "Wet-spell days",
        "description": "Days falling inside wet spells (≥{wetspell} consecutive days with ≥{wet} mm).",
        "aggregations": ["total", "daily", "weekly"],
    },
}
LAYER_LEGENDS = {
    ("rainfall", "daily"): legends.RAIN_DAILY,
    ("rainfall", "weekly"): legends.RAIN_WEEKLY,
    ("rainfall", "total"): legends.RAIN_TOTAL,
    ("rainy_days", "daily"): legends.RAINY_DAY_DAILY,
    ("rainy_days", "weekly"): legends.RAINY_DAYS_WEEKLY,
    ("rainy_days", "total"): legends.RAINY_DAYS,
    ("dry_spell_days", "daily"): legends.DRY_SPELL_DAILY,
    ("dry_spell_days", "weekly"): legends.DRY_SPELL_DAYS_WEEKLY,
    ("dry_spell_days", "total"): legends.DRY_SPELL_DAYS,
    ("wet_spell_days", "daily"): legends.WET_SPELL_DAILY,
    ("wet_spell_days", "weekly"): legends.WET_SPELL_DAYS_WEEKLY,
    ("wet_spell_days", "total"): legends.WET_SPELL_DAYS,
}
GUIDANCE = (
    "Single deterministic model run. Day-to-day detail beyond about 10 days is indicative only; "
    "rely on weekly totals and spell counts for planning."
)

TILE_CACHE_SIZE = 2048

_LOAD_LOCK = RLock()


@dataclass
class PreparedRun:
    run_id: str
    manifest: dict[str, Any]
    init_time: datetime
    latitudes: np.ndarray  # ascending
    longitudes: np.ndarray  # ascending
    daily: np.ndarray  # (day, lat, lon) mm
    areas: geomask.AreaIndex
    wet_threshold_mm: float
    dry_spell_min_days: int
    wet_spell_min_days: int
    windows: list[metrics.Window] = field(default_factory=list)
    weekly: np.ndarray | None = None
    outlook: dict[str, np.ndarray] = field(default_factory=dict)
    # Per indicator: (day, lat, lon) 100/0 flags and (week, lat, lon) day counts.
    indicator_daily: dict[str, np.ndarray] = field(default_factory=dict)
    indicator_weekly: dict[str, np.ndarray] = field(default_factory=dict)
    land_weights: np.ndarray | None = None
    area_matrices: dict[str, tuple[tuple[str, ...], np.ndarray]] = field(default_factory=dict)
    tile_cache: OrderedDict[tuple[Any, ...], bytes] = field(default_factory=OrderedDict)

    @property
    def day_count(self) -> int:
        return int(self.daily.shape[0])

    @property
    def init_date(self) -> date:
        return self.init_time.date()


# --------------------------------------------------------------------------- loading


def _active_run_id(settings: Settings) -> str:
    active_path = source_dir(settings) / ACTIVE_FILE_NAME
    try:
        run_id = json.loads(active_path.read_text(encoding="utf-8")).get("run_id")
    except (OSError, json.JSONDecodeError):
        run_id = None
    if run_id and (source_dir(settings) / run_id / MANIFEST_FILE_NAME).exists():
        return str(run_id)
    available = list_run_ids(settings)
    if available:
        return available[-1]
    raise SubseasonalRunNotAvailableError(
        "No sub-seasonal forecast run has been ingested yet. Run `python -m cumulus.subseasonal.ingest`."
    )


def load_run(settings: Settings, run_id: str | None = None) -> PreparedRun:
    resolved = run_id or _active_run_id(settings)
    if resolved not in list_run_ids(settings):
        raise SubseasonalRunNotFoundError(f"Sub-seasonal run '{resolved}' is not available.")
    run_dir = source_dir(settings) / resolved
    config = settings.subseasonal
    with _LOAD_LOCK:
        return _load_run_cached(
            str(run_dir),
            (run_dir / MANIFEST_FILE_NAME).stat().st_mtime_ns,
            str(settings.seasonal_map.district_geojson_path),
            str(config.artifact_dir),
            float(config.wet_day_threshold_mm),
            int(config.dry_spell_min_days),
            int(config.wet_spell_min_days),
        )


@lru_cache(maxsize=4)
def _load_run_cached(
    run_dir: str,
    _manifest_mtime: int,
    geojson_path: str,
    mask_cache_dir: str,
    wet_threshold_mm: float,
    dry_spell_min_days: int,
    wet_spell_min_days: int,
) -> PreparedRun:
    directory = Path(run_dir)
    manifest = json.loads((directory / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    with xr.open_dataset(directory / manifest.get("artifact", "rainfall.nc"), engine="scipy") as dataset:
        dataset = dataset.load()
    daily = np.asarray(dataset["precip"].values, dtype=np.float32)
    latitudes = np.asarray(dataset["latitude"].values, dtype=float)
    longitudes = np.asarray(dataset["longitude"].values, dtype=float)
    areas = geomask.load_area_index(Path(geojson_path), latitudes, longitudes, Path(mask_cache_dir))
    init_time = datetime.fromisoformat(manifest["init_time"])
    if init_time.tzinfo is None:
        init_time = init_time.replace(tzinfo=UTC)

    run = PreparedRun(
        run_id=str(manifest["run_id"]),
        manifest=manifest,
        init_time=init_time,
        latitudes=latitudes,
        longitudes=longitudes,
        daily=daily,
        areas=areas,
        wet_threshold_mm=wet_threshold_mm,
        dry_spell_min_days=dry_spell_min_days,
        wet_spell_min_days=wet_spell_min_days,
    )
    run.windows = metrics.weekly_windows(run.day_count)
    run.weekly = metrics.window_sums(daily, run.windows).astype(np.float32)
    run.outlook = {
        key: value.astype(np.float32)
        for key, value in metrics.outlook_metrics(
            daily,
            wet_threshold_mm=wet_threshold_mm,
            dry_spell_min_days=dry_spell_min_days,
            wet_spell_min_days=wet_spell_min_days,
        ).items()
    }
    run.indicator_daily, run.indicator_weekly = _indicator_fields(
        daily,
        run.windows,
        wet_threshold_mm=wet_threshold_mm,
        dry_spell_min_days=dry_spell_min_days,
        wet_spell_min_days=wet_spell_min_days,
    )
    land = areas.cell_land_fraction().astype(np.float32)
    run.land_weights = land
    for level in ("region", "district"):
        weights = areas.all_weights(level)
        names = tuple(weights)
        matrix = np.stack([weights[name].ravel() for name in names]).astype(np.float32) if names else np.zeros((0, land.size), np.float32)
        run.area_matrices[level] = (names, matrix)
    return run


def _indicator_fields(
    daily: np.ndarray,
    windows: list[metrics.Window],
    *,
    wet_threshold_mm: float,
    dry_spell_min_days: int,
    wet_spell_min_days: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Daily flags and weekly counts per indicator, consistent with the whole-window outlook.

    Spells are found over the full run, so a week's dry-spell days are the days of that week
    that belong to a spell (which may start or end outside the week), and the weekly counts
    add up to the outlook total.
    """
    wet = metrics.wet_mask(daily, wet_threshold_mm)
    dry = metrics.dry_mask(daily, wet_threshold_mm)
    flags = {
        "rainy_days": wet,
        "dry_spell_days": metrics.spell_day_mask(dry, dry_spell_min_days),
        "wet_spell_days": metrics.spell_day_mask(wet, wet_spell_min_days),
    }
    missing = np.isnan(daily)
    per_day: dict[str, np.ndarray] = {}
    per_week: dict[str, np.ndarray] = {}
    for key, flag in flags.items():
        counts = np.where(missing, np.nan, flag.astype(np.float32))
        per_day[key] = (counts * 100).astype(np.float32)
        per_week[key] = metrics.window_sums(counts, windows).astype(np.float32)
    return per_day, per_week


# --------------------------------------------------------------------------- run summaries


def list_runs(settings: Settings) -> dict[str, Any]:
    run_ids = list_run_ids(settings)
    if not run_ids:
        raise SubseasonalRunNotAvailableError(
            "No sub-seasonal forecast run has been ingested yet. Run `python -m cumulus.subseasonal.ingest`."
        )
    active = _active_run_id(settings)
    runs = [run_summary(settings, load_run(settings, run_id), active=run_id == active) for run_id in reversed(run_ids)]
    return {"active_run_id": active, "runs": runs}


def run_summary(settings: Settings, run: PreparedRun, *, active: bool) -> dict[str, Any]:
    manifest = run.manifest
    config = settings.subseasonal
    today = datetime.now(UTC).date()
    age_days = (today - run.init_date).days
    national_daily = _land_mean(run, run.daily)
    national_weekly = _land_mean(run, run.weekly)
    expected = int(manifest.get("expected_lead_days") or run.day_count)
    return {
        "run_id": run.run_id,
        "active": active,
        "source_id": manifest.get("source_id", config.source_id),
        "source_label": manifest.get("source_label", config.source_label),
        "model_label": manifest.get("model_label", config.model_label),
        "ensemble": manifest.get("ensemble", "deterministic"),
        "init_time": run.init_time,
        "ingested_at": manifest.get("ingested_at"),
        "lead_days": run.day_count,
        "expected_lead_days": expected,
        "missing_lead_days": list(range(run.day_count + 1, expected + 1)),
        "first_day": run.init_date,
        "last_day": metrics.rain_day(run.init_date, run.day_count),
        "age_days": age_days,
        "is_stale": age_days > int(config.stale_after_days),
        "unit": "mm",
        "grid_resolution_degrees": manifest.get("grid", {}).get("resolution_degrees"),
        "bounds": manifest.get("bounds"),
        "thresholds": {
            "wet_day_mm": run.wet_threshold_mm,
            "dry_spell_min_days": run.dry_spell_min_days,
            "wet_spell_min_days": run.wet_spell_min_days,
        },
        "days": [_day_payload(run, day, round(float(value), 1)) for day, value in enumerate(national_daily, start=1)],
        "weeks": [_window_payload(run, window, round(float(value), 1)) for window, value in zip(run.windows, national_weekly)],
        "layers": [
            {
                "layer": key,
                "label": spec["label"],
                "description": _describe(spec["description"], run),
                "aggregations": spec["aggregations"],
            }
            for key, spec in LAYERS.items()
        ],
        "guidance": GUIDANCE,
    }


# --------------------------------------------------------------------------- layers and tiles


@dataclass(frozen=True)
class LayerSelection:
    layer: str
    aggregation: str
    index: int  # day (daily), week (weekly) or 1 (total)


def resolve_selection(run: PreparedRun, layer: str, aggregation: str | None, index: int | None) -> LayerSelection:
    layer_key = str(layer or "rainfall").strip().lower()
    if layer_key not in LAYERS:
        raise InvalidSubseasonalLayerError(f"Unknown layer '{layer}'. Choose one of: {', '.join(LAYERS)}.")
    allowed = LAYERS[layer_key]["aggregations"]
    aggregation_key = str(aggregation or allowed[0]).strip().lower()
    if aggregation_key not in allowed:
        raise InvalidSubseasonalLayerError(f"Layer '{layer_key}' supports aggregations: {', '.join(allowed)}.")
    limit = {"daily": run.day_count, "weekly": len(run.windows), "total": 1}[aggregation_key]
    resolved_index = 1 if aggregation_key == "total" else int(index or 1)
    if not 1 <= resolved_index <= limit:
        raise InvalidSubseasonalLayerError(f"Index {index} is out of range for {aggregation_key} (1-{limit}).")
    return LayerSelection(layer_key, aggregation_key, resolved_index)


def layer_field(run: PreparedRun, selection: LayerSelection) -> np.ndarray:
    if selection.layer == "rainfall":
        if selection.aggregation == "daily":
            return run.daily[selection.index - 1]
        if selection.aggregation == "weekly":
            assert run.weekly is not None
            return run.weekly[selection.index - 1]
        return run.outlook["total"]
    if selection.aggregation == "daily":
        return run.indicator_daily[selection.layer][selection.index - 1]
    if selection.aggregation == "weekly":
        return run.indicator_weekly[selection.layer][selection.index - 1]
    return run.outlook[selection.layer]


def get_layer(settings: Settings, *, layer: str, aggregation: str | None, index: int | None, run_id: str | None) -> dict[str, Any]:
    run = load_run(settings, run_id)
    selection = resolve_selection(run, layer, aggregation, index)
    legend = LAYER_LEGENDS[(selection.layer, selection.aggregation)]
    values = layer_field(run, selection)
    start_day, end_day = _selection_days(run, selection)
    weights = run.land_weights
    assert weights is not None
    land_values = values[weights > 0]
    query = urlencode(
        {"run_id": run.run_id, "layer": selection.layer, "aggregation": selection.aggregation, "index": selection.index}
    )
    return {
        "run_id": run.run_id,
        "layer": selection.layer,
        "layer_label": LAYERS[selection.layer]["label"],
        "aggregation": selection.aggregation,
        "index": selection.index,
        "index_count": {"daily": run.day_count, "weekly": len(run.windows), "total": 1}[selection.aggregation],
        "title": _selection_title(run, selection),
        "start_day": start_day,
        "end_day": end_day,
        "start_date": metrics.rain_day(run.init_date, start_day),
        "end_date": metrics.rain_day(run.init_date, end_day),
        "period_start": run.init_time + timedelta(days=start_day - 1),
        "period_end": run.init_time + timedelta(days=end_day),
        "unit": legend.unit,
        "legend": legends.legend_payload(legend),
        "tile_url": f"/subseasonal/tiles/{{z}}/{{x}}/{{y}}.png?{query}",
        "bounds": run.manifest.get("bounds"),
        "stats": {
            "mean": _round(float(np.average(values, weights=weights))) if float(weights.sum()) > 0 else None,
            "max": _round(float(np.nanmax(land_values))) if land_values.size else None,
            "min": _round(float(np.nanmin(land_values))) if land_values.size else None,
        },
        "description": _describe(LAYERS[selection.layer]["description"], run),
    }


def render_tile(
    settings: Settings,
    *,
    z: int,
    x: int,
    y: int,
    layer: str,
    aggregation: str | None,
    index: int | None,
    run_id: str | None,
) -> bytes:
    run = load_run(settings, run_id)
    selection = resolve_selection(run, layer, aggregation, index)
    key = (selection, z, x, y)
    with _LOAD_LOCK:
        cached = run.tile_cache.get(key)
        if cached is not None:
            run.tile_cache.move_to_end(key)
            return cached
    png = _render_tile(run, selection, z, x, y)
    with _LOAD_LOCK:
        run.tile_cache[key] = png
        while len(run.tile_cache) > TILE_CACHE_SIZE:
            run.tile_cache.popitem(last=False)
    return png


@lru_cache(maxsize=1)
def _empty_tile() -> bytes:
    return encode_png(np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8))


def _render_tile(run: PreparedRun, selection: LayerSelection, z: int, x: int, y: int) -> bytes:
    longitudes = tile_pixel_longitudes(z, x)
    latitudes = tile_pixel_latitudes(z, y)
    areas = run.areas
    fine_lat, fine_lon = areas.fine_latitudes, areas.fine_longitudes
    lat_step = fine_lat[1] - fine_lat[0]
    lon_step = fine_lon[1] - fine_lon[0]
    row = np.floor((latitudes - (fine_lat[0] - lat_step / 2)) / lat_step).astype(int)
    col = np.floor((longitudes - (fine_lon[0] - lon_step / 2)) / lon_step).astype(int)
    row_ok = (row >= 0) & (row < fine_lat.size)
    col_ok = (col >= 0) & (col < fine_lon.size)
    if not row_ok.any() or not col_ok.any():
        return _empty_tile()
    inside = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    inside[np.ix_(row_ok, col_ok)] = areas.fine_inside[np.ix_(row[row_ok], col[col_ok])]
    if not inside.any():
        return _empty_tile()

    values = _bilinear(layer_field(run, selection), run.latitudes, run.longitudes, latitudes, longitudes)
    legend = LAYER_LEGENDS[(selection.layer, selection.aggregation)]
    classes = legends.classify(values, legend)
    colors = legends.bin_colors_rgb(legend)
    visible = inside & (classes >= 0)
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    rgba[..., :3] = colors[np.clip(classes, 0, len(colors) - 1)]
    rgba[..., 3] = np.where(visible, legend.opacity, 0).astype(np.uint8)
    rgba[~visible, :3] = 0
    return encode_png(rgba)


def _bilinear(grid: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of an ascending regular grid onto the (lat x lon) pixel mesh."""
    fy = np.clip(np.interp(lat, grid_lat, np.arange(grid_lat.size)), 0, grid_lat.size - 1)
    fx = np.clip(np.interp(lon, grid_lon, np.arange(grid_lon.size)), 0, grid_lon.size - 1)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, grid_lat.size - 1)
    x1 = np.minimum(x0 + 1, grid_lon.size - 1)
    wy = (fy - y0)[:, None]
    wx = (fx - x0)[None, :]
    top = grid[np.ix_(y0, x0)] * (1 - wx) + grid[np.ix_(y0, x1)] * wx
    bottom = grid[np.ix_(y1, x0)] * (1 - wx) + grid[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bottom * wy


# --------------------------------------------------------------------------- area values for map hover


def get_area_values(
    settings: Settings,
    *,
    level: str,
    layer: str,
    aggregation: str | None,
    index: int | None,
    run_id: str | None,
) -> dict[str, Any]:
    run = load_run(settings, run_id)
    level_key = _resolve_level(level)
    selection = resolve_selection(run, layer, aggregation, index)
    names, matrix = run.area_matrices[level_key]
    values = layer_field(run, selection).ravel()
    sums = matrix.sum(axis=1)
    means = np.divide(matrix @ values, sums, out=np.full(len(names), np.nan, dtype=np.float32), where=sums > 0)
    legend = LAYER_LEGENDS[(selection.layer, selection.aggregation)]
    return {
        "run_id": run.run_id,
        "level": level_key,
        "layer": selection.layer,
        "aggregation": selection.aggregation,
        "index": selection.index,
        "unit": legend.unit,
        "values": {name: _round(float(value)) for name, value in zip(names, means)},
    }


# --------------------------------------------------------------------------- point and area series


def sample_point(settings: Settings, *, latitude: float, longitude: float, run_id: str | None) -> dict[str, Any]:
    run = load_run(settings, run_id)
    bounds = run.manifest.get("bounds") or {}
    if not (
        bounds.get("latitude_min", -90) <= latitude <= bounds.get("latitude_max", 90)
        and bounds.get("longitude_min", -180) <= longitude <= bounds.get("longitude_max", 180)
    ):
        raise InvalidCoordinatesError("The point is outside the sub-seasonal forecast domain (Ghana).")
    row = int(np.abs(run.latitudes - latitude).argmin())
    col = int(np.abs(run.longitudes - longitude).argmin())
    fine_row = int(np.abs(run.areas.fine_latitudes - latitude).argmin())
    fine_col = int(np.abs(run.areas.fine_longitudes - longitude).argmin())
    district_index = int(run.areas.fine_district[fine_row, fine_col])
    district = run.areas.district_names[district_index] if district_index >= 0 else None
    region = run.areas.district_regions[district_index] if district_index >= 0 else None
    series = run.daily[:, row, col].astype(float)
    return {
        "kind": "point",
        "name": district or "Selected point",
        "region": region,
        "district": district,
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "nearest_latitude": round(float(run.latitudes[row]), 4),
        "nearest_longitude": round(float(run.longitudes[col]), 4),
        "inside_ghana": district is not None,
        "cell_count": 1,
        "support": f"{run.manifest.get('grid', {}).get('resolution_degrees', 0.1)}° grid cell",
        **_series_payload(run, series),
    }


def sample_area(settings: Settings, *, level: str, name: str, run_id: str | None) -> dict[str, Any]:
    run = load_run(settings, run_id)
    level_key = _resolve_level(level)
    names, matrix = run.area_matrices[level_key]
    lookup = {item.casefold(): position for position, item in enumerate(names)}
    position = lookup.get(str(name).strip().casefold())
    if position is None:
        raise SubseasonalAreaNotFoundError(f"No {level_key} named '{name}' in the Ghana boundaries.")
    weights = matrix[position]
    total_weight = float(weights.sum())
    flat = run.daily.reshape(run.day_count, -1).astype(float)
    series = flat @ weights / total_weight
    region = None
    if level_key == "district":
        region = run.areas.district_regions[run.areas.district_names.index(names[position])]
    else:
        region = names[position]
    cells_touched = int(np.count_nonzero(weights))
    return {
        "kind": level_key,
        "name": names[position],
        "region": region,
        "district": names[position] if level_key == "district" else None,
        "latitude": None,
        "longitude": None,
        "nearest_latitude": None,
        "nearest_longitude": None,
        "inside_ghana": True,
        "cell_count": cells_touched,
        "support": f"Area mean of {cells_touched} grid cell{'s' if cells_touched != 1 else ''}",
        **_series_payload(run, series),
    }


def _series_payload(run: PreparedRun, series: np.ndarray) -> dict[str, Any]:
    series = np.asarray(series, dtype=float)
    weekly = metrics.window_sums(series, run.windows)
    outlook = metrics.outlook_metrics(
        series,
        wet_threshold_mm=run.wet_threshold_mm,
        dry_spell_min_days=run.dry_spell_min_days,
        wet_spell_min_days=run.wet_spell_min_days,
    )
    spells = metrics.find_spells(
        series,
        wet_threshold_mm=run.wet_threshold_mm,
        dry_spell_min_days=run.dry_spell_min_days,
        wet_spell_min_days=run.wet_spell_min_days,
    )
    dry_days = metrics.spell_day_mask(metrics.dry_mask(series, run.wet_threshold_mm), run.dry_spell_min_days)
    wet_days = metrics.spell_day_mask(metrics.wet_mask(series, run.wet_threshold_mm), run.wet_spell_min_days)
    days = []
    for position, value in enumerate(series):
        payload = _day_payload(run, position + 1, _round(float(value)))
        payload["wet"] = bool(math.isfinite(value) and value >= run.wet_threshold_mm)
        payload["spell"] = "dry" if dry_days[position] else ("wet" if wet_days[position] else None)
        days.append(payload)
    return {
        "run_id": run.run_id,
        "init_time": run.init_time,
        "unit": "mm",
        "days": days,
        "weeks": [_window_payload(run, window, _round(float(value))) for window, value in zip(run.windows, weekly)],
        "metrics": {
            "total_mm": _round(float(outlook["total"])),
            "rainy_days": int(outlook["rainy_days"]),
            "dry_spell_days": int(outlook["dry_spell_days"]),
            "wet_spell_days": int(outlook["wet_spell_days"]),
            "max_day_mm": _round(float(np.nanmax(series))) if series.size else None,
            "max_day": int(np.nanargmax(series)) + 1 if series.size else None,
        },
        "spells": [
            {
                "kind": spell.kind,
                "start_day": spell.start_day,
                "end_day": spell.end_day,
                "days": spell.days,
                "start_date": metrics.rain_day(run.init_date, spell.start_day),
                "end_date": metrics.rain_day(run.init_date, spell.end_day),
                "open_start": spell.open_start,
                "open_end": spell.open_end,
                "total_mm": _round(float(np.nansum(series[spell.start_day - 1 : spell.end_day]))),
            }
            for spell in spells
        ],
        "thresholds": {
            "wet_day_mm": run.wet_threshold_mm,
            "dry_spell_min_days": run.dry_spell_min_days,
            "wet_spell_min_days": run.wet_spell_min_days,
        },
        "guidance": GUIDANCE,
    }


# --------------------------------------------------------------------------- helpers


def _land_mean(run: PreparedRun, values: np.ndarray) -> np.ndarray:
    weights = run.land_weights
    assert weights is not None
    flat = values.reshape(values.shape[0], -1).astype(float)
    return flat @ weights.ravel() / float(weights.sum())


def _day_payload(run: PreparedRun, day: int, value: float | None) -> dict[str, Any]:
    return {
        "day": day,
        "date": metrics.rain_day(run.init_date, day),
        "period_start": run.init_time + timedelta(days=day - 1),
        "period_end": run.init_time + timedelta(days=day),
        "value": value,
    }


def _window_payload(run: PreparedRun, window: metrics.Window, value: float | None) -> dict[str, Any]:
    return {
        "week": window.index,
        "start_day": window.start_day,
        "end_day": window.end_day,
        "start_date": metrics.rain_day(run.init_date, window.start_day),
        "end_date": metrics.rain_day(run.init_date, window.end_day),
        "days": window.days,
        "partial": window.partial,
        "value": value,
    }


def _selection_days(run: PreparedRun, selection: LayerSelection) -> tuple[int, int]:
    if selection.aggregation == "daily":
        return selection.index, selection.index
    if selection.aggregation == "weekly":
        window = run.windows[selection.index - 1]
        return window.start_day, window.end_day
    return 1, run.day_count


def _selection_title(run: PreparedRun, selection: LayerSelection) -> str:
    start_day, end_day = _selection_days(run, selection)
    start = metrics.rain_day(run.init_date, start_day)
    end = metrics.rain_day(run.init_date, end_day)
    label = LAYERS[selection.layer]["label"]
    if selection.aggregation == "daily":
        return f"{label} · {start:%a} {_day_month(start)}"
    if selection.aggregation == "weekly":
        window = run.windows[selection.index - 1]
        name = f"Days {window.start_day}–{window.end_day}" if window.partial else f"Week {selection.index}"
        return f"{label} · {name} ({_day_month(start)}–{_day_month(end)})"
    return f"{label} · {run.day_count} days ({_day_month(start)}–{_day_month(end)})"


def _day_month(value: date) -> str:
    return f"{value.day} {value:%b}"


def _describe(template: str, run: PreparedRun) -> str:
    return template.format(
        wet=_fmt(run.wet_threshold_mm),
        dry=run.dry_spell_min_days,
        wetspell=run.wet_spell_min_days,
    )


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _resolve_level(level: str) -> str:
    key = str(level or "").strip().lower()
    if key not in {"region", "district"}:
        raise InvalidSubseasonalLayerError("Area level must be 'region' or 'district'.")
    return key


def _round(value: float, digits: int = 1) -> float | None:
    return round(value, digits) if math.isfinite(value) else None
