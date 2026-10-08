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

Layer = Literal["rainfall", "rainy_days", "dry_spell_days", "wet_spell_days", "onset"]
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
    # Daily: where the rains have set in by the chosen day, and how soon elsewhere. Total: the date.
    "onset": {
        "label": "Onset",
        "description": (
            "First day with at least {onset_mm} mm of rain within {onset_window} days and no dry spell longer than "
            "{onset_dry} days in the {onset_guard} days from then. Searched only inside this forecast."
        ),
        "aggregations": ["daily", "total"],
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
# Prepared runs by cache key, so the runs list can reuse a run already in memory.
_PREPARED: dict[tuple[Any, ...], "PreparedRun"] = {}


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
    onset_rule: dict[str, float] = field(default_factory=dict)
    windows: list[metrics.Window] = field(default_factory=list)
    weekly: np.ndarray | None = None
    outlook: dict[str, np.ndarray] = field(default_factory=dict)
    # Per indicator: (day, lat, lon) 100/0 flags and (week, lat, lon) day counts.
    indicator_daily: dict[str, np.ndarray] = field(default_factory=dict)
    indicator_weekly: dict[str, np.ndarray] = field(default_factory=dict)
    onset: metrics.Onset | None = None
    # Onset of each area's mean rainfall, so map hover agrees with the drawer: {level: lead days}.
    area_onset: dict[str, np.ndarray] = field(default_factory=dict)
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
    with _LOAD_LOCK:
        key = _run_key(settings, resolved)
        run = _load_run_cached(*key)
        _PREPARED[key] = run
        while len(_PREPARED) > 4:
            _PREPARED.pop(next(iter(_PREPARED)))
        return run


def _run_key(settings: Settings, run_id: str) -> tuple[Any, ...]:
    """Everything a prepared run depends on: the artifact (by mtime), boundaries and thresholds."""
    run_dir = source_dir(settings) / run_id
    config = settings.subseasonal
    return (
        str(run_dir),
        (run_dir / MANIFEST_FILE_NAME).stat().st_mtime_ns,
        str(settings.seasonal_map.district_geojson_path),
        str(config.artifact_dir),
        float(config.wet_day_threshold_mm),
        int(config.dry_spell_min_days),
        int(config.wet_spell_min_days),
        (
            float(config.onset_threshold_mm),
            int(config.onset_window_days),
            int(config.onset_guard_days),
            int(config.onset_guard_max_dry_days),
        ),
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
    onset_rule: tuple[float, int, int, int] = (20.0, 3, 30, 10),
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
        onset_rule=dict(zip(("threshold_mm", "window_days", "guard_days", "guard_max_dry_days"), onset_rule)),
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
    run.onset = _onset(run, daily)
    run.outlook["onset"] = run.onset.day.astype(np.float32)
    weights = geomask.area_weights(areas)
    run.land_weights = weights.land
    flat = daily.reshape(run.day_count, -1).astype(float)
    for level in ("region", "district"):
        names, matrix = weights.matrices[level]
        run.area_matrices[level] = (names, matrix)
        sums = matrix.sum(axis=1)
        area_daily = np.divide(flat @ matrix.T, sums, out=np.full((run.day_count, len(names)), np.nan), where=sums > 0)
        run.area_onset[level] = _onset(run, area_daily).day
    return run


def _onset(run: PreparedRun, values: np.ndarray) -> metrics.Onset:
    rule = run.onset_rule
    return metrics.onset(
        values,
        threshold_mm=float(rule["threshold_mm"]),
        window_days=int(rule["window_days"]),
        guard_days=int(rule["guard_days"]),
        guard_max_dry_days=int(rule["guard_max_dry_days"]),
        dry_threshold_mm=run.wet_threshold_mm,
    )


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
    runs = [run_summary(settings, _summary_view(settings, run_id), active=run_id == active) for run_id in reversed(run_ids)]
    return {"active_run_id": active, "runs": runs}


@dataclass
class _SummaryView:
    """What a run summary needs: national means only, so listing every kept run stays cheap."""

    run_id: str
    manifest: dict[str, Any]
    init_time: datetime
    day_count: int
    windows: list[metrics.Window]
    national_daily: np.ndarray
    national_weekly: np.ndarray
    wet_threshold_mm: float
    dry_spell_min_days: int
    wet_spell_min_days: int
    onset_rule: dict[str, float]

    @property
    def init_date(self) -> date:
        return self.init_time.date()


def _summary_view(settings: Settings, run_id: str) -> _SummaryView:
    run_dir = source_dir(settings) / run_id
    config = settings.subseasonal
    with _LOAD_LOCK:
        cached = _prepared_if_loaded(settings, run_id)
        if cached is not None:
            daily_mean = _land_mean(cached, cached.daily)
            weekly_mean = _land_mean(cached, cached.weekly)
            return _SummaryView(
                cached.run_id, cached.manifest, cached.init_time, cached.day_count, cached.windows, daily_mean, weekly_mean,
                cached.wet_threshold_mm, cached.dry_spell_min_days, cached.wet_spell_min_days, cached.onset_rule,
            )
        manifest, init_time, daily_mean, windows, weekly_mean = _national_means_cached(
            str(run_dir),
            (run_dir / MANIFEST_FILE_NAME).stat().st_mtime_ns,
            str(settings.seasonal_map.district_geojson_path),
            str(config.artifact_dir),
        )
    return _SummaryView(
        str(manifest["run_id"]), manifest, init_time, int(daily_mean.size), windows, daily_mean, weekly_mean,
        float(config.wet_day_threshold_mm), int(config.dry_spell_min_days), int(config.wet_spell_min_days),
        {
            "threshold_mm": float(config.onset_threshold_mm),
            "window_days": int(config.onset_window_days),
            "guard_days": int(config.onset_guard_days),
            "guard_max_dry_days": int(config.onset_guard_max_dry_days),
        },
    )


def _prepared_if_loaded(settings: Settings, run_id: str) -> PreparedRun | None:
    """The fully prepared run if it is already in memory (e.g. the active run), without loading it."""
    return _PREPARED.get(_run_key(settings, run_id))


@lru_cache(maxsize=32)
def _national_means_cached(run_dir: str, _manifest_mtime: int, geojson_path: str, mask_cache_dir: str):
    directory = Path(run_dir)
    manifest = json.loads((directory / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    with xr.open_dataset(directory / manifest.get("artifact", "rainfall.nc"), engine="scipy") as dataset:
        dataset = dataset.load()
    daily = np.asarray(dataset["precip"].values, dtype=np.float32)
    latitudes = np.asarray(dataset["latitude"].values, dtype=float)
    longitudes = np.asarray(dataset["longitude"].values, dtype=float)
    areas = geomask.load_area_index(Path(geojson_path), latitudes, longitudes, Path(mask_cache_dir))
    land = geomask.area_weights(areas).land.ravel()
    init_time = datetime.fromisoformat(manifest["init_time"])
    if init_time.tzinfo is None:
        init_time = init_time.replace(tzinfo=UTC)
    windows = metrics.weekly_windows(int(daily.shape[0]))
    weekly = metrics.window_sums(daily, windows)

    def _mean(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(values.shape[0], -1).astype(float)
        return flat @ land / float(land.sum())

    return manifest, init_time, _mean(daily), windows, _mean(weekly)


def run_summary(settings: Settings, run: _SummaryView, *, active: bool) -> dict[str, Any]:
    manifest = run.manifest
    config = settings.subseasonal
    today = datetime.now(UTC).date()
    age_days = (today - run.init_date).days
    national_daily = run.national_daily
    national_weekly = run.national_weekly
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
        "thresholds": _thresholds(run),
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


def onset_countdown(onset_day: np.ndarray, day: int) -> np.ndarray:
    """Days from ``day`` until onset (<= 0: already started), NO_ONSET where none is forecast."""
    values = np.where(onset_day > 0, onset_day - day, legends.NO_ONSET)
    return np.where(np.isnan(onset_day), np.nan, values).astype(np.float32)


def layer_field(run: PreparedRun, selection: LayerSelection) -> np.ndarray:
    if selection.layer == "onset" and selection.aggregation == "daily":
        return onset_countdown(run.outlook["onset"], selection.index)
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
    legend = _legend_for(run, selection)
    values = layer_field(run, selection)
    start_day, end_day = _selection_days(run, selection)
    weights = run.land_weights
    assert weights is not None
    land_values = values[weights > 0]
    if selection.layer == "onset":
        stats = _onset_stats(run, selection.index if selection.aggregation == "daily" else None)
    else:
        stats = {
            "mean": _round(float(np.average(values, weights=weights))) if float(weights.sum()) > 0 else None,
            "max": _round(float(np.nanmax(land_values))) if land_values.size else None,
            "min": _round(float(np.nanmin(land_values))) if land_values.size else None,
        }
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
        "stats": stats,
        "description": _describe(LAYERS[selection.layer]["description"], run),
        "progress": _onset_progress(run) if selection.layer == "onset" else None,
    }


def _onset_progress(run: PreparedRun) -> list[float]:
    """% of Ghana where the rains have set in by each forecast day: the onset front, for the timeline."""
    assert run.onset is not None and run.land_weights is not None
    weights = run.land_weights
    total = float(weights.sum())
    day = np.nan_to_num(run.onset.day)
    if total <= 0:
        return [0.0] * run.day_count
    return [round(100 * float(weights[(day > 0) & (day <= lead)].sum()) / total, 1) for lead in range(1, run.day_count + 1)]


def _legend_for(run: PreparedRun, selection: LayerSelection) -> legends.Legend:
    if selection.layer == "onset":
        if selection.aggregation == "daily":
            return legends.ONSET_COUNTDOWN
        return _onset_legend(run.init_date, run.day_count)
    return LAYER_LEGENDS[(selection.layer, selection.aggregation)]


@lru_cache(maxsize=8)
def _onset_legend(init_date: date, day_count: int) -> legends.Legend:
    return legends.onset_legend(init_date, day_count)


def _onset_stats(run: PreparedRun, as_of: int | None = None) -> dict[str, Any]:
    """Land-weighted onset shares of Ghana and the date range (lead days).

    Whole window: ``share`` = has an onset in the forecast. For one day (``as_of``): ``share`` =
    rains have set in by that day, ``upcoming_share`` = still to come within the forecast.
    """
    assert run.onset is not None and run.land_weights is not None
    weights = run.land_weights
    total = float(weights.sum())
    day = run.onset.day
    found = np.nan_to_num(day) > 0
    provisional = found & (np.nan_to_num(run.onset.guard_days, nan=0) < run.onset_rule["guard_days"])
    on_land = found & (weights > 0)

    def _share(mask: np.ndarray) -> float | None:
        return _round(100 * float(weights[mask].sum()) / total) if total > 0 else None

    started = found if as_of is None else found & (np.nan_to_num(day) <= as_of)
    upcoming = found & ~started
    return {
        "mean": None,
        "min": _round(float(day[on_land].min())) if on_land.any() else None,
        "max": _round(float(day[on_land].max())) if on_land.any() else None,
        "share": _share(started),
        "upcoming_share": _share(upcoming) if as_of is not None else None,
        "provisional_share": _share(provisional),
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

    field_values = layer_field(run, selection)
    if selection.layer == "onset":
        # Interpolating between day numbers would invent dates, so each pixel takes its cell's value.
        values = _nearest(field_values, run.latitudes, run.longitudes, latitudes, longitudes)
    else:
        values = _bilinear(field_values, run.latitudes, run.longitudes, latitudes, longitudes)
    legend = _legend_for(run, selection)
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


def _nearest(grid: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Nearest-cell lookup of an ascending regular grid onto the (lat x lon) pixel mesh."""
    rows = np.clip(np.rint(np.interp(lat, grid_lat, np.arange(grid_lat.size))).astype(int), 0, grid_lat.size - 1)
    cols = np.clip(np.rint(np.interp(lon, grid_lon, np.arange(grid_lon.size))).astype(int), 0, grid_lon.size - 1)
    return grid[np.ix_(rows, cols)]


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
    if selection.layer == "onset":
        # A mean of onset days is not a date anyone would see; use the onset of the area's mean rain.
        means = run.area_onset[level_key]
        if selection.aggregation == "daily":
            means = onset_countdown(means, selection.index)
    else:
        values = layer_field(run, selection).ravel()
        sums = matrix.sum(axis=1)
        means = np.divide(matrix @ values, sums, out=np.full(len(names), np.nan, dtype=np.float32), where=sums > 0)
    legend = _legend_for(run, selection)
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
        "onset": _onset_payload(run, series),
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
        "thresholds": _thresholds(run),
        "guidance": GUIDANCE,
    }


def _onset_payload(run: PreparedRun, series: np.ndarray) -> dict[str, Any] | None:
    result = _onset(run, series)
    day = float(result.day)
    if not math.isfinite(day) or day <= 0:
        return None
    guard_days = int(result.guard_days)
    return {
        "day": int(day),
        "date": metrics.rain_day(run.init_date, int(day)),
        "rain_mm": _round(float(result.rain_mm)),
        "longest_dry_after": int(result.longest_dry_after),
        "guard_days": guard_days,
        "provisional": guard_days < int(run.onset_rule["guard_days"]),
    }


# --------------------------------------------------------------------------- helpers


def _thresholds(run: Any) -> dict[str, Any]:
    rule = run.onset_rule
    return {
        "wet_day_mm": run.wet_threshold_mm,
        "dry_spell_min_days": run.dry_spell_min_days,
        "wet_spell_min_days": run.wet_spell_min_days,
        "onset_mm": float(rule["threshold_mm"]),
        "onset_window_days": int(rule["window_days"]),
        "onset_guard_days": int(rule["guard_days"]),
        "onset_max_dry_days": int(rule["guard_max_dry_days"]),
    }


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
    if selection.layer == "onset":
        label = "Onset status" if selection.aggregation == "daily" else "Onset date"
    if selection.aggregation == "daily":
        return f"{label} · {start:%a} {_day_month(start)}"
    if selection.aggregation == "weekly":
        window = run.windows[selection.index - 1]
        name = f"Days {window.start_day}–{window.end_day}" if window.partial else f"Week {selection.index}"
        return f"{label} · {name} ({_day_month(start)}–{_day_month(end)})"
    return f"{label} · {run.day_count} days ({_day_month(start)}–{_day_month(end)})"


def _day_month(value: date) -> str:
    return f"{value.day} {value:%b}"


def _describe(template: str, run: Any) -> str:
    return template.format(
        wet=_fmt(run.wet_threshold_mm),
        dry=run.dry_spell_min_days,
        wetspell=run.wet_spell_min_days,
        onset_mm=_fmt(float(run.onset_rule["threshold_mm"])),
        onset_window=int(run.onset_rule["window_days"]),
        onset_guard=int(run.onset_rule["guard_days"]),
        onset_dry=int(run.onset_rule["guard_max_dry_days"]),
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
