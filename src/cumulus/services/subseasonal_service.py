"""Serve sub-seasonal (46-day) IFS-UNet rainfall: runs, map layers, tiles, point and area series.

Everything is computed from the compact NetCDF3 artifact written by
``cumulus.subseasonal.ingest``. A run is loaded once per process (a few MB in memory) and every
derived field (weekly totals, outlook metrics, area weights) is cached alongside it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
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
from cumulus.subseasonal.ingest import (
    ACTIVE_FILE_NAME,
    MANIFEST_FILE_NAME,
    UnknownModelError,
    for_model,
    list_run_ids,
    source_dir,
)
from cumulus.utils.tiles import TILE_SIZE, encode_png, tile_pixel_latitudes, tile_pixel_longitudes

Layer = Literal["rainfall", "rainy_days", "dry_spell_days", "wet_spell_days", "extreme_rain", "onset"]
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
    # Flood risk: the chance (ensemble) or forecast (single run) of a very wet day.
    "extreme_rain": {
        "label": "Extreme rain",
        "description": "Chance of {extreme} mm or more of rain in 24 hours.",
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
# Ensemble runs: daily indicators are % of members, so they need probability ramps.
ENSEMBLE_DAILY_LEGENDS = {
    "rainy_days": legends.RAINY_DAY_PROBABILITY,
    "dry_spell_days": legends.DRY_SPELL_PROBABILITY,
    "wet_spell_days": legends.WET_SPELL_PROBABILITY,
}
# A day counts as inside a spell (or as a rain day) in an ensemble series when at least half the members agree.
CONSENSUS_PERCENT = 50.0
SPREAD_QUANTILES = (10, 25, 50, 75, 90)
# Ensemble rainfall on the map: the members' mean, their median, or the chance of reaching an amount.
RAIN_STATS = ("mean", "median", "chance")
# Amounts offered for "chance of at least", per period; a fixed list keeps the tile cache bounded.
CHANCE_THRESHOLDS = {"daily": (10.0, 25.0, 50.0), "weekly": (25.0, 50.0, 100.0), "total": (100.0, 200.0, 300.0)}
GUIDANCE = (
    "Single deterministic model run. Day-to-day detail beyond about 10 days is indicative only; "
    "rely on weekly totals and spell counts for planning."
)
ENSEMBLE_GUIDANCE = (
    "Ensemble of {members} members. Rainfall maps show the ensemble mean; rain-day, spell and extreme-rain "
    "day maps show the share of members. Day-to-day detail beyond about 10 days is indicative only."
)


def _guidance(members: int) -> str:
    return ENSEMBLE_GUIDANCE.format(members=members) if members > 1 else GUIDANCE

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
    daily: np.ndarray  # (day, lat, lon) mm; the ensemble mean for an ensemble run
    areas: geomask.AreaIndex
    wet_threshold_mm: float
    dry_spell_min_days: int
    wet_spell_min_days: int
    onset_rule: dict[str, float] = field(default_factory=dict)
    extreme_rain_mm: float = 50.0
    # (member, day, lat, lon) mm for an ensemble run; None for a single run.
    members: np.ndarray | None = None
    windows: list[metrics.Window] = field(default_factory=list)
    weekly: np.ndarray | None = None
    # Ensemble only: each member's weekly and whole-run totals, and the members' median fields.
    member_weekly: np.ndarray | None = None  # (week, member, lat, lon)
    member_total: np.ndarray | None = None  # (member, lat, lon)
    median: dict[str, np.ndarray] = field(default_factory=dict)  # daily (day, ...), weekly (week, ...), total
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
    def member_count(self) -> int:
        return 1 if self.members is None else int(self.members.shape[0])

    def by_day(self) -> np.ndarray:
        """Rain shaped (day, member, lat, lon), so metrics run once over every member; a single run has one."""
        if self.members is None:
            return self.daily[:, None]
        return np.moveaxis(self.members, 0, 1)

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
        float(config.extreme_rain_mm),
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
    extreme_rain_mm: float = 50.0,
) -> PreparedRun:
    directory = Path(run_dir)
    manifest = json.loads((directory / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    with xr.open_dataset(directory / manifest.get("artifact", "rainfall.nc"), engine="scipy") as dataset:
        dataset = dataset.load()
    members, daily = _split_members(dataset)
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
        extreme_rain_mm=extreme_rain_mm,
        members=members,
    )
    by_day = run.by_day()
    run.windows = metrics.weekly_windows(run.day_count)
    run.weekly = metrics.window_sums(daily, run.windows).astype(np.float32)
    if members is not None:
        run.member_weekly = metrics.window_sums(by_day, run.windows).astype(np.float32)
        run.member_total = np.nansum(members, axis=1).astype(np.float32)
        run.median = {
            "daily": np.nanmedian(members, axis=0).astype(np.float32),
            "weekly": np.nanmedian(run.member_weekly, axis=1).astype(np.float32),
            "total": np.nanmedian(run.member_total, axis=0).astype(np.float32),
        }
    # Counts are found in each member and then averaged: spells in the mean of the members would
    # be smoothed away, since averaging rarely leaves 5 days in a row under 1 mm.
    run.outlook = {
        key: np.nanmean(value, axis=0).astype(np.float32)
        for key, value in metrics.outlook_metrics(
            by_day,
            wet_threshold_mm=wet_threshold_mm,
            dry_spell_min_days=dry_spell_min_days,
            wet_spell_min_days=wet_spell_min_days,
        ).items()
    }
    run.indicator_daily, run.indicator_weekly = _indicator_fields(
        by_day,
        run.windows,
        wet_threshold_mm=wet_threshold_mm,
        dry_spell_min_days=dry_spell_min_days,
        wet_spell_min_days=wet_spell_min_days,
        extreme_rain_mm=extreme_rain_mm,
    )
    run.outlook["extreme_rain"] = _any_exceed_percent(by_day, extreme_rain_mm)
    run.onset = _ensemble_onset(run, by_day)
    run.outlook["onset"] = run.onset.day.astype(np.float32)
    weights = geomask.area_weights(areas)
    run.land_weights = weights.land
    flat = by_day.reshape(run.day_count, run.member_count, -1).astype(float)
    for level in ("region", "district"):
        names, matrix = weights.matrices[level]
        run.area_matrices[level] = (names, matrix)
        sums = matrix.sum(axis=1)
        area_daily = np.divide(
            flat @ matrix.T, sums, out=np.full((run.day_count, run.member_count, len(names)), np.nan), where=sums > 0
        )
        run.area_onset[level] = _ensemble_onset(run, area_daily).day
    return run


def _split_members(dataset: xr.Dataset) -> tuple[np.ndarray | None, np.ndarray]:
    """(members, daily): members (member, day, lat, lon) for an ensemble (None for one run); daily is their mean."""
    precip = dataset["precip"]
    if "member" not in precip.dims:
        return None, np.asarray(precip.values, dtype=np.float32)
    values = np.asarray(precip.transpose("member", "lead_day", "latitude", "longitude").values, dtype=np.float32)
    if values.shape[0] == 1:
        return None, values[0]
    return values, np.nanmean(values, axis=0).astype(np.float32)


def _any_exceed_percent(by_day: np.ndarray, threshold_mm: float) -> np.ndarray:
    """% of members with at least one day at or above ``threshold_mm`` (0/100 for a single run)."""
    hit = metrics.wet_mask(by_day, threshold_mm).any(axis=0)
    missing = np.all(np.isnan(by_day), axis=0)
    return np.nanmean(np.where(missing, np.nan, hit * 100.0), axis=0).astype(np.float32)


def _ensemble_onset(run: PreparedRun, by_day: np.ndarray) -> metrics.Onset:
    """Onset per member, summarised by the median day with "no onset" ranked latest.

    A place has an onset when at least half of the members do. ``guard_days`` is recomputed from
    the median day, so "provisional" still means the dry-spell check ran past the forecast.
    """
    result = _onset(run, by_day)
    if by_day.shape[1] == 1:
        return metrics.Onset(
            day=result.day[0],
            rain_mm=result.rain_mm[0],
            longest_dry_after=result.longest_dry_after[0],
            guard_days=result.guard_days[0],
        )
    day = median_onset_day(result.day)
    guard = float(run.onset_rule["guard_days"])
    covered = np.where(day > 0, np.minimum(guard, run.day_count - day + 1), np.nan)
    empty = np.full(day.shape, np.nan)
    return metrics.Onset(day=day, rain_mm=empty, longest_dry_after=empty, guard_days=covered)


def median_onset_day(days: np.ndarray) -> np.ndarray:
    """Median over axis 0 of onset days (0 = none, NaN = no data); none ranks last, and a median of none is 0."""
    missing = np.all(np.isnan(days), axis=0)
    ranked = np.where(np.nan_to_num(days) > 0, days, np.inf)
    median = np.quantile(ranked, 0.5, axis=0, method="lower")
    return np.where(missing, np.nan, np.where(np.isfinite(median), median, 0.0))


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
    by_day: np.ndarray,
    windows: list[metrics.Window],
    *,
    wet_threshold_mm: float,
    dry_spell_min_days: int,
    wet_spell_min_days: int,
    extreme_rain_mm: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Daily shares and weekly counts per indicator, from rain shaped (day, member, lat, lon).

    Each member gives a 100/0 flag per day and the daily field is their mean: the % of members
    (just 100/0 for a single run). Spells are found over each member's full run, so a week's
    dry-spell days are the days of that week inside a spell, and the weekly counts (averaged over
    members) add up to the outlook total. Extreme rain per week is instead the % of members with
    at least one day over the threshold in that week.
    """
    wet = metrics.wet_mask(by_day, wet_threshold_mm)
    dry = metrics.dry_mask(by_day, wet_threshold_mm)
    flags = {
        "rainy_days": wet,
        "dry_spell_days": metrics.spell_day_mask(dry, dry_spell_min_days),
        "wet_spell_days": metrics.spell_day_mask(wet, wet_spell_min_days),
        "extreme_rain": metrics.wet_mask(by_day, extreme_rain_mm),
    }
    missing = np.isnan(by_day)
    per_day: dict[str, np.ndarray] = {}
    per_week: dict[str, np.ndarray] = {}
    for key, flag in flags.items():
        counts = np.where(missing, np.nan, flag.astype(np.float32))
        per_day[key] = (np.nanmean(counts, axis=1) * 100).astype(np.float32)
        weekly = metrics.window_sums(counts, windows)
        if key == "extreme_rain":
            weekly = np.where(np.isnan(weekly), np.nan, (weekly > 0) * 100.0)
        per_week[key] = np.nanmean(weekly, axis=1).astype(np.float32)
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
    extreme_rain_mm: float = 50.0
    member_count: int = 1

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
                cached.extreme_rain_mm, cached.member_count,
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
        float(config.extreme_rain_mm),
        int(manifest.get("members") or 1),
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
    _, daily = _split_members(dataset)
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
        "members": run.member_count,
        # A made-up demo run (not model output), so the viewer can label it.
        "synthetic": bool(manifest.get("synthetic")),
        "init_time": run.init_time,
        "ingested_at": manifest.get("ingested_at"),
        "provenance": manifest.get("provenance") or {},
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
        "guidance": _guidance(run.member_count),
    }


# --------------------------------------------------------------------------- layers and tiles


@dataclass(frozen=True)
class LayerSelection:
    layer: str
    aggregation: str
    index: int  # day (daily), week (weekly) or 1 (total)
    # Ensemble rainfall only: "mean", "median" or "chance" (with ``threshold`` mm); else "default".
    stat: str = "default"
    threshold: float | None = None


def model_settings(settings: Settings, model: str | None) -> Settings:
    """Settings for the requested model, as an API error when the id is unknown."""
    try:
        return for_model(settings, model)
    except UnknownModelError as exc:
        raise InvalidSubseasonalLayerError(str(exc)) from exc


def list_models(settings: Settings) -> dict[str, Any]:
    """The model picker: every configured model, with its newest run if it has one."""
    config = settings.subseasonal
    ids = list(config.models) or [config.source_id]
    if config.source_id not in ids:
        ids.insert(0, config.source_id)
    models = []
    for model_id in ids:
        scoped = for_model(settings, model_id)
        run_ids = list_run_ids(scoped)
        active = _active_run_id(scoped) if run_ids else None
        manifest: dict[str, Any] = {}
        if active:
            try:
                manifest = json.loads((source_dir(scoped) / active / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        spec = config.models.get(model_id)
        models.append(
            {
                "id": model_id,
                "label": scoped.subseasonal.model_label,
                "source_label": scoped.subseasonal.source_label,
                "description": spec.description if spec else None,
                "available": bool(run_ids),
                "active_run_id": active,
                "init_time": manifest.get("init_time"),
                "members": int(manifest.get("members") or 1) if manifest else None,
                "synthetic": bool(manifest.get("synthetic")),
            }
        )
    return {"default_model": config.source_id, "models": models}


def resolve_selection(
    run: PreparedRun,
    layer: str,
    aggregation: str | None,
    index: int | None,
    stat: str | None = None,
    threshold: float | None = None,
) -> LayerSelection:
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
    if layer_key != "rainfall" or run.member_count == 1:
        # Only ensemble rainfall has a choice of statistic; anything asked of other layers is ignored.
        return LayerSelection(layer_key, aggregation_key, resolved_index)
    stat_key = str(stat or "mean").strip().lower()
    if stat_key not in RAIN_STATS:
        raise InvalidSubseasonalLayerError(f"Statistic must be one of: {', '.join(RAIN_STATS)}.")
    if stat_key != "chance":
        return LayerSelection(layer_key, aggregation_key, resolved_index, stat_key)
    allowed_thresholds = CHANCE_THRESHOLDS[aggregation_key]
    amount = float(threshold) if threshold is not None else allowed_thresholds[0]
    if amount not in allowed_thresholds:
        raise InvalidSubseasonalLayerError(
            f"Threshold for {aggregation_key} chance must be one of: {', '.join(_fmt(value) for value in allowed_thresholds)} mm."
        )
    return LayerSelection(layer_key, aggregation_key, resolved_index, stat_key, amount)


def onset_countdown(onset_day: np.ndarray, day: int) -> np.ndarray:
    """Days from ``day`` until onset (<= 0: already started), NO_ONSET where none is forecast."""
    values = np.where(onset_day > 0, onset_day - day, legends.NO_ONSET)
    return np.where(np.isnan(onset_day), np.nan, values).astype(np.float32)


def layer_field(run: PreparedRun, selection: LayerSelection) -> np.ndarray:
    if selection.layer == "onset" and selection.aggregation == "daily":
        return onset_countdown(run.outlook["onset"], selection.index)
    if selection.layer == "rainfall" and selection.stat == "median":
        field_values = run.median[selection.aggregation]
        return field_values if selection.aggregation == "total" else field_values[selection.index - 1]
    if selection.layer == "rainfall" and selection.stat == "chance":
        return _chance_field(run, selection)
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


def _chance_field(run: PreparedRun, selection: LayerSelection) -> np.ndarray:
    """% of members with at least ``selection.threshold`` mm over the selected day, week or whole run."""
    assert run.members is not None and run.member_weekly is not None and run.member_total is not None
    if selection.aggregation == "daily":
        amounts = run.members[:, selection.index - 1]
    elif selection.aggregation == "weekly":
        amounts = run.member_weekly[selection.index - 1]
    else:
        amounts = run.member_total
    missing = np.all(np.isnan(amounts), axis=0)
    share = np.mean(np.nan_to_num(amounts, nan=-1.0) >= float(selection.threshold or 0), axis=0) * 100
    return np.where(missing, np.nan, share).astype(np.float32)


def get_layer(
    settings: Settings,
    *,
    layer: str,
    aggregation: str | None,
    index: int | None,
    run_id: str | None,
    stat: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    run = load_run(settings, run_id)
    selection = resolve_selection(run, layer, aggregation, index, stat, threshold)
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
    params: dict[str, Any] = {
        "model": settings.subseasonal.source_id,
        "run_id": run.run_id,
        "layer": selection.layer,
        "aggregation": selection.aggregation,
        "index": selection.index,
    }
    if selection.stat != "default":
        params["stat"] = selection.stat
    if selection.threshold is not None:
        params["threshold"] = _fmt(selection.threshold)
    query = urlencode(params)
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
        "statistic": _statistic(run, selection),
        "members": run.member_count,
        "threshold_mm": selection.threshold,
        **_timeline(run, selection),
        "chance_thresholds": list(CHANCE_THRESHOLDS[selection.aggregation]) if selection.layer == "rainfall" and run.member_count > 1 else [],
    }


def _timeline(run: PreparedRun, selection: LayerSelection) -> dict[str, Any]:
    """National (land-weighted) value of the chosen ensemble statistic for every day or week.

    The dock timeline draws these instead of the national mean rainfall, so its bars follow the
    Median / Chance switch. Mean (and every other layer) keeps the run's own national means.
    """
    if selection.layer != "rainfall" or selection.stat not in {"median", "chance"} or selection.aggregation == "total":
        return {"timeline": None, "timeline_unit": None}
    count = run.day_count if selection.aggregation == "daily" else len(run.windows)
    weights = run.land_weights
    assert weights is not None
    total = float(weights.sum())
    values: list[float | None] = []
    for position in range(1, count + 1):
        field_values = layer_field(run, replace(selection, index=position))
        finite = np.isfinite(field_values) & (weights > 0)
        values.append(_round(float((field_values[finite] * weights[finite]).sum()) / total) if total > 0 else None)
    return {"timeline": values, "timeline_unit": "%" if selection.stat == "chance" else "mm"}


def _statistic(run: PreparedRun, selection: LayerSelection) -> str:
    """How the members are combined in this field: one run, the ensemble mean or median, or a share."""
    if run.member_count == 1:
        return "single"
    if selection.layer == "rainfall":
        return "probability" if selection.stat == "chance" else selection.stat
    if selection.layer == "onset":
        return "median"
    if selection.layer == "extreme_rain" or selection.aggregation == "daily":
        return "probability"
    return "mean"


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
    if selection.stat == "chance":
        return legends.RAIN_CHANCE
    if selection.layer == "extreme_rain":
        if run.member_count > 1:
            return legends.EXTREME_RAIN_PROBABILITY
        return _extreme_flag_legend(run.extreme_rain_mm, "day" if selection.aggregation == "daily" else "window")
    if run.member_count > 1 and selection.aggregation == "daily" and selection.layer in ENSEMBLE_DAILY_LEGENDS:
        return ENSEMBLE_DAILY_LEGENDS[selection.layer]
    return LAYER_LEGENDS[(selection.layer, selection.aggregation)]


@lru_cache(maxsize=8)
def _extreme_flag_legend(threshold_mm: float, period: str) -> legends.Legend:
    return legends.extreme_rain_flag(threshold_mm, period=period)


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
    stat: str | None = None,
    threshold: float | None = None,
) -> bytes:
    run = load_run(settings, run_id)
    selection = resolve_selection(run, layer, aggregation, index, stat, threshold)
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
    stat: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    run = load_run(settings, run_id)
    level_key = _resolve_level(level)
    selection = resolve_selection(run, layer, aggregation, index, stat, threshold)
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
    member_series = None if run.members is None else run.members[:, :, row, col].astype(float)
    cell_weights = np.zeros(run.latitudes.size * run.longitudes.size)
    cell_weights[row * run.longitudes.size + col] = 1.0
    payload = _series_payload(run, series, member_series)
    _attach_local_chances(run, payload, cell_weights)
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
        **payload,
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
    member_series = None
    if run.members is not None:
        member_series = run.members.reshape(run.member_count, run.day_count, -1).astype(float) @ weights / total_weight
    region = None
    if level_key == "district":
        region = run.areas.district_regions[run.areas.district_names.index(names[position])]
    else:
        region = names[position]
    cells_touched = int(np.count_nonzero(weights))
    payload = _series_payload(run, series, member_series)
    _attach_local_chances(run, payload, weights)
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
        **payload,
    }


def _attach_local_chances(run: PreparedRun, payload: dict[str, Any], cell_weights: np.ndarray) -> None:
    """Chances worked out cell by cell, then averaged over the place: the same figures as the map.

    Testing the area-mean rain against an amount would hide a local storm (a 60 mm cell barely
    moves a region's mean), so a region would show 0% where the map shows 40%. Instead each grid
    cell gets the share of members reaching the amount and the place gets the weighted average of
    those shares, exactly what the map and its hover show. For one point it is that cell's share.

    Adds per day ``chance`` (daily amounts), ``extreme_prob``, ``extreme_by_date`` and
    ``total_by_date`` (whole-run amounts reached by that date); per week ``chance`` and
    ``extreme_prob``; and ``total_chance`` for the whole run. A single run gives 0/100 per cell,
    so for an area the figures are the share of the area.
    """
    cells = np.flatnonzero(cell_weights > 0)
    if not cells.size:
        return
    weights = cell_weights[cells].astype(float)
    weights /= weights.sum()
    values = run.by_day().reshape(run.day_count, run.member_count, -1)[:, :, cells].astype(float)  # (day, member, cell)
    finite = np.isfinite(values)
    rain = np.where(finite, values, 0.0)

    def share(mask: np.ndarray) -> np.ndarray:
        """% of members per cell (member axis -2), averaged over the cells with the place's weights."""
        return np.mean(mask, axis=-2) @ weights * 100

    def keyed(amounts: tuple[float, ...], results: list[np.ndarray]) -> list[dict[str, float | None]]:
        return [{_fmt(amount): _round(float(result[position]), 0) for amount, result in zip(amounts, results)} for position in range(results[0].shape[0])]

    daily_amounts = CHANCE_THRESHOLDS["daily"]
    weekly_amounts = CHANCE_THRESHOLDS["weekly"]
    total_amounts = CHANCE_THRESHOLDS["total"]
    weekly_rain = metrics.window_sums(rain, run.windows)  # (week, member, cell)
    running = np.cumsum(rain, axis=0)  # (day, member, cell)
    extreme = rain >= run.extreme_rain_mm
    extreme_by_date = np.maximum.accumulate(extreme, axis=0)
    extreme_weekly = metrics.window_sums(extreme.astype(float), run.windows) > 0

    day_chance = keyed(daily_amounts, [share(rain >= amount) for amount in daily_amounts])
    by_date = keyed(total_amounts, [share(running >= amount) for amount in total_amounts])
    extreme_daily = share(extreme)
    extreme_cumulative = share(extreme_by_date)
    # Rain-day and in-spell chances on the same footing, so the drawer's % charts match the day maps.
    wet = rain >= run.wet_threshold_mm
    dry = finite & (rain < run.wet_threshold_mm)
    wet_share = share(wet)
    dry_spell_share = share(metrics.spell_day_mask(dry, run.dry_spell_min_days))
    wet_spell_share = share(metrics.spell_day_mask(wet, run.wet_spell_min_days))
    ensemble = run.member_count > 1
    for position, day in enumerate(payload["days"]):
        if ensemble:
            day["wet_prob"] = _round(float(wet_share[position]), 0)
            day["spell_prob"] = {"dry": _round(float(dry_spell_share[position]), 0), "wet": _round(float(wet_spell_share[position]), 0)}
        day["chance"] = day_chance[position]
        day["total_by_date"] = by_date[position]
        day["extreme_prob"] = _round(float(extreme_daily[position]), 0)
        day["extreme_by_date"] = _round(float(extreme_cumulative[position]), 0)
    week_chance = keyed(weekly_amounts, [share(weekly_rain >= amount) for amount in weekly_amounts])
    extreme_week = share(extreme_weekly)
    for position, week in enumerate(payload["weeks"]):
        week["chance"] = week_chance[position]
        week["extreme_prob"] = _round(float(extreme_week[position]), 0)
    payload["total_chance"] = {_fmt(amount): _round(float(share(running[-1] >= amount)), 0) for amount in total_amounts}


def _series_payload(run: PreparedRun, series: np.ndarray, member_series: np.ndarray | None = None) -> dict[str, Any]:
    """Daily and weekly series, outlook counts, spells and onset for one place.

    ``series`` is the single run or the ensemble mean. With ``member_series`` (member, day) the
    counts are averaged over members, each day carries the members' spread and the share in a
    spell, and spells are the runs of days where at least half of the members are in one.
    """
    if member_series is not None and member_series.shape[0] > 1:
        return _ensemble_series_payload(run, np.asarray(series, dtype=float), np.asarray(member_series, dtype=float))
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
    extreme = metrics.wet_mask(series, run.extreme_rain_mm)
    days = []
    for position, value in enumerate(series):
        payload = _day_payload(run, position + 1, _round(float(value)))
        payload["wet"] = bool(math.isfinite(value) and value >= run.wet_threshold_mm)
        payload["spell"] = "dry" if dry_days[position] else ("wet" if wet_days[position] else None)
        payload["extreme_prob"] = 100.0 if extreme[position] else 0.0
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
            "extreme_days": float(extreme.sum()),
        },
        "onset": _onset_payload(run, series),
        "spells": [_spell_payload(run, spell, series) for spell in spells],
        "thresholds": _thresholds(run),
        "guidance": GUIDANCE,
        "members": 1,
    }


def _ensemble_series_payload(run: PreparedRun, series: np.ndarray, member_series: np.ndarray) -> dict[str, Any]:
    by_day = member_series.T  # (day, member)
    member_count = by_day.shape[1]
    weekly = metrics.window_sums(series, run.windows)
    member_weekly = metrics.window_sums(by_day, run.windows)  # (week, member)
    outlook = metrics.outlook_metrics(
        by_day,
        wet_threshold_mm=run.wet_threshold_mm,
        dry_spell_min_days=run.dry_spell_min_days,
        wet_spell_min_days=run.wet_spell_min_days,
    )
    wet = metrics.wet_mask(by_day, run.wet_threshold_mm)
    in_dry = metrics.spell_day_mask(metrics.dry_mask(by_day, run.wet_threshold_mm), run.dry_spell_min_days)
    in_wet = metrics.spell_day_mask(wet, run.wet_spell_min_days)
    extreme = metrics.wet_mask(by_day, run.extreme_rain_mm)
    wet_share = wet.mean(axis=1) * 100
    dry_share = in_dry.mean(axis=1) * 100
    wet_spell_share = in_wet.mean(axis=1) * 100
    extreme_share = extreme.mean(axis=1) * 100
    consensus_dry = dry_share >= CONSENSUS_PERCENT
    consensus_wet = wet_spell_share >= CONSENSUS_PERCENT
    # Each member's days in a dry spell, in a wet spell and with rain, week by week (week, member):
    # counted per member like the map's weekly layers, so the drawer can show their spread as box plots.
    member_day_counts = {
        "dry_spell": metrics.window_sums(in_dry.astype(float), run.windows),
        "wet_spell": metrics.window_sums(in_wet.astype(float), run.windows),
        "rainy": metrics.window_sums(wet.astype(float), run.windows),
    }
    days = []
    for position, value in enumerate(series):
        payload = _day_payload(run, position + 1, _round(float(value)))
        payload["wet"] = bool(wet_share[position] >= CONSENSUS_PERCENT)
        payload["spell"] = "dry" if consensus_dry[position] else ("wet" if consensus_wet[position] else None)
        payload["spread"] = _spread(by_day[position])
        payload["members"] = _member_values(by_day[position])
        payload["wet_prob"] = _round(float(wet_share[position]), 0)
        payload["spell_prob"] = {"dry": _round(float(dry_share[position]), 0), "wet": _round(float(wet_spell_share[position]), 0)}
        payload["extreme_prob"] = _round(float(extreme_share[position]), 0)
        days.append(payload)
    weeks = []
    for position, (window, value) in enumerate(zip(run.windows, weekly)):
        payload = _window_payload(run, window, _round(float(value)))
        payload["spread"] = _spread(member_weekly[position])
        payload["members"] = _member_values(member_weekly[position])
        payload["day_counts"] = {key: _spread(counts[position]) for key, counts in member_day_counts.items()}
        weeks.append(payload)
    max_day = int(np.nanargmax(series)) + 1 if series.size else None
    spells = []
    for kind, consensus, member_mask in (("dry", consensus_dry, in_dry), ("wet", consensus_wet, in_wet)):
        for start, end in _true_runs(consensus):
            spell = metrics.Spell(
                kind=kind, start_day=start, end_day=end, open_start=start == 1, open_end=end == run.day_count
            )
            payload = _spell_payload(run, spell, series)
            # Members with any spell day inside this stretch.
            payload["probability"] = _round(float(member_mask[start - 1 : end].any(axis=0).mean() * 100), 0)
            spells.append(payload)
    spells.sort(key=lambda item: item["start_day"])
    return {
        "run_id": run.run_id,
        "init_time": run.init_time,
        "unit": "mm",
        "days": days,
        "weeks": weeks,
        "metrics": {
            "total_mm": _round(float(np.nanmean(outlook["total"]))),
            "rainy_days": int(round(float(np.nanmean(outlook["rainy_days"])))),
            "dry_spell_days": int(round(float(np.nanmean(outlook["dry_spell_days"])))),
            "wet_spell_days": int(round(float(np.nanmean(outlook["wet_spell_days"])))),
            "max_day_mm": _round(float(np.nanmax(series))) if series.size else None,
            "max_day": max_day,
            "extreme_days": _round(float(extreme.sum(axis=0).mean())),
        },
        "onset": _ensemble_onset_payload(run, by_day),
        "spells": spells,
        "thresholds": _thresholds(run),
        "guidance": _guidance(member_count),
        "members": member_count,
        "member_totals": _member_values(np.nansum(by_day, axis=0)),
    }


def _member_values(values: np.ndarray) -> list[float | None]:
    """Each member's amount (mm, 0.1 precision), so the client can take medians and chances itself."""
    return [_round(float(value)) for value in values]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) 1-based inclusive day ranges of consecutive True values."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position, flag in enumerate(mask, start=1):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            runs.append((start, position - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _spread(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {}
    quantiles = np.percentile(finite, SPREAD_QUANTILES)
    payload: dict[str, float | None] = {f"p{q}": _round(float(v)) for q, v in zip(SPREAD_QUANTILES, quantiles)}
    payload.update(min=_round(float(finite.min())), max=_round(float(finite.max())), mean=_round(float(finite.mean())))
    return payload


def _spell_payload(run: PreparedRun, spell: metrics.Spell, series: np.ndarray) -> dict[str, Any]:
    return {
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


def _ensemble_onset_payload(run: PreparedRun, by_day: np.ndarray) -> dict[str, Any] | None:
    """The members' median onset day, if at least half of them have one, and the share that do."""
    result = _onset(run, by_day)
    found = np.nan_to_num(result.day) > 0
    day = float(median_onset_day(result.day))
    if not math.isfinite(day) or day <= 0:
        return None
    guard_days = int(min(float(run.onset_rule["guard_days"]), run.day_count - day + 1))
    return {
        "day": int(day),
        "date": metrics.rain_day(run.init_date, int(day)),
        "rain_mm": _round(float(np.nanmedian(result.rain_mm[found]))) if found.any() else None,
        "longest_dry_after": None,
        "guard_days": guard_days,
        "provisional": guard_days < int(run.onset_rule["guard_days"]),
        "probability": _round(float(found.mean() * 100), 0),
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
        "extreme_rain_mm": float(run.extreme_rain_mm),
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
    if selection.stat == "median":
        label = "Rainfall (ensemble median)"
    elif selection.stat == "chance":
        label = f"Chance of {_fmt(float(selection.threshold or 0))} mm or more"
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
        extreme=_fmt(float(run.extreme_rain_mm)),
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
