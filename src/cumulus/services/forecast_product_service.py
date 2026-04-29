"""Cumulus-managed seasonal forecast artifact products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import shutil
import struct
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import urlencode
import zlib

import numpy as np
import pandas as pd
import xarray as xr

from cumulus.api.errors import (
    ForecastProductArtifactsNotAvailableError,
    ForecastProductIncompleteError,
    InvalidForecastProductSelectionError,
    InvalidForecastProductThemeError,
)
from cumulus.settings import ForecastProductPairSourceConfig, ForecastProductSourceConfig, SeasonalProfileConfig, Settings
from cumulus.utils.io import ensure_directory

TILE_SIZE = 256
TILE_GEOMETRY_MASK_SIZE = 64
ViewMode = Literal["probability", "deterministic"]
SEASON_BASED_THEMES = frozenset({"onset", "cessation", "early_dry_spell", "late_dry_spell"})
SUBSEASON_BASED_THEMES = frozenset({"rainfall_amount", "rainy_days"})
SUBSEASON_DISPLAY_ORDER = ("MAM", "AMJ", "MJJ", "JJA", "JAS", "SON")
GHANA_PRODUCT_MASK_ZONE = "ghana"
CALENDAR_SUBSEASON_MONTHS: dict[str, tuple[int, int, int]] = {
    "MAM": (3, 4, 5),
    "AMJ": (4, 5, 6),
    "MJJ": (5, 6, 7),
    "JJA": (6, 7, 8),
    "JAS": (7, 8, 9),
    "SON": (9, 10, 11),
}
DAILY_DERIVED_THEMES = frozenset({"onset", "cessation", "late_dry_spell", "rainy_days"})
PROFILE_DERIVED_ONSET_PROFILES = frozenset({"southern_minor"})
FINAL_PRODUCT_ONLY_THEMES = frozenset({"rainfall_amount"})
PROBABILITY_CODES = ("PB", "PN", "PA")
DEFAULT_FINAL_PRODUCT_SELECTOR = "__default__"
FINAL_SUBSEASON_TOKEN_MAP = {
    "maraprmay": "MAM",
    "aprmayjun": "AMJ",
    "mayjunjul": "MJJ",
    "junjulaug": "JJA",
    "julaugsep": "JAS",
    "sepoctnov": "SON",
    "septoctnov": "SON",
}
FINAL_THEME_TOKEN_MAP = {
    "prcponset": "onset",
    "onset": "onset",
    "prcpdryspellonset": "early_dry_spell",
    "dryspellonset": "early_dry_spell",
    "earlydryspell": "early_dry_spell",
    "earlydryspellonset": "early_dry_spell",
    "prcpcessation": "cessation",
    "cessation": "cessation",
    "prcpdryspellcessation": "late_dry_spell",
    "dryspellcessation": "late_dry_spell",
    "latedryspell": "late_dry_spell",
}
FORECAST_PRODUCT_SOUTHERN_NATIVE_REGIONS = frozenset(
    {
        "ahafo",
        "ashanti",
        "bono",
        "bono east",
        "central",
        "eastern",
        "greater accra",
        "oti",
        "volta",
        "western",
        "western north",
    }
)
FORECAST_PRODUCT_NORTHERN_NATIVE_REGIONS = frozenset(
    {
        "north east",
        "northern",
        "savannah",
        "upper east",
        "upper west",
    }
)
DETERMINISTIC_REFERENCE_COLOR_RAMP = (
    (0.0, "#440154"),
    (0.25, "#3b528b"),
    (0.5, "#21918c"),
    (0.75, "#8fd744"),
    (1.0, "#fde725"),
)
FINAL_DETERMINISTIC_TILE_ALPHA = 248
FALLBACK_DETERMINISTIC_TILE_ALPHA = 188
FINAL_PROBABILITY_TILE_ALPHA_MIN = 96
FINAL_PROBABILITY_TILE_ALPHA_MAX = 235
FALLBACK_PROBABILITY_TILE_ALPHA_MIN = 72
FALLBACK_PROBABILITY_TILE_ALPHA_MAX = 176


@dataclass(frozen=True)
class ProductThemeSpec:
    theme: str
    theme_label: str
    description: str
    deterministic_unit: str
    deterministic_color_ramp: tuple[tuple[float, str], ...]
    probability_categories: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class ForecastProductSelection:
    theme: str
    season_profile: str | None
    season_label: str | None
    subseason: str | None
    subseason_label: str | None
    requires_season: bool
    requires_subseason: bool


@dataclass(frozen=True)
class PreparedDeterministicProduct:
    theme: str
    theme_label: str
    season_profile: str | None
    season_label: str | None
    subseason: str | None
    subseason_label: str | None
    product_id: str
    forecast_year: int
    valid_time: datetime
    generated_at: datetime
    refresh_interval_seconds: int
    freshness_threshold_hours: int
    source_label: str
    source_run_id: str
    generation_backend: str
    source_artifact_type: str
    is_low_resolution_fallback: bool
    lower_bound: float
    upper_bound: float
    legend_ticks: tuple[float, ...]
    color_ramp: tuple[tuple[float, str], ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    values: np.ndarray
    unit: str
    mask_zone: str
    preview_url: str | None


@dataclass(frozen=True)
class PreparedProbabilityProduct:
    theme: str
    theme_label: str
    season_profile: str | None
    season_label: str | None
    subseason: str | None
    subseason_label: str | None
    product_id: str
    forecast_year: int
    valid_time: datetime
    generated_at: datetime
    refresh_interval_seconds: int
    freshness_threshold_hours: int
    source_label: str
    source_run_id: str
    generation_backend: str
    source_artifact_type: str
    is_low_resolution_fallback: bool
    category_codes: tuple[str, ...]
    legend: tuple[dict[str, Any], ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    probabilities: np.ndarray
    mask_zone: str
    preview_url: str | None


@dataclass(frozen=True)
class DerivedDailyForecastGrid:
    forecast_year: int
    valid_time: datetime
    latitudes: np.ndarray
    longitudes: np.ndarray
    deterministic_values: np.ndarray
    probabilities: np.ndarray


THEME_SPECS: dict[str, ProductThemeSpec] = {
    "onset": ProductThemeSpec(
        theme="onset",
        theme_label="Onset Date",
        description="Seasonal onset timing forecast across Ghana.",
        deterministic_unit="day_of_year",
        deterministic_color_ramp=DETERMINISTIC_REFERENCE_COLOR_RAMP,
        probability_categories=(
            ("PB", "Early", "Earlier than the climatological timing window.", "#2f8f86"),
            ("PN", "Near-Normal", "Within the climatological timing window.", "#b8b9b4"),
            ("PA", "Late", "Later than the climatological timing window.", "#b47a34"),
        ),
    ),
    "early_dry_spell": ProductThemeSpec(
        theme="early_dry_spell",
        theme_label="Early-Season Dry Spell",
        description="Early-season dry-spell duration forecast across Ghana.",
        deterministic_unit="days",
        deterministic_color_ramp=DETERMINISTIC_REFERENCE_COLOR_RAMP,
        probability_categories=(
            ("PB", "Short", "Shorter early-season dry-spell duration.", "#2f8f86"),
            ("PN", "Near-Normal", "Near the climatological dry-spell duration.", "#b8b9b4"),
            ("PA", "Long", "Longer early-season dry-spell duration.", "#b47a34"),
        ),
    ),
    "cessation": ProductThemeSpec(
        theme="cessation",
        theme_label="Cessation Date",
        description="Seasonal cessation timing forecast across Ghana.",
        deterministic_unit="day_of_year",
        deterministic_color_ramp=DETERMINISTIC_REFERENCE_COLOR_RAMP,
        probability_categories=(
            ("PB", "Early", "Earlier than the climatological timing window.", "#b47a34"),
            ("PN", "Near-Normal", "Within the climatological timing window.", "#b8b9b4"),
            ("PA", "Late", "Later than the climatological timing window.", "#2f8f86"),
        ),
    ),
    "late_dry_spell": ProductThemeSpec(
        theme="late_dry_spell",
        theme_label="Late-Season Dry Spell",
        description="Late-season dry-spell duration forecast across Ghana.",
        deterministic_unit="days",
        deterministic_color_ramp=DETERMINISTIC_REFERENCE_COLOR_RAMP,
        probability_categories=(
            ("PB", "Short", "Shorter late-season dry-spell duration.", "#2f8f86"),
            ("PN", "Near-Normal", "Near the climatological dry-spell duration.", "#b8b9b4"),
            ("PA", "Long", "Longer late-season dry-spell duration.", "#b47a34"),
        ),
    ),
    "rainfall_amount": ProductThemeSpec(
        theme="rainfall_amount",
        theme_label="Seasonal Rainfall Total",
        description="Seasonal rainfall amount forecast across Ghana.",
        deterministic_unit="mm",
        deterministic_color_ramp=DETERMINISTIC_REFERENCE_COLOR_RAMP,
        probability_categories=(
            ("PB", "BELOW-AVERAGE", "Below the climatological rainfall amount.", "#b47a34"),
            ("PN", "NEAR-AVERAGE", "Near the climatological rainfall amount.", "#b8b9b4"),
            ("PA", "ABOVE-AVERAGE", "Above the climatological rainfall amount.", "#2f8f86"),
        ),
    ),
    "rainy_days": ProductThemeSpec(
        theme="rainy_days",
        theme_label="Number of Rainy Days",
        description="Seasonal rainy-day count forecast across Ghana.",
        deterministic_unit="days",
        deterministic_color_ramp=DETERMINISTIC_REFERENCE_COLOR_RAMP,
        probability_categories=(
            ("PB", "BELOW-AVERAGE", "Below the climatological rainy-day count.", "#b47a34"),
            ("PN", "NEAR-AVERAGE", "Near the climatological rainy-day count.", "#b8b9b4"),
            ("PA", "ABOVE-AVERAGE", "Above the climatological rainy-day count.", "#2f8f86"),
        ),
    ),
}


def refresh_forecast_products(settings: Settings, *, theme: str | None = None) -> dict[str, Any]:
    attempted: list[dict[str, Any]] = []
    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for selection in _refreshable_selections(settings, theme):
        for view_mode in ("probability", "deterministic"):
            attempt_payload = {
                "theme": selection.theme,
                "view_mode": view_mode,
            }
            if selection.season_profile is not None:
                attempt_payload["season_profile"] = selection.season_profile
            if selection.subseason is not None:
                attempt_payload["subseason"] = selection.subseason
            attempted.append(attempt_payload)
            try:
                manifest = _ensure_active_manifest(
                    settings,
                    selection.theme,
                    view_mode,
                    season_profile=selection.season_profile,
                    subseason=selection.subseason,
                    materialize_missing=True,
                )
            except Exception as exc:
                failed.append({**attempt_payload, "error": str(exc)})
                continue
            succeeded.append(
                {
                    **attempt_payload,
                    "product_id": manifest["product_id"],
                    "generated_at": manifest["generated_at"],
                    "manifest_path": manifest["manifest_path"],
                }
            )
    return {
        "attempted_count": len(attempted),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
}


def get_active_probability_product(
    settings: Settings,
    *,
    theme: str,
    season_profile: str | None = None,
    subseason: str | None = None,
    api_base_url: str | None = None,
) -> dict[str, Any]:
    prepared = _prepare_probability_product(settings, theme, season_profile=season_profile, subseason=subseason)
    return {
        "product_id": prepared.product_id,
        "theme": prepared.theme,
        "theme_label": prepared.theme_label,
        "season_profile": prepared.season_profile,
        "season_label": prepared.season_label,
        "subseason": prepared.subseason,
        "subseason_label": prepared.subseason_label,
        "forecast_year": prepared.forecast_year,
        "valid_time": prepared.valid_time,
        "generated_at": prepared.generated_at,
        "forecast_source": "cumulus_bridge",
        "forecast_source_label": prepared.source_label,
        "source_run_id": prepared.source_run_id,
        "generation_backend": prepared.generation_backend,
        "source_artifact_type": prepared.source_artifact_type,
        "grid_shape": _grid_shape_payload(prepared.latitudes, prepared.longitudes),
        "grid_resolution_degrees": _grid_resolution_payload(prepared.latitudes, prepared.longitudes),
        "is_low_resolution_fallback": prepared.is_low_resolution_fallback,
        "refresh_interval_seconds": prepared.refresh_interval_seconds,
        "freshness_threshold_hours": prepared.freshness_threshold_hours,
        "tile_url": _resolve_browser_url(
            api_base_url,
            _asset_query_path(
                "probability",
                "tiles/{z}/{x}/{y}.png",
                theme=prepared.theme,
                season_profile=prepared.season_profile,
                subseason=prepared.subseason,
            ),
        ),
        "preview_url": _resolve_browser_url(api_base_url, prepared.preview_url),
        "bounds": _bounds_payload(prepared.latitudes, prepared.longitudes),
        "legend": list(prepared.legend),
    }


def get_active_deterministic_product(
    settings: Settings,
    *,
    theme: str,
    season_profile: str | None = None,
    subseason: str | None = None,
    api_base_url: str | None = None,
) -> dict[str, Any]:
    prepared = _prepare_deterministic_product(settings, theme, season_profile=season_profile, subseason=subseason)
    return {
        "product_id": prepared.product_id,
        "theme": prepared.theme,
        "theme_label": prepared.theme_label,
        "season_profile": prepared.season_profile,
        "season_label": prepared.season_label,
        "subseason": prepared.subseason,
        "subseason_label": prepared.subseason_label,
        "forecast_year": prepared.forecast_year,
        "valid_time": prepared.valid_time,
        "generated_at": prepared.generated_at,
        "forecast_source": "cumulus_bridge",
        "forecast_source_label": prepared.source_label,
        "source_run_id": prepared.source_run_id,
        "generation_backend": prepared.generation_backend,
        "source_artifact_type": prepared.source_artifact_type,
        "grid_shape": _grid_shape_payload(prepared.latitudes, prepared.longitudes),
        "grid_resolution_degrees": _grid_resolution_payload(prepared.latitudes, prepared.longitudes),
        "is_low_resolution_fallback": prepared.is_low_resolution_fallback,
        "refresh_interval_seconds": prepared.refresh_interval_seconds,
        "freshness_threshold_hours": prepared.freshness_threshold_hours,
        "tile_url": _resolve_browser_url(
            api_base_url,
            _asset_query_path(
                "deterministic",
                "tiles/{z}/{x}/{y}.png",
                theme=prepared.theme,
                season_profile=prepared.season_profile,
                subseason=prepared.subseason,
            ),
        ),
        "preview_url": _resolve_browser_url(api_base_url, prepared.preview_url),
        "bounds": _bounds_payload(prepared.latitudes, prepared.longitudes),
        "unit": prepared.unit,
        "lower_bound": prepared.lower_bound,
        "upper_bound": prepared.upper_bound,
        "legend_ticks": list(prepared.legend_ticks),
        "color_ramp": [{"offset": offset, "color": color} for offset, color in prepared.color_ramp],
    }


def sample_active_probability_product(
    settings: Settings,
    *,
    theme: str,
    latitude: float,
    longitude: float,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> dict[str, Any]:
    prepared = _prepare_probability_product(settings, theme, season_profile=season_profile, subseason=subseason)
    valid_mask = np.isfinite(prepared.probabilities).any(axis=0)
    y_index, x_index = _resolve_nearest_valid_index(valid_mask, prepared.latitudes, prepared.longitudes, latitude, longitude)
    values = prepared.probabilities[:, y_index, x_index]
    dominant_index = int(np.nanargmax(values))
    legend_lookup = {item["category_code"]: item for item in prepared.legend}
    categories = []
    for idx, category_code in enumerate(prepared.category_codes):
        legend_item = legend_lookup[category_code]
        percentage = float(values[idx] * 100.0) if math.isfinite(float(values[idx])) else 0.0
        categories.append(
            {
                "category_code": category_code,
                "label": legend_item["label"],
                "hint": legend_item["hint"],
                "color": legend_item["color"],
                "percentage": round(percentage, 1),
            }
        )
    dominant = categories[dominant_index]
    return {
        "theme": prepared.theme,
        "theme_label": prepared.theme_label,
        "season_profile": prepared.season_profile,
        "season_label": prepared.season_label,
        "subseason": prepared.subseason,
        "subseason_label": prepared.subseason_label,
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "nearest_latitude": round(float(prepared.latitudes[y_index]), 4),
        "nearest_longitude": round(float(prepared.longitudes[x_index]), 4),
        "dominant_category_code": dominant["category_code"],
        "dominant_category_label": dominant["label"],
        "dominant_percentage": dominant["percentage"],
        "display_value": f"{dominant['label']} {round(dominant['percentage'])}%",
        "interpretation": f"{prepared.theme_label} leans {dominant['label'].lower()} at the nearest forecast cell.",
        "criteria_note": "Category confidence is sampled from the nearest generated forecast grid cell.",
        "category_probabilities": categories,
        "valid_time": prepared.valid_time,
        "forecast_year": prepared.forecast_year,
        "forecast_source": "cumulus_bridge",
        "forecast_source_label": prepared.source_label,
        "source_run_id": prepared.source_run_id,
        "generation_backend": prepared.generation_backend,
    }


def sample_active_deterministic_product(
    settings: Settings,
    *,
    theme: str,
    latitude: float,
    longitude: float,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> dict[str, Any]:
    prepared = _prepare_deterministic_product(settings, theme, season_profile=season_profile, subseason=subseason)
    valid_mask = np.isfinite(prepared.values)
    y_index, x_index = _resolve_nearest_valid_index(valid_mask, prepared.latitudes, prepared.longitudes, latitude, longitude)
    value = float(prepared.values[y_index, x_index])
    return {
        "theme": prepared.theme,
        "theme_label": prepared.theme_label,
        "season_profile": prepared.season_profile,
        "season_label": prepared.season_label,
        "subseason": prepared.subseason,
        "subseason_label": prepared.subseason_label,
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "nearest_latitude": round(float(prepared.latitudes[y_index]), 4),
        "nearest_longitude": round(float(prepared.longitudes[x_index]), 4),
        "value": round(value, 1),
        "display_value": _format_deterministic_display_value(prepared.theme, prepared.forecast_year, value),
        "unit": prepared.unit,
        "interpretation": _deterministic_interpretation(prepared.theme, value),
        "criteria_note": "Deterministic value is sampled from the nearest generated forecast grid cell.",
        "valid_time": prepared.valid_time,
        "forecast_year": prepared.forecast_year,
        "forecast_source": "cumulus_bridge",
        "forecast_source_label": prepared.source_label,
        "source_run_id": prepared.source_run_id,
        "generation_backend": prepared.generation_backend,
    }


def render_probability_tile(
    settings: Settings,
    *,
    theme: str,
    z: int,
    x: int,
    y: int,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> bytes:
    prepared = _prepare_probability_product(settings, theme, season_profile=season_profile, subseason=subseason)
    return _render_probability_tile_png(settings, prepared, z=z, x=x, y=y)


def render_deterministic_tile(
    settings: Settings,
    *,
    theme: str,
    z: int,
    x: int,
    y: int,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> bytes:
    prepared = _prepare_deterministic_product(settings, theme, season_profile=season_profile, subseason=subseason)
    return _render_deterministic_tile_png(settings, prepared, z=z, x=x, y=y)


def list_supported_product_themes(settings: Settings) -> list[dict[str, Any]]:
    items = []
    for theme in THEME_SPECS:
        spec = _theme_spec(theme)
        selection = _selection_metadata(settings, theme)
        items.append(
            {
                "theme": theme,
                "label": spec.theme_label,
                "title": spec.description,
                "requires_season": selection["requires_season"],
                "requires_subseason": selection["requires_subseason"],
                "enabled": selection["enabled"],
                "reason": selection["reason"],
                "seasons": selection["seasons"],
                "subseasons": selection["subseasons"],
            }
        )
    return items


def _is_source_wired(source: ForecastProductSourceConfig) -> bool:
    return source.deterministic_path is not None and source.probability_path is not None


def _prepare_probability_product(
    settings: Settings,
    theme: str,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> PreparedProbabilityProduct:
    selection = _resolve_selection(settings, theme, season_profile=season_profile, subseason=subseason)
    manifest = _ensure_active_manifest(
        settings,
        selection.theme,
        "probability",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    spec = _theme_spec(selection.theme)
    data_path = Path(str(manifest["data_path"]))
    _validate_product_dataset_for_selection(settings, selection, "probability", data_path)
    with xr.open_dataset(manifest["data_path"]) as dataset:
        data_var = dataset[list(dataset.data_vars)[0]].isel(T=0)
        category_codes = tuple(str(item) for item in data_var.coords["probability"].values.tolist())
        probabilities = np.asarray(data_var.values, dtype=float)
        latitudes = np.asarray(dataset["Y"].values, dtype=float)
        longitudes = np.asarray(dataset["X"].values, dtype=float)
        valid_time = _to_utc_datetime(data_var.coords["T"].item())
    probabilities = _clean_probability_grid(probabilities)
    probabilities = _apply_selection_spatial_mask(settings, selection, latitudes, longitudes, probabilities)
    preview_url = _manifest_preview_url(
        selection.theme,
        "probability",
        manifest,
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    return PreparedProbabilityProduct(
        theme=selection.theme,
        theme_label=spec.theme_label,
        season_profile=selection.season_profile,
        season_label=selection.season_label,
        subseason=selection.subseason,
        subseason_label=selection.subseason_label,
        product_id=manifest["product_id"],
        forecast_year=int(manifest["forecast_year"]),
        valid_time=valid_time,
        generated_at=_to_utc_datetime(manifest["generated_at"]),
        refresh_interval_seconds=int(manifest["refresh_interval_seconds"]),
        freshness_threshold_hours=int(manifest["freshness_threshold_hours"]),
        source_label=str(manifest["source_label"]),
        source_run_id=str(manifest["source_run_id"]),
        generation_backend=str(manifest["generation_backend"]),
        source_artifact_type=_manifest_source_artifact_type(manifest),
        is_low_resolution_fallback=_manifest_is_low_resolution_fallback(manifest),
        category_codes=category_codes,
        legend=tuple(_probability_legend(spec)),
        latitudes=latitudes,
        longitudes=longitudes,
        probabilities=probabilities,
        mask_zone=_selection_mask_zone(settings, selection),
        preview_url=preview_url,
    )


def _prepare_deterministic_product(
    settings: Settings,
    theme: str,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> PreparedDeterministicProduct:
    selection = _resolve_selection(settings, theme, season_profile=season_profile, subseason=subseason)
    manifest = _ensure_active_manifest(
        settings,
        selection.theme,
        "deterministic",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    spec = _theme_spec(selection.theme)
    data_path = Path(str(manifest["data_path"]))
    _validate_product_dataset_for_selection(settings, selection, "deterministic", data_path)
    with xr.open_dataset(manifest["data_path"]) as dataset:
        data_var = dataset[list(dataset.data_vars)[0]].isel(T=0)
        values = np.asarray(data_var.values, dtype=float)
        latitudes = np.asarray(dataset["Y"].values, dtype=float)
        longitudes = np.asarray(dataset["X"].values, dtype=float)
        valid_time = _to_utc_datetime(data_var.coords["T"].item())
    values = _apply_selection_spatial_mask(settings, selection, latitudes, longitudes, values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ForecastProductIncompleteError(f"Deterministic product '{selection.theme}' does not contain any finite values.")
    lower_bound = float(np.nanmin(finite))
    upper_bound = float(np.nanmax(finite))
    preview_url = _manifest_preview_url(
        selection.theme,
        "deterministic",
        manifest,
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    return PreparedDeterministicProduct(
        theme=selection.theme,
        theme_label=spec.theme_label,
        season_profile=selection.season_profile,
        season_label=selection.season_label,
        subseason=selection.subseason,
        subseason_label=selection.subseason_label,
        product_id=manifest["product_id"],
        forecast_year=int(manifest["forecast_year"]),
        valid_time=valid_time,
        generated_at=_to_utc_datetime(manifest["generated_at"]),
        refresh_interval_seconds=int(manifest["refresh_interval_seconds"]),
        freshness_threshold_hours=int(manifest["freshness_threshold_hours"]),
        source_label=str(manifest["source_label"]),
        source_run_id=str(manifest["source_run_id"]),
        generation_backend=str(manifest["generation_backend"]),
        source_artifact_type=_manifest_source_artifact_type(manifest),
        is_low_resolution_fallback=_manifest_is_low_resolution_fallback(manifest),
        lower_bound=round(lower_bound, 1),
        upper_bound=round(upper_bound, 1),
        legend_ticks=tuple(round(float(item), 1) for item in np.linspace(lower_bound, upper_bound, num=5)),
        color_ramp=spec.deterministic_color_ramp,
        latitudes=latitudes,
        longitudes=longitudes,
        values=values,
        unit=spec.deterministic_unit,
        mask_zone=_selection_mask_zone(settings, selection),
        preview_url=preview_url,
    )


def get_probability_preview_path(
    settings: Settings,
    *,
    theme: str,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> Path:
    selection = _resolve_selection(settings, theme, season_profile=season_profile, subseason=subseason)
    manifest = _ensure_active_manifest(
        settings,
        selection.theme,
        "probability",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    data_path = Path(str(manifest["data_path"]))
    _validate_product_dataset_for_selection(settings, selection, "probability", data_path)
    target = Path(str(manifest.get("preview_path") or ""))
    if not target.exists():
        raise ForecastProductArtifactsNotAvailableError("Probability preview not available for the selected forecast product.")
    return target


def get_deterministic_preview_path(
    settings: Settings,
    *,
    theme: str,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> Path:
    selection = _resolve_selection(settings, theme, season_profile=season_profile, subseason=subseason)
    manifest = _ensure_active_manifest(
        settings,
        selection.theme,
        "deterministic",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    data_path = Path(str(manifest["data_path"]))
    _validate_product_dataset_for_selection(settings, selection, "deterministic", data_path)
    target = Path(str(manifest.get("preview_path") or ""))
    if not target.exists():
        raise ForecastProductArtifactsNotAvailableError("Deterministic preview not available for the selected forecast product.")
    return target


def _generate_product_artifact(
    settings: Settings,
    source: ForecastProductSourceConfig,
    view_mode: ViewMode,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> dict[str, Any]:
    selection = _resolve_selection(settings, source.theme, season_profile=season_profile, subseason=subseason)
    final_source = _final_product_source_for_selection(settings, selection)
    if final_source is not None:
        source_path = final_source.probability_path if view_mode == "probability" else final_source.deterministic_path
        return _copy_product_artifact(
            settings,
            theme=selection.theme,
            title=final_source.title or source.title,
            forecast_year=final_source.forecast_year,
            view_mode=view_mode,
            source_path=source_path,
            season_profile=selection.season_profile,
            subseason=selection.subseason,
            generation_backend=f"{settings.forecast_products.generation_backend}_final_netcdf",
            source_artifact_type="final_netcdf",
        )

    if _selection_can_derive_from_daily(settings, selection):
        return _generate_daily_derived_product_artifact(settings, selection, source, view_mode)

    source_path = source.probability_path if view_mode == "probability" else source.deterministic_path
    if source_path is None or not source_path.exists():
        raise ForecastProductArtifactsNotAvailableError(f"Forecast source file is missing: {source_path}")

    return _copy_product_artifact(
        settings,
        theme=source.theme,
        title=source.title,
        forecast_year=source.forecast_year,
        view_mode=view_mode,
        source_path=source_path,
        preview_source_path=source.preview_path,
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )


def _copy_product_artifact(
    settings: Settings,
    *,
    theme: str,
    title: str,
    forecast_year: int,
    view_mode: ViewMode,
    source_path: Path,
    preview_source_path: Path | None = None,
    season_profile: str | None = None,
    subseason: str | None = None,
    generation_backend: str | None = None,
    source_artifact_type: str = "final_netcdf",
) -> dict[str, Any]:
    if not source_path.exists():
        raise ForecastProductArtifactsNotAvailableError(f"Forecast source file is missing: {source_path}")
    product_dir = _product_scope_path(
        settings,
        theme,
        view_mode,
        season_profile=season_profile,
        subseason=subseason,
    )
    ensure_directory(product_dir)
    copied_path = product_dir / source_path.name
    temp_copy_path = copied_path.with_suffix(f"{copied_path.suffix}.tmp")
    if temp_copy_path.exists():
        temp_copy_path.unlink()
    shutil.copyfile(source_path, temp_copy_path)
    temp_copy_path.replace(copied_path)

    preview_path = product_dir / "preview.png"
    if preview_source_path is not None and preview_source_path.exists():
        temp_preview_path = preview_path.with_suffix(".png.tmp")
        if temp_preview_path.exists():
            temp_preview_path.unlink()
        shutil.copyfile(preview_source_path, temp_preview_path)
        temp_preview_path.replace(preview_path)
    else:
        if view_mode == "probability":
            preview_bytes = _build_probability_preview_png(copied_path, theme)
        else:
            preview_bytes = _build_deterministic_preview_png(copied_path, theme)
        preview_path.write_bytes(preview_bytes)

    generated_at = datetime.now(UTC)
    suffix = ""
    if season_profile is not None:
        suffix = f"{suffix}_{season_profile}"
    if subseason is not None:
        suffix = f"{suffix}_{subseason.lower()}"
    product_id = f"{theme}_{view_mode}{suffix}_{forecast_year}_{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "product_id": product_id,
        "theme": theme,
        "view_mode": view_mode,
        "season_profile": season_profile,
        "subseason": subseason,
        "title": title,
        "forecast_year": forecast_year,
        "generated_at": generated_at.isoformat(),
        "source_label": settings.forecast_products.source_label,
        "source_run_id": product_id,
        "generation_backend": generation_backend or settings.forecast_products.generation_backend,
        "source_artifact_type": source_artifact_type,
        "source_path": str(source_path),
        "refresh_interval_seconds": settings.forecast_products.refresh_interval_seconds,
        "freshness_threshold_hours": settings.forecast_products.freshness_threshold_hours,
        "data_path": str(copied_path),
        "preview_path": str(preview_path),
    }
    manifest_path = product_dir / "active.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _generate_daily_derived_product_artifact(
    settings: Settings,
    selection: ForecastProductSelection,
    source: ForecastProductSourceConfig,
    view_mode: ViewMode,
) -> dict[str, Any]:
    forecast_year = _daily_derivation_forecast_year(settings, selection, source.forecast_year)
    derived = _derive_daily_forecast_grid(settings, selection, forecast_year)
    product_dir = _product_scope_path(
        settings,
        selection.theme,
        view_mode,
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    ensure_directory(product_dir)

    mode_token = "Prob" if view_mode == "probability" else "Det"
    selector_token = selection.season_profile or (selection.subseason.lower() if selection.subseason else "global")
    data_path = product_dir / f"Forecast_{mode_token}_{selection.theme}_{selector_token}_{derived.forecast_year}.nc"
    if view_mode == "probability":
        _write_probability_product_netcdf(
            data_path,
            probabilities=derived.probabilities,
            latitudes=derived.latitudes,
            longitudes=derived.longitudes,
            valid_time=derived.valid_time,
        )
        preview_bytes = _build_probability_preview_png(data_path, selection.theme)
    else:
        _write_deterministic_product_netcdf(
            data_path,
            values=derived.deterministic_values,
            latitudes=derived.latitudes,
            longitudes=derived.longitudes,
            valid_time=derived.valid_time,
        )
        preview_bytes = _build_deterministic_preview_png(data_path, selection.theme)

    preview_path = product_dir / "preview.png"
    preview_path.write_bytes(preview_bytes)

    generated_at = datetime.now(UTC)
    suffix = ""
    if selection.season_profile is not None:
        suffix = f"{suffix}_{selection.season_profile}"
    if selection.subseason is not None:
        suffix = f"{suffix}_{selection.subseason.lower()}"
    product_id = f"{selection.theme}_{view_mode}{suffix}_{derived.forecast_year}_{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "product_id": product_id,
        "theme": selection.theme,
        "view_mode": view_mode,
        "season_profile": selection.season_profile,
        "subseason": selection.subseason,
        "title": source.title,
        "forecast_year": derived.forecast_year,
        "generated_at": generated_at.isoformat(),
        "source_label": settings.forecast_products.source_label,
        "source_run_id": product_id,
        "generation_backend": f"{settings.forecast_products.generation_backend}_daily_wass2s",
        "source_artifact_type": "daily_wass2s_derived",
        "refresh_interval_seconds": settings.forecast_products.refresh_interval_seconds,
        "freshness_threshold_hours": settings.forecast_products.freshness_threshold_hours,
        "data_path": str(data_path),
        "preview_path": str(preview_path),
    }
    manifest_path = product_dir / "active.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_probability_product_netcdf(
    data_path: Path,
    *,
    probabilities: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    valid_time: datetime,
) -> None:
    dataset = xr.Dataset(
        {
            "forecast_probability": (
                ("probability", "T", "Y", "X"),
                probabilities[:, None, :, :].astype(float),
            )
        },
        coords={
            "probability": list(PROBABILITY_CODES),
            "T": [np.datetime64(valid_time.replace(tzinfo=None))],
            "Y": latitudes.astype(float),
            "X": longitudes.astype(float),
        },
    )
    temp_path = data_path.with_suffix(f"{data_path.suffix}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    dataset.to_netcdf(temp_path)
    dataset.close()
    temp_path.replace(data_path)


def _write_deterministic_product_netcdf(
    data_path: Path,
    *,
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    valid_time: datetime,
) -> None:
    dataset = xr.Dataset(
        {
            "forecast_deterministic": (
                ("T", "Y", "X"),
                values[None, :, :].astype(float),
            )
        },
        coords={
            "T": [np.datetime64(valid_time.replace(tzinfo=None))],
            "Y": latitudes.astype(float),
            "X": longitudes.astype(float),
        },
    )
    temp_path = data_path.with_suffix(f"{data_path.suffix}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    dataset.to_netcdf(temp_path)
    dataset.close()
    temp_path.replace(data_path)


def _final_product_source_for_selection(
    settings: Settings,
    selection: ForecastProductSelection,
) -> ForecastProductPairSourceConfig | None:
    if _selection_requires_profile_derived_onset(selection):
        return None
    configured = _configured_final_product_source_for_selection(settings, selection)
    if configured is not None and _final_product_pair_has_files(configured):
        return configured
    discovered = _discovered_final_product_source_for_selection(settings, selection)
    if discovered is not None and _final_product_pair_has_files(discovered):
        return discovered
    return None


def _configured_final_product_source_for_selection(
    settings: Settings,
    selection: ForecastProductSelection,
) -> ForecastProductPairSourceConfig | None:
    theme_sources = _case_insensitive_lookup(settings.forecast_products.final_product_sources, selection.theme)
    if theme_sources is not None:
        matched = _source_for_selector(theme_sources, selection)
        if matched is not None:
            return matched

    if selection.theme == "rainfall_amount" and selection.subseason is not None:
        return _source_for_selector(settings.forecast_products.rainfall_total_sources, selection)
    return None


def _final_product_source_is_configured_for_selection(settings: Settings, selection: ForecastProductSelection) -> bool:
    return _configured_final_product_source_for_selection(settings, selection) is not None


def _source_for_selector(
    sources: dict[str, ForecastProductPairSourceConfig],
    selection: ForecastProductSelection,
) -> ForecastProductPairSourceConfig | None:
    for candidate in _final_product_selector_candidates(selection):
        source = _case_insensitive_lookup(sources, candidate)
        if source is not None:
            return source
    return None


def _final_product_selector_candidates(selection: ForecastProductSelection) -> tuple[str, ...]:
    if selection.subseason is not None:
        return (selection.subseason, selection.subseason.lower())
    if selection.season_profile is not None:
        return (
            selection.season_profile,
            selection.season_profile.upper(),
            DEFAULT_FINAL_PRODUCT_SELECTOR,
            "default",
            "global",
            "all",
        )
    return (DEFAULT_FINAL_PRODUCT_SELECTOR, "default", "global", "all")


def _case_insensitive_lookup(mapping: dict[str, Any], key: str) -> Any | None:
    normalized_key = str(key).strip().lower()
    for candidate, value in mapping.items():
        if str(candidate).strip().lower() == normalized_key:
            return value
    return None


def _final_product_pair_has_files(source: ForecastProductPairSourceConfig) -> bool:
    return source.deterministic_path.exists() and source.probability_path.exists()


def _final_product_source_has_file(source: ForecastProductPairSourceConfig, view_mode: ViewMode) -> bool:
    source_path = source.probability_path if view_mode == "probability" else source.deterministic_path
    return source_path.exists()


def _discovered_final_product_source_for_selection(
    settings: Settings,
    selection: ForecastProductSelection,
) -> ForecastProductPairSourceConfig | None:
    discovered = _discover_final_product_sources(tuple(str(path) for path in settings.forecast_products.final_product_dirs))
    selectors = _final_product_selector_candidates(selection)
    for selector in selectors:
        source = discovered.get((selection.theme, selector.lower()))
        if source is not None:
            return source
    return None


def _discover_final_product_sources(source_dirs: tuple[str, ...]) -> dict[tuple[str, str], ForecastProductPairSourceConfig]:
    pairs: dict[tuple[str, str, int], dict[str, Path]] = {}
    for source_dir in source_dirs:
        directory = Path(source_dir)
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.nc")):
            parsed = _parse_final_product_filename(path)
            if parsed is None:
                continue
            view_mode, theme, selector, forecast_year = parsed
            pairs.setdefault((theme, selector.lower(), forecast_year), {})[view_mode] = path

    discovered: dict[tuple[str, str], ForecastProductPairSourceConfig] = {}
    for theme, selector, forecast_year in sorted(pairs, key=lambda item: (item[0], item[1], item[2]), reverse=True):
        paths = pairs[(theme, selector, forecast_year)]
        deterministic_path = paths.get("deterministic")
        probability_path = paths.get("probability")
        if deterministic_path is None or probability_path is None:
            continue
        discovered.setdefault(
            (theme, selector),
            ForecastProductPairSourceConfig(
                deterministic_path=deterministic_path,
                probability_path=probability_path,
                forecast_year=forecast_year,
                title=_theme_spec(theme).theme_label,
            ),
        )
    return discovered


def _parse_final_product_filename(path: Path) -> tuple[ViewMode, str, str, int] | None:
    match = re.match(r"^Forecast_(Det|Prob)_(?P<token>.+?)_(?P<year>\d{4})\.nc$", path.name, flags=re.IGNORECASE)
    if match is None:
        return None
    view_mode: ViewMode = "deterministic" if match.group(1).lower() == "det" else "probability"
    metadata = _final_product_token_metadata(match.group("token"))
    if metadata is None:
        return None
    theme, selector = metadata
    return view_mode, theme, selector, int(match.group("year"))


def _final_product_token_metadata(token: str) -> tuple[str, str] | None:
    normalized = re.sub(r"[^a-z0-9]+", "", token.strip().lower())
    subseason = _subseason_from_final_token(normalized)
    if subseason is not None:
        if any(marker in normalized for marker in ("rainyday", "rainydays", "rainday", "raindays", "wetday", "wetdays")):
            return "rainy_days", subseason
        return "rainfall_amount", subseason

    theme = FINAL_THEME_TOKEN_MAP.get(normalized)
    if theme is None and normalized.startswith("prcp"):
        theme = FINAL_THEME_TOKEN_MAP.get(normalized[4:])
    if theme is None:
        return None
    return theme, DEFAULT_FINAL_PRODUCT_SELECTOR


def _subseason_from_final_token(normalized_token: str) -> str | None:
    for token, subseason in FINAL_SUBSEASON_TOKEN_MAP.items():
        if token in normalized_token:
            return subseason
    for subseason in SUBSEASON_DISPLAY_ORDER:
        token = subseason.lower()
        if normalized_token == token or normalized_token.endswith(token):
            return subseason
    return None


def _selection_can_derive_from_daily(settings: Settings, selection: ForecastProductSelection) -> bool:
    if selection.theme not in DAILY_DERIVED_THEMES:
        return False
    if selection.theme in FINAL_PRODUCT_ONLY_THEMES:
        return False
    if not _selection_requires_profile_derived_onset(selection):
        if _final_product_source_is_configured_for_selection(settings, selection):
            return False
        if _discovered_final_product_source_for_selection(settings, selection) is not None:
            return False
    if selection.theme in SEASON_BASED_THEMES and selection.season_profile is None:
        return False
    if selection.theme in SUBSEASON_BASED_THEMES and selection.subseason is None:
        return False
    if selection.theme in SUBSEASON_BASED_THEMES and not _subseason_has_profile_normals(settings, selection):
        return False
    paths = _daily_forecast_paths(settings)
    if len(paths) < max(int(settings.forecast_products.derived_min_member_count), 1):
        return False
    return _daily_source_has_required_dates(settings, selection, paths)


def _daily_forecast_paths(settings: Settings) -> list[Path]:
    source_dir = settings.forecast_products.daily_corrected_dir
    if not source_dir.exists() or not source_dir.is_dir():
        return []
    return sorted(path for path in source_dir.glob(settings.forecast_products.daily_forecast_glob) if path.is_file())


def _daily_derivation_forecast_year(
    settings: Settings,
    selection: ForecastProductSelection,
    fallback_year: int,
) -> int:
    if not _selection_requires_profile_derived_onset(selection):
        return int(fallback_year)
    inferred_year = _infer_daily_forecast_year(settings)
    return inferred_year if inferred_year is not None else int(fallback_year)


def _infer_daily_forecast_year(settings: Settings) -> int | None:
    for path in _daily_forecast_paths(settings):
        try:
            with xr.open_dataset(path) as dataset:
                field = _normalize_daily_rainfall_field(_resolve_daily_rainfall_data_array(dataset))
                times = pd.DatetimeIndex(field["T"].values)
        except Exception:
            continue
        if len(times):
            return int(times[0].year)
    return None


def _daily_source_has_required_dates(
    settings: Settings,
    selection: ForecastProductSelection,
    paths: list[Path],
) -> bool:
    source = _source_for_theme(settings, selection.theme)
    forecast_year = _daily_derivation_forecast_year(settings, selection, source.forecast_year)
    expected_dates = _required_daily_dates(selection, forecast_year)
    if expected_dates.empty:
        return False
    required_fraction = _daily_min_coverage_fraction(settings)
    usable_members = 0
    for path in paths:
        try:
            with xr.open_dataset(path) as dataset:
                field = _normalize_daily_rainfall_field(_resolve_daily_rainfall_data_array(dataset))
                available = pd.DatetimeIndex(field["T"].values).normalize()
        except Exception:
            continue
        expected = expected_dates.normalize()
        coverage = float(np.isin(expected.values, available.values).sum()) / float(len(expected))
        if coverage >= required_fraction:
            usable_members += 1
    return usable_members >= max(int(settings.forecast_products.derived_min_member_count), 1)


def _derive_daily_forecast_grid(
    settings: Settings,
    selection: ForecastProductSelection,
    forecast_year: int,
) -> DerivedDailyForecastGrid:
    member_values, times, latitudes, longitudes = _load_daily_member_stack(settings, selection, forecast_year)
    if selection.theme in SEASON_BASED_THEMES:
        profile = settings.seasonal_map.profiles[str(selection.season_profile)]
        metrics = _derive_season_member_metrics(settings, selection.theme, member_values, times, profile)
        deterministic, probabilities = _build_season_member_outputs(selection.theme, metrics, profile, forecast_year, settings)
    else:
        metrics = _derive_subseason_member_metrics(settings, selection.theme, member_values)
        deterministic, probabilities = _build_subseason_member_outputs(settings, selection, metrics, latitudes, longitudes)

    if not np.isfinite(deterministic).any():
        raise ForecastProductArtifactsNotAvailableError(
            _artifacts_not_available_message(
                selection.theme,
                "deterministic",
                season_profile=selection.season_profile,
                subseason=selection.subseason,
            )
        )
    if not np.isfinite(probabilities).any():
        raise ForecastProductArtifactsNotAvailableError(
            _artifacts_not_available_message(
                selection.theme,
                "probability",
                season_profile=selection.season_profile,
                subseason=selection.subseason,
            )
        )

    return DerivedDailyForecastGrid(
        forecast_year=forecast_year,
        valid_time=_derived_valid_time(settings, selection, forecast_year),
        latitudes=latitudes,
        longitudes=longitudes,
        deterministic_values=deterministic,
        probabilities=probabilities,
    )


def _load_daily_member_stack(
    settings: Settings,
    selection: ForecastProductSelection,
    forecast_year: int,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    paths = _daily_forecast_paths(settings)
    if len(paths) < max(int(settings.forecast_products.derived_min_member_count), 1):
        raise ForecastProductArtifactsNotAvailableError(
            f"Daily WASS2S source requires at least {settings.forecast_products.derived_min_member_count} forecast member file(s)."
        )
    expected_dates = _required_daily_dates(selection, forecast_year)
    required_fraction = _daily_min_coverage_fraction(settings)
    arrays: list[np.ndarray] = []
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    for path in paths:
        with xr.open_dataset(path) as dataset:
            field = _normalize_daily_rainfall_field(_resolve_daily_rainfall_data_array(dataset))
            available = pd.DatetimeIndex(field["T"].values).normalize()
            coverage = float(np.isin(expected_dates.normalize().values, available.values).sum()) / float(len(expected_dates))
            if coverage < required_fraction:
                continue
            field = field.sel(T=slice(expected_dates[0], expected_dates[-1])).reindex(T=expected_dates)
            field = field.transpose("T", "Y", "X")
            current_latitudes = np.asarray(field["Y"].values, dtype=float)
            current_longitudes = np.asarray(field["X"].values, dtype=float)
            if latitudes is None:
                latitudes = current_latitudes
                longitudes = current_longitudes
            elif not (np.allclose(latitudes, current_latitudes) and np.allclose(longitudes, current_longitudes)):
                raise ForecastProductIncompleteError(f"Daily WASS2S grid mismatch in {path}.")
            arrays.append(np.asarray(field.values, dtype=float))

    min_members = max(int(settings.forecast_products.derived_min_member_count), 1)
    if len(arrays) < min_members:
        raise ForecastProductArtifactsNotAvailableError(
            f"Daily WASS2S source has {len(arrays)} usable member file(s); {min_members} required."
        )
    assert latitudes is not None
    assert longitudes is not None
    times = pd.DatetimeIndex(expected_dates)
    return np.stack(arrays, axis=0), times, latitudes, longitudes


def _resolve_daily_rainfall_data_array(dataset: xr.Dataset) -> xr.DataArray:
    for candidate in ("corrected", "precip_mm", "rainfall", "rainfall_mm", "prcp", "tp"):
        if candidate in dataset.data_vars:
            field = dataset[candidate]
            break
    else:
        field = dataset[list(dataset.data_vars)[0]]
    units = str(field.attrs.get("units", "")).lower()
    if units in {"m", "meter", "meters"}:
        field = field * 1000.0
        field.attrs["units"] = "mm"
    return field


def _normalize_daily_rainfall_field(field: xr.DataArray) -> xr.DataArray:
    rename_map: dict[str, str] = {}
    for name in field.dims:
        normalized = str(name).strip().lower()
        if normalized in {"time", "date"}:
            rename_map[name] = "T"
        elif normalized in {"latitude", "lat"}:
            rename_map[name] = "Y"
        elif normalized in {"longitude", "lon"}:
            rename_map[name] = "X"
    if rename_map:
        field = field.rename(rename_map)
    missing = {"T", "Y", "X"} - set(field.dims)
    if missing:
        raise ForecastProductIncompleteError(f"Daily WASS2S rainfall field is missing dimensions: {', '.join(sorted(missing))}.")
    return field.transpose("T", "Y", "X")


def _required_daily_dates(selection: ForecastProductSelection, forecast_year: int) -> pd.DatetimeIndex:
    if selection.subseason is not None:
        months = CALENDAR_SUBSEASON_MONTHS[selection.subseason]
        start = pd.Timestamp(year=forecast_year, month=months[0], day=1)
        end = pd.Timestamp(year=forecast_year, month=months[-1], day=1) + pd.offsets.MonthEnd(0)
        return pd.date_range(start, end, freq="D")
    return pd.date_range(
        pd.Timestamp(year=forecast_year, month=1, day=1),
        pd.Timestamp(year=forecast_year, month=12, day=31),
        freq="D",
    )


def _daily_min_coverage_fraction(settings: Settings) -> float:
    return min(max(float(settings.forecast_products.derived_min_coverage_fraction), 0.0), 1.0)


def _derive_subseason_member_metrics(
    settings: Settings,
    theme: str,
    member_values: np.ndarray,
) -> np.ndarray:
    finite_days = np.isfinite(member_values).sum(axis=1)
    min_days = max(1, int(math.ceil(member_values.shape[1] * _daily_min_coverage_fraction(settings))))
    if theme == "rainfall_amount":
        metrics = np.nansum(np.where(np.isfinite(member_values), member_values, 0.0), axis=1)
    else:
        metrics = np.sum(member_values >= float(settings.seasonal_map.rainy_day_threshold_mm), axis=1).astype(float)
    return np.where(finite_days >= min_days, metrics, np.nan)


def _derive_season_member_metrics(
    settings: Settings,
    theme: str,
    member_values: np.ndarray,
    times: pd.DatetimeIndex,
    profile: SeasonalProfileConfig,
) -> np.ndarray:
    members, days, rows, cols = member_values.shape
    min_days = max(1, int(math.ceil(days * _daily_min_coverage_fraction(settings))))
    metrics = np.full((members, rows, cols), np.nan, dtype=float)
    for member_index in range(members):
        for y_index in range(rows):
            for x_index in range(cols):
                series = member_values[member_index, :, y_index, x_index]
                if int(np.isfinite(series).sum()) < min_days:
                    continue
                filled = np.where(np.isfinite(series), np.maximum(series, 0.0), 0.0).astype(float)
                onset_index = _detect_onset_index(filled, times, profile, settings)
                if theme == "onset":
                    metrics[member_index, y_index, x_index] = float(pd.Timestamp(times[onset_index]).dayofyear)
                elif theme == "cessation":
                    cessation_index = _detect_cessation_index(filled, times, onset_index, profile)
                    metrics[member_index, y_index, x_index] = float(pd.Timestamp(times[cessation_index]).dayofyear)
                elif theme == "late_dry_spell":
                    cessation_index = _detect_cessation_index(filled, times, onset_index, profile)
                    start_index = min(days, onset_index + 50)
                    stop_index = min(days, cessation_index + 1)
                    if start_index >= stop_index:
                        metrics[member_index, y_index, x_index] = 0.0
                    else:
                        metrics[member_index, y_index, x_index] = float(
                            _longest_dry_spell_from_values(
                                filled[start_index:stop_index],
                                float(settings.seasonal_map.dry_day_threshold_mm),
                            )
                        )
                else:
                    raise InvalidForecastProductThemeError(f"Daily derivation is not supported for theme '{theme}'.")
    return metrics


def _detect_onset_index(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    profile: SeasonalProfileConfig,
    settings: Settings,
) -> int:
    start_idx = _nearest_date_index(times, profile.onset_search_start_month, profile.onset_search_start_day)
    window_days = max(1, int(profile.onset_window_days))
    for index in range(start_idx, max(start_idx, len(values) - window_days + 1)):
        window = values[index : index + window_days]
        if float(np.sum(window)) < float(profile.onset_threshold_mm):
            continue
        guard = values[index : min(len(values), index + int(profile.onset_guard_window_days))]
        guard_spell = _longest_dry_spell_from_values(guard, float(settings.seasonal_map.dry_day_threshold_mm))
        if guard_spell <= int(profile.onset_guard_max_dry_spell_days):
            return index

    best_index = start_idx
    best_total = float("-inf")
    for index in range(start_idx, max(start_idx + 1, len(values) - window_days + 1)):
        total = float(np.sum(values[index : index + window_days]))
        if total > best_total:
            best_total = total
            best_index = index
    return int(best_index)


def _detect_cessation_index(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    onset_index: int,
    profile: SeasonalProfileConfig,
) -> int:
    start_idx = max(_nearest_date_index(times, profile.cessation_search_start_month, profile.cessation_search_start_day), onset_index)
    balance = float(profile.cessation_soil_water_mm)
    for index in range(start_idx, len(values)):
        balance = min(
            float(profile.cessation_soil_water_mm),
            balance + float(values[index]) - float(profile.cessation_et_mm_per_day),
        )
        if balance <= 0.0:
            return int(index)
    return len(values) - 1


def _nearest_date_index(times: pd.DatetimeIndex, month: int, day: int) -> int:
    target = pd.Timestamp(year=int(times[0].year), month=int(month), day=int(day))
    return int(times.get_indexer([target], method="nearest")[0])


def _longest_dry_spell_from_values(values: np.ndarray, dry_day_threshold_mm: float) -> int:
    max_run = 0
    current_run = 0
    for value in values.tolist():
        if float(value) < dry_day_threshold_mm:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return int(max_run)


def _build_season_member_outputs(
    theme: str,
    metrics: np.ndarray,
    profile: SeasonalProfileConfig,
    forecast_year: int,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    deterministic = _member_mean_grid(settings, metrics)
    if theme == "onset":
        reference = float(_day_of_year(forecast_year, profile.onset_reference_month, profile.onset_reference_day))
        band = max(float(profile.onset_normal_band_days), 1.0)
        probabilities = _probabilities_from_member_thresholds(metrics, reference - band, reference + band, settings)
    elif theme == "cessation":
        reference = float(_day_of_year(forecast_year, profile.cessation_reference_month, profile.cessation_reference_day))
        band = max(float(profile.cessation_normal_band_days), 1.0)
        probabilities = _probabilities_from_member_thresholds(metrics, reference - band, reference + band, settings)
    elif theme == "late_dry_spell":
        lower = float(profile.late_dry_spell_moderate_days)
        upper = float(profile.late_dry_spell_high_days)
        probabilities = _probabilities_from_member_thresholds(metrics, lower, upper, settings)
    else:
        raise InvalidForecastProductThemeError(f"Daily derivation is not supported for theme '{theme}'.")
    return deterministic, probabilities


def _build_subseason_member_outputs(
    settings: Settings,
    selection: ForecastProductSelection,
    metrics: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    deterministic = _member_mean_grid(settings, metrics)
    normal_grid = _subseason_normal_grid(settings, selection, latitudes, longitudes)
    if selection.theme == "rainfall_amount":
        band_pct = _subseason_band_grid(settings, selection, latitudes, longitudes, "rainfall")
        lower = normal_grid * (1.0 - band_pct / 100.0)
        upper = normal_grid * (1.0 + band_pct / 100.0)
    elif selection.theme == "rainy_days":
        band = _subseason_band_grid(settings, selection, latitudes, longitudes, "rainy_days")
        lower = normal_grid - band
        upper = normal_grid + band
    else:
        raise InvalidForecastProductThemeError(f"Daily derivation is not supported for theme '{selection.theme}'.")
    probabilities = _probabilities_from_member_thresholds(metrics, lower, upper, settings)
    return deterministic, probabilities


def _member_mean_grid(settings: Settings, member_values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(member_values)
    counts = valid.sum(axis=0)
    totals = np.sum(np.where(valid, member_values, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = totals / counts
    return np.where(counts >= max(int(settings.forecast_products.derived_min_member_count), 1), mean, np.nan)


def _probabilities_from_member_thresholds(
    member_values: np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    settings: Settings,
) -> np.ndarray:
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    valid = np.isfinite(member_values) & np.isfinite(lower_array) & np.isfinite(upper_array)
    counts = valid.sum(axis=0).astype(float)
    below = np.sum(valid & (member_values < lower_array), axis=0)
    near = np.sum(valid & (member_values >= lower_array) & (member_values <= upper_array), axis=0)
    above = np.sum(valid & (member_values > upper_array), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.stack((below / counts, near / counts, above / counts), axis=0).astype(float)
    probabilities[:, counts < max(int(settings.forecast_products.derived_min_member_count), 1)] = np.nan
    return probabilities


def _subseason_has_profile_normals(settings: Settings, selection: ForecastProductSelection) -> bool:
    if selection.subseason is None:
        return False
    return any(_profile_subseason_normal(profile, selection.theme, selection.subseason) is not None for profile in settings.seasonal_map.profiles.values())


def _subseason_normal_grid(
    settings: Settings,
    selection: ForecastProductSelection,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    if selection.subseason is None:
        return grid
    for profile in settings.seasonal_map.profiles.values():
        normal = _profile_subseason_normal(profile, selection.theme, selection.subseason)
        if normal is None:
            continue
        mask = _district_zone_cell_mask(settings, profile.native_zone, latitudes, longitudes)
        grid = np.where(mask & ~np.isfinite(grid), float(normal), grid)
    return grid


def _subseason_band_grid(
    settings: Settings,
    selection: ForecastProductSelection,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    band_kind: str,
) -> np.ndarray:
    grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    if selection.subseason is None:
        return grid
    for profile in settings.seasonal_map.profiles.values():
        if _profile_subseason_normal(profile, selection.theme, selection.subseason) is None:
            continue
        band = float(profile.rainfall_band_pct if band_kind == "rainfall" else profile.rainy_days_band)
        mask = _district_zone_cell_mask(settings, profile.native_zone, latitudes, longitudes)
        grid = np.where(mask & ~np.isfinite(grid), band, grid)
    return grid


def _profile_subseason_normal(profile: SeasonalProfileConfig, theme: str, subseason: str) -> float | None:
    if subseason not in profile.calendar_subseasons:
        return None
    if theme == "rainfall_amount":
        value = profile.calendar_rainfall_normals_mm.get(subseason)
    elif theme == "rainy_days":
        value = profile.calendar_rainy_days_normals.get(subseason)
    else:
        value = None
    return float(value) if value is not None else None


def _day_of_year(forecast_year: int, month: int, day: int) -> int:
    return int(pd.Timestamp(year=forecast_year, month=month, day=day).dayofyear)


def _derived_valid_time(settings: Settings, selection: ForecastProductSelection, forecast_year: int) -> datetime:
    if selection.subseason is not None:
        middle_month = CALENDAR_SUBSEASON_MONTHS[selection.subseason][1]
        return datetime(forecast_year, middle_month, 1, tzinfo=UTC)
    if selection.season_profile is not None:
        profile = settings.seasonal_map.profiles[selection.season_profile]
        return datetime(forecast_year, profile.cessation_reference_month, profile.cessation_reference_day, tzinfo=UTC)
    return datetime(forecast_year, 1, 1, tzinfo=UTC)


def _ensure_active_manifest(
    settings: Settings,
    theme: str,
    view_mode: ViewMode,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
    materialize_missing: bool = False,
) -> dict[str, Any]:
    selection = _resolve_selection(settings, theme, season_profile=season_profile, subseason=subseason)
    normalized_theme = selection.theme
    manifest_path = _canonical_manifest_path(
        settings,
        normalized_theme,
        view_mode,
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    source = _source_for_theme(settings, normalized_theme)
    final_source = _final_product_source_for_selection(settings, selection)
    if materialize_missing and final_source is not None:
        return _generate_product_artifact(
            settings,
            source,
            view_mode,
            season_profile=selection.season_profile,
            subseason=selection.subseason,
        )
    if not manifest_path.exists():
        if final_source is not None:
            return _generate_product_artifact(
                settings,
                source,
                view_mode,
                season_profile=selection.season_profile,
                subseason=selection.subseason,
            )
        if not materialize_missing:
            raise ForecastProductArtifactsNotAvailableError(
                _artifacts_not_available_message(
                    normalized_theme,
                    view_mode,
                    season_profile=selection.season_profile,
                    subseason=selection.subseason,
                )
            )
        legacy_path = _legacy_manifest_path(settings, normalized_theme, view_mode)
        if legacy_path.exists():
            legacy_manifest = json.loads(legacy_path.read_text(encoding="utf-8"))
            manifest = _promote_manifest_to_selection(
                settings,
                legacy_manifest,
                normalized_theme,
                view_mode,
                season_profile=selection.season_profile,
                subseason=selection.subseason,
            )
        elif _selection_view_is_materializable(settings, selection, view_mode):
            manifest = _generate_product_artifact(
                settings,
                source,
                view_mode,
                season_profile=selection.season_profile,
                subseason=selection.subseason,
            )
        else:
            raise ForecastProductArtifactsNotAvailableError(
                _artifacts_not_available_message(
                    normalized_theme,
                    view_mode,
                    season_profile=selection.season_profile,
                    subseason=selection.subseason,
                )
            )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _normalize_manifest_selection(manifest, selection, manifest_path)
    if final_source is not None and not _manifest_payload_is_available_for_selection(settings, selection, manifest):
        return _generate_product_artifact(
            settings,
            source,
            view_mode,
            season_profile=selection.season_profile,
            subseason=selection.subseason,
        )
    if not _manifest_payload_is_available_for_selection(settings, selection, manifest):
        if not materialize_missing:
            raise ForecastProductArtifactsNotAvailableError(
                _artifacts_not_available_message(
                    normalized_theme,
                    view_mode,
                    season_profile=selection.season_profile,
                    subseason=selection.subseason,
                )
            )
        if _selection_view_is_materializable(settings, selection, view_mode):
            manifest = _generate_product_artifact(
                settings,
                source,
                view_mode,
                season_profile=selection.season_profile,
                subseason=selection.subseason,
            )
        else:
            raise ForecastProductArtifactsNotAvailableError(
                _artifacts_not_available_message(
                    normalized_theme,
                    view_mode,
                    season_profile=selection.season_profile,
                    subseason=selection.subseason,
                )
            )
    return manifest


def _selection_metadata(settings: Settings, theme: str) -> dict[str, Any]:
    normalized_theme = _normalize_theme(theme)
    _theme_spec(normalized_theme)
    requires_season = normalized_theme in SEASON_BASED_THEMES
    requires_subseason = normalized_theme in SUBSEASON_BASED_THEMES
    ready_selections = [
        selection
        for selection in _all_selections_for_theme(settings, normalized_theme)
        if _selection_has_complete_ready_product(settings, selection)
    ]
    enabled = bool(ready_selections)
    seasons = [str(selection.season_profile) for selection in ready_selections if selection.season_profile is not None]
    subseasons = [str(selection.subseason) for selection in ready_selections if selection.subseason is not None]
    return {
        "requires_season": requires_season,
        "requires_subseason": requires_subseason,
        "enabled": enabled,
        "reason": None if enabled else "artifacts_not_generated",
        "seasons": seasons,
        "subseasons": subseasons,
    }


def _theme_is_enabled(settings: Settings, theme: str) -> bool:
    for selection in _all_selections_for_theme(settings, theme):
        if _selection_has_complete_artifacts(settings, selection):
            return True
    return False


def _selection_has_complete_artifacts(settings: Settings, selection: ForecastProductSelection) -> bool:
    has_probability = _view_mode_is_available(
        settings,
        selection.theme,
        "probability",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    has_deterministic = _view_mode_is_available(
        settings,
        selection.theme,
        "deterministic",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    return has_probability and has_deterministic and _selection_product_pair_is_compatible(settings, selection)


def _selection_has_complete_published_manifests(settings: Settings, selection: ForecastProductSelection) -> bool:
    return (
        _available_manifest_path(settings, selection, "probability") is not None
        and _available_manifest_path(settings, selection, "deterministic") is not None
    )


def _selection_has_complete_ready_product(settings: Settings, selection: ForecastProductSelection) -> bool:
    if _selection_has_complete_published_manifests(settings, selection):
        return True
    final_source = _final_product_source_for_selection(settings, selection)
    return final_source is not None and _final_product_pair_has_files(final_source)


def _available_manifest_path(
    settings: Settings,
    selection: ForecastProductSelection,
    view_mode: ViewMode,
) -> Path | None:
    canonical = _canonical_manifest_path(
        settings,
        selection.theme,
        view_mode,
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    if _manifest_file_is_available(settings, selection, canonical):
        return canonical
    legacy = _legacy_manifest_path(settings, selection.theme, view_mode)
    if _manifest_file_is_available(settings, selection, legacy):
        return legacy
    return None


def _manifest_file_is_available(
    settings: Settings,
    selection: ForecastProductSelection,
    manifest_path: Path,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _manifest_payload_is_available_for_selection(settings, selection, manifest)


def _apply_selection_spatial_mask(
    settings: Settings,
    selection: ForecastProductSelection,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    mask_zone = _selection_mask_zone(settings, selection)
    cell_mask = _district_zone_cell_mask(settings, mask_zone, latitudes, longitudes)
    if not np.any(cell_mask):
        raise ForecastProductArtifactsNotAvailableError(
            f"No forecast cells overlap the product footprint for mask_zone='{mask_zone}'."
        )

    masked = np.array(values, copy=True)
    if masked.ndim == 2:
        masked = np.where(cell_mask, masked, np.nan)
    elif masked.ndim == 3:
        masked = np.where(cell_mask[None, :, :], masked, np.nan)
    else:
        raise ForecastProductIncompleteError("Forecast product grid has an unsupported dimensionality.")

    if not np.isfinite(masked).any():
        raise ForecastProductArtifactsNotAvailableError(
            f"No valid forecast cells remain after applying the product footprint for mask_zone='{mask_zone}'."
    )
    return masked


def _selection_mask_zone(settings: Settings, selection: ForecastProductSelection) -> str:
    if selection.season_profile is None:
        return GHANA_PRODUCT_MASK_ZONE
    profile = settings.seasonal_map.profiles.get(selection.season_profile)
    if profile is None:
        return GHANA_PRODUCT_MASK_ZONE
    return str(profile.native_zone).strip().lower() or GHANA_PRODUCT_MASK_ZONE


def _validate_product_dataset_for_selection(
    settings: Settings,
    selection: ForecastProductSelection,
    view_mode: ViewMode,
    data_path: Path,
) -> None:
    with xr.open_dataset(data_path) as dataset:
        data_var = dataset[list(dataset.data_vars)[0]]
        _validate_product_dimensions(view_mode, data_var)
        if "Y" not in dataset.coords or "X" not in dataset.coords:
            raise ForecastProductIncompleteError("Forecast product is missing Y/X coordinates.")
        latitudes = np.asarray(dataset["Y"].values, dtype=float)
        longitudes = np.asarray(dataset["X"].values, dtype=float)
        if latitudes.ndim != 1 or longitudes.ndim != 1 or latitudes.size == 0 or longitudes.size == 0:
            raise ForecastProductIncompleteError("Forecast product grid coordinates must be non-empty one-dimensional arrays.")
        if view_mode == "probability":
            category_codes = tuple(str(item) for item in data_var.coords["probability"].values.tolist())
            if category_codes != PROBABILITY_CODES:
                raise ForecastProductIncompleteError("Probability product must expose PB, PN, and PA categories.")
            values = _clean_probability_grid(np.asarray(data_var.isel(T=0).values, dtype=float))
            sums = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=0)
            valid = np.isfinite(values).any(axis=0) & (sums > 0.0)
            if not np.any(valid):
                raise ForecastProductIncompleteError("Probability product does not contain any valid probability cells.")
            if not np.allclose(sums[valid], 1.0, atol=0.05):
                raise ForecastProductIncompleteError("Probability product cell probabilities must sum close to 1.0.")
            masked = _apply_selection_spatial_mask(settings, selection, latitudes, longitudes, values)
            if not np.isfinite(masked).any():
                raise ForecastProductIncompleteError("Probability product has no finite values for the requested selection.")
        else:
            values = np.asarray(data_var.isel(T=0).values, dtype=float)
            masked = _apply_selection_spatial_mask(settings, selection, latitudes, longitudes, values)
            if not np.isfinite(masked).any():
                raise ForecastProductIncompleteError("Deterministic product has no finite values for the requested selection.")


def _validate_product_dimensions(view_mode: ViewMode, data_var: xr.DataArray) -> None:
    expected = ("probability", "T", "Y", "X") if view_mode == "probability" else ("T", "Y", "X")
    if tuple(data_var.dims) != expected:
        raise ForecastProductIncompleteError(
            f"{view_mode.title()} product must have dimensions {expected}; received {tuple(data_var.dims)}."
        )


def _district_zone_cell_mask(
    settings: Settings,
    native_zone: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    normalized_zone = str(native_zone).strip().lower()
    if normalized_zone not in {GHANA_PRODUCT_MASK_ZONE, "north", "south"}:
        return np.ones((len(latitudes), len(longitudes)), dtype=bool)

    mask_rows = _district_zone_cell_mask_cached(
        str(settings.seasonal_map.district_geojson_path),
        float(settings.seasonal_map.northern_latitude_threshold),
        normalized_zone,
        tuple(round(float(value), 4) for value in latitudes.tolist()),
        tuple(round(float(value), 4) for value in longitudes.tolist()),
    )
    return np.asarray(mask_rows, dtype=bool)


@lru_cache(maxsize=16)
def _district_zone_cell_mask_cached(
    geojson_path: str,
    threshold: float,
    native_zone: str,
    latitude_key: tuple[float, ...],
    longitude_key: tuple[float, ...],
) -> tuple[tuple[bool, ...], ...]:
    features = _load_district_zone_features(geojson_path, threshold)
    latitude_bounds = _axis_cell_bounds(latitude_key)
    longitude_bounds = _axis_cell_bounds(longitude_key)
    mask: list[tuple[bool, ...]] = []
    for latitude_index, _latitude in enumerate(latitude_key):
        lat_min, lat_max = latitude_bounds[latitude_index]
        row: list[bool] = []
        for longitude_index, _longitude in enumerate(longitude_key):
            lon_min, lon_max = longitude_bounds[longitude_index]
            row.append(_cell_matches_any_zone_feature(lon_min, lat_min, lon_max, lat_max, native_zone, features))
        mask.append(tuple(row))
    return tuple(mask)


@lru_cache(maxsize=4)
def _load_district_zone_features(geojson_path: str, threshold: float) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        region = str(properties.get("region") or "").strip().casefold()
        min_lon, min_lat, max_lon, max_lat = _geometry_bbox(geometry)
        zones: set[str] = set()
        if region in FORECAST_PRODUCT_SOUTHERN_NATIVE_REGIONS:
            zones.add("south")
        elif region in FORECAST_PRODUCT_NORTHERN_NATIVE_REGIONS:
            zones.add("north")
        if not zones:
            if max_lat >= threshold:
                zones.add("north")
            if min_lat <= threshold:
                zones.add("south")
        records.append(
            {
                "geometry": geometry,
                "bbox": (min_lon, min_lat, max_lon, max_lat),
                "zones": frozenset(zones),
            }
        )
    return tuple(records)


def _axis_cell_bounds(axis_key: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    if not axis_key:
        return tuple()
    if len(axis_key) == 1:
        value = float(axis_key[0])
        return ((value, value),)

    bounds: list[tuple[float, float]] = []
    last_index = len(axis_key) - 1
    for index, value in enumerate(axis_key):
        current = float(value)
        if index == 0:
            first_delta = float(axis_key[1]) - current
            edge_a = current - first_delta / 2
        else:
            edge_a = (float(axis_key[index - 1]) + current) / 2
        if index == last_index:
            last_delta = current - float(axis_key[index - 1])
            edge_b = current + last_delta / 2
        else:
            edge_b = (current + float(axis_key[index + 1])) / 2
        bounds.append((min(edge_a, edge_b), max(edge_a, edge_b)))
    return tuple(bounds)


def _cell_matches_any_zone_feature(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    native_zone: str,
    features: tuple[dict[str, Any], ...],
) -> bool:
    for feature in features:
        if native_zone != GHANA_PRODUCT_MASK_ZONE and native_zone not in feature["zones"]:
            continue
        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat = feature["bbox"]
        if lon_max < bbox_min_lon or lon_min > bbox_max_lon or lat_max < bbox_min_lat or lat_min > bbox_max_lat:
            continue
        if _geometry_intersects_cell(lon_min, lat_min, lon_max, lat_max, feature["geometry"]):
            return True
    return False


def _geometry_intersects_cell(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    geometry: dict[str, Any],
) -> bool:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        return _polygon_intersects_cell(lon_min, lat_min, lon_max, lat_max, coordinates)
    if geometry_type == "MultiPolygon":
        return any(
            _polygon_intersects_cell(lon_min, lat_min, lon_max, lat_max, polygon)
            for polygon in coordinates
        )
    return False


def _polygon_intersects_cell(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    polygon: list[list[list[float]]],
) -> bool:
    if not polygon:
        return False

    for longitude, latitude in _cell_sample_points(lon_min, lat_min, lon_max, lat_max):
        if _point_in_polygon(longitude, latitude, polygon):
            return True

    for ring in polygon:
        if _ring_intersects_cell(lon_min, lat_min, lon_max, lat_max, ring):
            return True
    return False


def _cell_sample_points(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> tuple[tuple[float, float], ...]:
    lon_mid = (lon_min + lon_max) / 2
    lat_mid = (lat_min + lat_max) / 2
    return (
        (lon_mid, lat_mid),
        (lon_min, lat_min),
        (lon_mid, lat_min),
        (lon_max, lat_min),
        (lon_min, lat_mid),
        (lon_max, lat_mid),
        (lon_min, lat_max),
        (lon_mid, lat_max),
        (lon_max, lat_max),
    )


def _ring_intersects_cell(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    ring: list[list[float]],
) -> bool:
    point_count = len(ring)
    if point_count < 2:
        return False
    for point in ring:
        longitude, latitude = point
        if lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max:
            return True
    for index in range(point_count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % point_count]
        if _segment_intersects_cell(float(x1), float(y1), float(x2), float(y2), lon_min, lat_min, lon_max, lat_max):
            return True
    return False


def _segment_intersects_cell(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> bool:
    if max(x1, x2) < lon_min or min(x1, x2) > lon_max or max(y1, y2) < lat_min or min(y1, y2) > lat_max:
        return False
    if (lon_min <= x1 <= lon_max and lat_min <= y1 <= lat_max) or (
        lon_min <= x2 <= lon_max and lat_min <= y2 <= lat_max
    ):
        return True

    edges = (
        (lon_min, lat_min, lon_max, lat_min),
        (lon_max, lat_min, lon_max, lat_max),
        (lon_max, lat_max, lon_min, lat_max),
        (lon_min, lat_max, lon_min, lat_min),
    )
    return any(_segments_intersect(x1, y1, x2, y2, *edge) for edge in edges)


def _segments_intersect(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    def orientation(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> float:
        return (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

    def on_segment(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
        tolerance = 1e-12
        return (
            min(px, rx) - tolerance <= qx <= max(px, rx) + tolerance
            and min(py, ry) - tolerance <= qy <= max(py, ry) + tolerance
        )

    o1 = orientation(ax, ay, bx, by, cx, cy)
    o2 = orientation(ax, ay, bx, by, dx, dy)
    o3 = orientation(cx, cy, dx, dy, ax, ay)
    o4 = orientation(cx, cy, dx, dy, bx, by)

    tolerance = 1e-12
    if abs(o1) <= tolerance and on_segment(ax, ay, cx, cy, bx, by):
        return True
    if abs(o2) <= tolerance and on_segment(ax, ay, dx, dy, bx, by):
        return True
    if abs(o3) <= tolerance and on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if abs(o4) <= tolerance and on_segment(cx, cy, bx, by, dx, dy):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _geometry_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    points = list(_iter_geometry_points(geometry))
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))


def _iter_geometry_points(geometry: dict[str, Any]):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        for ring in coordinates:
            for point in ring:
                yield point
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            for ring in polygon:
                for point in ring:
                    yield point


def _point_in_geometry(longitude: float, latitude: float, geometry: dict[str, Any]) -> bool:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        return _point_in_polygon(longitude, latitude, coordinates)
    if geometry_type == "MultiPolygon":
        return any(_point_in_polygon(longitude, latitude, polygon) for polygon in coordinates)
    return False


def _point_in_polygon(longitude: float, latitude: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon:
        return False
    if not _point_in_ring(longitude, latitude, polygon[0]):
        return False
    for hole in polygon[1:]:
        if _point_in_ring(longitude, latitude, hole):
            return False
    return True


def _point_in_ring(longitude: float, latitude: float, ring: list[list[float]]) -> bool:
    inside = False
    point_count = len(ring)
    if point_count < 3:
        return False
    for index in range(point_count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % point_count]
        intersects = ((y1 > latitude) != (y2 > latitude)) and (
            longitude < (x2 - x1) * (latitude - y1) / ((y2 - y1) or 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


def _view_mode_is_available(
    settings: Settings,
    theme: str,
    view_mode: ViewMode,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> bool:
    selection = _resolve_selection(settings, theme, season_profile=season_profile, subseason=subseason)
    canonical = _canonical_manifest_path(
        settings,
        theme,
        view_mode,
        season_profile=season_profile,
        subseason=subseason,
    )
    if _manifest_file_is_usable(settings, selection, view_mode, canonical):
        return True
    legacy = _legacy_manifest_path(settings, theme, view_mode)
    return _manifest_file_is_usable(settings, selection, view_mode, legacy)


def _manifest_file_is_usable(
    settings: Settings,
    selection: ForecastProductSelection,
    view_mode: ViewMode,
    manifest_path: Path,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _manifest_payload_is_usable_for_selection(settings, selection, view_mode, manifest)


def _manifest_payload_is_usable_for_selection(
    settings: Settings,
    selection: ForecastProductSelection,
    view_mode: ViewMode,
    manifest: dict[str, Any],
) -> bool:
    try:
        if not _manifest_payload_is_available_for_selection(settings, selection, manifest):
            return False
        data_path = Path(str(manifest["data_path"]))
        if not _manifest_source_policy_allows_selection(settings, selection, manifest):
            return False
        _validate_product_dataset_for_selection(settings, selection, view_mode, data_path)
    except Exception:
        return False
    return True


def _manifest_payload_is_available_for_selection(
    settings: Settings,
    selection: ForecastProductSelection,
    manifest: dict[str, Any],
) -> bool:
    try:
        data_path = Path(str(manifest["data_path"]))
    except Exception:
        return False
    return (
        data_path.exists()
        and data_path.stat().st_size > 0
        and _manifest_source_policy_allows_selection(settings, selection, manifest)
    )


def _usable_manifest_path(
    settings: Settings,
    selection: ForecastProductSelection,
    view_mode: ViewMode,
) -> Path | None:
    canonical = _canonical_manifest_path(
        settings,
        selection.theme,
        view_mode,
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )
    if _manifest_file_is_usable(settings, selection, view_mode, canonical):
        return canonical
    legacy = _legacy_manifest_path(settings, selection.theme, view_mode)
    if _manifest_file_is_usable(settings, selection, view_mode, legacy):
        return legacy
    return None


def _selection_product_pair_is_compatible(settings: Settings, selection: ForecastProductSelection) -> bool:
    probability_manifest_path = _usable_manifest_path(settings, selection, "probability")
    deterministic_manifest_path = _usable_manifest_path(settings, selection, "deterministic")
    if probability_manifest_path is None or deterministic_manifest_path is None:
        return False
    try:
        probability_manifest = json.loads(probability_manifest_path.read_text(encoding="utf-8"))
        deterministic_manifest = json.loads(deterministic_manifest_path.read_text(encoding="utf-8"))
        probability_path = Path(str(probability_manifest["data_path"]))
        deterministic_path = Path(str(deterministic_manifest["data_path"]))
        with xr.open_dataset(probability_path) as probability_dataset, xr.open_dataset(deterministic_path) as deterministic_dataset:
            probability_var = probability_dataset[list(probability_dataset.data_vars)[0]]
            deterministic_var = deterministic_dataset[list(deterministic_dataset.data_vars)[0]]
            _validate_product_dimensions("probability", probability_var)
            _validate_product_dimensions("deterministic", deterministic_var)
            probability_y = np.asarray(probability_dataset["Y"].values, dtype=float)
            probability_x = np.asarray(probability_dataset["X"].values, dtype=float)
            deterministic_y = np.asarray(deterministic_dataset["Y"].values, dtype=float)
            deterministic_x = np.asarray(deterministic_dataset["X"].values, dtype=float)
            return (
                probability_var.sizes["Y"] == deterministic_var.sizes["Y"]
                and probability_var.sizes["X"] == deterministic_var.sizes["X"]
                and np.allclose(probability_y, deterministic_y)
                and np.allclose(probability_x, deterministic_x)
            )
    except Exception:
        return False


def _manifest_source_policy_allows_selection(
    settings: Settings,
    selection: ForecastProductSelection,
    manifest: dict[str, Any],
) -> bool:
    generation_backend = str(manifest.get("generation_backend") or "").lower()
    source_artifact_type = str(manifest.get("source_artifact_type") or "").lower()
    is_daily_derived = "daily_wass2s" in generation_backend or source_artifact_type == "daily_wass2s_derived"
    if _selection_requires_profile_derived_onset(selection) and not is_daily_derived:
        return False
    if selection.theme in FINAL_PRODUCT_ONLY_THEMES and is_daily_derived:
        return False
    if _final_product_source_for_selection(settings, selection) is not None and is_daily_derived:
        return False
    return True


def _selection_view_is_materializable(
    settings: Settings,
    selection: ForecastProductSelection,
    view_mode: ViewMode,
) -> bool:
    if _selection_requires_profile_derived_onset(selection):
        return _selection_can_derive_from_daily(settings, selection)
    final_source = _final_product_source_for_selection(settings, selection)
    if final_source is not None:
        return _final_product_source_has_file(final_source, view_mode)
    if _selection_can_derive_from_daily(settings, selection):
        return True
    return _has_source_file(_source_for_theme(settings, selection.theme), view_mode)


def _selection_requires_profile_derived_onset(selection: ForecastProductSelection) -> bool:
    return selection.theme == "onset" and selection.season_profile in PROFILE_DERIVED_ONSET_PROFILES


def _selection_has_complete_refresh_source(settings: Settings, selection: ForecastProductSelection) -> bool:
    return (
        _view_mode_is_available(
            settings,
            selection.theme,
            "probability",
            season_profile=selection.season_profile,
            subseason=selection.subseason,
        )
        or _selection_view_is_materializable(settings, selection, "probability")
    ) and (
        _view_mode_is_available(
            settings,
            selection.theme,
            "deterministic",
            season_profile=selection.season_profile,
            subseason=selection.subseason,
        )
        or _selection_view_is_materializable(settings, selection, "deterministic")
    )


def _refreshable_selections(settings: Settings, theme: str | None = None) -> list[ForecastProductSelection]:
    requested = [_normalize_theme(theme)] if theme is not None else list(THEME_SPECS)
    selections: list[ForecastProductSelection] = []
    for candidate in requested:
        selections.extend(_all_selections_for_theme(settings, candidate))
    return [selection for selection in selections if _selection_has_complete_refresh_source(settings, selection)]


def _all_selections_for_theme(settings: Settings, theme: str) -> list[ForecastProductSelection]:
    normalized_theme = _normalize_theme(theme)
    if normalized_theme in SEASON_BASED_THEMES:
        return [
            _resolve_selection(settings, normalized_theme, season_profile=season_profile)
            for season_profile in settings.seasonal_map.profiles
        ]
    if normalized_theme in SUBSEASON_BASED_THEMES:
        return [
            _resolve_selection(settings, normalized_theme, subseason=subseason)
            for subseason in _supported_subseasons(settings)
        ]
    return [_resolve_selection(settings, normalized_theme)]


def _resolve_selection(
    settings: Settings,
    theme: str,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> ForecastProductSelection:
    normalized_theme = _normalize_theme(theme)
    _theme_spec(normalized_theme)
    requires_season = normalized_theme in SEASON_BASED_THEMES
    requires_subseason = normalized_theme in SUBSEASON_BASED_THEMES
    normalized_season = str(season_profile).strip().lower() if season_profile else None
    normalized_subseason = str(subseason).strip().upper() if subseason else None

    if requires_season:
        if normalized_season is None:
            raise InvalidForecastProductSelectionError(
                f"Forecast product theme '{normalized_theme}' requires a season_profile selector."
            )
        if normalized_season not in settings.seasonal_map.profiles:
            allowed = ", ".join(settings.seasonal_map.profiles)
            raise InvalidForecastProductSelectionError(
                f"Invalid season_profile '{season_profile}' for theme '{normalized_theme}'. Expected one of: {allowed}."
            )
    else:
        normalized_season = None

    if requires_subseason:
        supported_subseasons = _supported_subseasons(settings)
        if normalized_subseason is None:
            raise InvalidForecastProductSelectionError(
                f"Forecast product theme '{normalized_theme}' requires a subseason selector."
            )
        if normalized_subseason not in supported_subseasons:
            allowed = ", ".join(supported_subseasons)
            raise InvalidForecastProductSelectionError(
                f"Invalid subseason '{subseason}' for theme '{normalized_theme}'. Expected one of: {allowed}."
            )
    else:
        normalized_subseason = None

    season_label = settings.seasonal_map.profiles[normalized_season].label if normalized_season is not None else None
    return ForecastProductSelection(
        theme=normalized_theme,
        season_profile=normalized_season,
        season_label=season_label,
        subseason=normalized_subseason,
        subseason_label=normalized_subseason,
        requires_season=requires_season,
        requires_subseason=requires_subseason,
    )


def _supported_subseasons(settings: Settings) -> list[str]:
    discovered: list[str] = []
    for candidate in SUBSEASON_DISPLAY_ORDER:
        if any(candidate in profile.calendar_subseasons for profile in settings.seasonal_map.profiles.values()):
            discovered.append(candidate)
    extras = {
        item
        for profile in settings.seasonal_map.profiles.values()
        for item in profile.calendar_subseasons
        if item not in SUBSEASON_DISPLAY_ORDER
    }
    discovered.extend(sorted(extras))
    return discovered


def _source_for_theme(settings: Settings, theme: str) -> ForecastProductSourceConfig:
    normalized_theme = _normalize_theme(theme)
    try:
        return settings.forecast_products.products[normalized_theme]
    except KeyError as exc:
        raise InvalidForecastProductThemeError(
            f"Unsupported forecast product theme '{theme}'. Supported values: {', '.join(sorted(THEME_SPECS))}."
        ) from exc


def _has_source_artifacts(source: ForecastProductSourceConfig) -> bool:
    return _has_source_file(source, "probability") and _has_source_file(source, "deterministic")


def _has_source_file(source: ForecastProductSourceConfig, view_mode: ViewMode) -> bool:
    source_path = source.probability_path if view_mode == "probability" else source.deterministic_path
    return source_path is not None and source_path.exists()


def _canonical_manifest_path(
    settings: Settings,
    theme: str,
    view_mode: ViewMode,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> Path:
    return _product_scope_path(
        settings,
        theme,
        view_mode,
        season_profile=season_profile,
        subseason=subseason,
    ) / "active.json"


def _legacy_manifest_path(settings: Settings, theme: str, view_mode: ViewMode) -> Path:
    return settings.forecast_products.artifact_dir / theme / view_mode / "active.json"


def _product_scope_path(
    settings: Settings,
    theme: str,
    view_mode: ViewMode,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> Path:
    scope = settings.forecast_products.artifact_dir / theme / view_mode
    if season_profile is not None:
        scope = scope / season_profile
    if subseason is not None:
        scope = scope / subseason.lower()
    return scope


def _promote_manifest_to_selection(
    settings: Settings,
    manifest: dict[str, Any],
    theme: str,
    view_mode: ViewMode,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> dict[str, Any]:
    manifest_path = _canonical_manifest_path(
        settings,
        theme,
        view_mode,
        season_profile=season_profile,
        subseason=subseason,
    )
    ensure_directory(manifest_path.parent)
    normalized = dict(manifest)
    normalized["theme"] = theme
    normalized["view_mode"] = view_mode
    normalized["season_profile"] = season_profile
    normalized["subseason"] = subseason
    normalized["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def _normalize_manifest_selection(
    manifest: dict[str, Any],
    selection: ForecastProductSelection,
    manifest_path: Path,
) -> dict[str, Any]:
    normalized = dict(manifest)
    normalized["theme"] = selection.theme
    normalized["season_profile"] = selection.season_profile
    normalized["subseason"] = selection.subseason
    normalized["manifest_path"] = str(manifest_path)
    return normalized


def _artifacts_not_available_message(
    theme: str,
    view_mode: ViewMode,
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> str:
    parts = [f"theme={theme}", f"view_mode={view_mode}"]
    if season_profile is not None:
        parts.append(f"season_profile={season_profile}")
    if subseason is not None:
        parts.append(f"subseason={subseason}")
    return (
        "No active forecast product exists for "
        + " ".join(parts)
        + ". The generated forecast artifacts for this seasonal selection are not available yet."
    )


def _asset_query_path(
    view_mode: ViewMode,
    suffix: str,
    *,
    theme: str,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> str:
    params: dict[str, str] = {"theme": theme}
    if season_profile is not None:
        params["season_profile"] = season_profile
    if subseason is not None:
        params["subseason"] = subseason
    return f"/forecast/{view_mode}/{suffix}?{urlencode(params)}"


def _resolve_browser_url(api_base_url: str | None, path_or_url: str | None) -> str | None:
    if path_or_url is None:
        return None
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    if not api_base_url:
        return path_or_url
    return f"{api_base_url.rstrip('/')}{path_or_url}"


def _theme_spec(theme: str) -> ProductThemeSpec:
    normalized = _normalize_theme(theme)
    try:
        return THEME_SPECS[normalized]
    except KeyError as exc:
        raise InvalidForecastProductThemeError(
            f"Unsupported forecast product theme '{theme}'. Supported values: {', '.join(sorted(THEME_SPECS))}."
        ) from exc


def _normalize_theme(theme: str) -> str:
    return str(theme).strip().lower()


def _probability_legend(spec: ProductThemeSpec) -> list[dict[str, Any]]:
    legend = []
    for order, (category_code, label, hint, color) in enumerate(spec.probability_categories):
        legend.append(
            {
                "category_code": category_code,
                "label": label,
                "hint": hint,
                "color": color,
                "display_order": order,
            }
        )
    return legend


def _clean_probability_grid(probabilities: np.ndarray) -> np.ndarray:
    cleaned = np.array(probabilities, dtype=float, copy=True)
    sums = np.nansum(np.where(np.isfinite(cleaned), cleaned, 0.0), axis=0)
    cleaned[:, sums <= 0.0] = np.nan
    return cleaned


def _bounds_payload(latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, float]:
    return {
        "latitude_min": round(float(np.nanmin(latitudes)), 4),
        "latitude_max": round(float(np.nanmax(latitudes)), 4),
        "longitude_min": round(float(np.nanmin(longitudes)), 4),
        "longitude_max": round(float(np.nanmax(longitudes)), 4),
    }


def _grid_shape_payload(latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, int]:
    return {"y": int(latitudes.size), "x": int(longitudes.size)}


def _grid_resolution_payload(latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, float | None]:
    return {
        "latitude": _axis_resolution_degrees(latitudes),
        "longitude": _axis_resolution_degrees(longitudes),
    }


def _axis_resolution_degrees(axis_values: np.ndarray) -> float | None:
    values = np.asarray(axis_values, dtype=float)
    if values.size < 2:
        return None
    diffs = np.abs(np.diff(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    return round(float(np.median(diffs)), 6)


def _manifest_source_artifact_type(manifest: dict[str, Any]) -> str:
    generation_backend = str(manifest.get("generation_backend") or "").lower()
    source_artifact_type = str(manifest.get("source_artifact_type") or "").lower()
    if source_artifact_type == "daily_wass2s_derived" or "daily_wass2s" in generation_backend:
        return "daily_wass2s_derived"
    return "final_netcdf"


def _manifest_is_low_resolution_fallback(manifest: dict[str, Any]) -> bool:
    return _manifest_source_artifact_type(manifest) == "daily_wass2s_derived"


def _resolve_nearest_valid_index(
    valid_mask: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    latitude: float,
    longitude: float,
) -> tuple[int, int]:
    if not np.any(valid_mask):
        raise ForecastProductIncompleteError("Forecast product does not contain any valid cells.")
    y_idx = _nearest_axis_index(latitudes, latitude)
    x_idx = _nearest_axis_index(longitudes, longitude)
    if valid_mask[y_idx, x_idx]:
        return y_idx, x_idx
    valid_positions = np.argwhere(valid_mask)
    distances = (valid_positions[:, 0] - y_idx) ** 2 + (valid_positions[:, 1] - x_idx) ** 2
    nearest = valid_positions[int(np.argmin(distances))]
    return int(nearest[0]), int(nearest[1])


def _nearest_axis_index(axis_values: np.ndarray, sample_value: float) -> int:
    if axis_values.size == 0:
        raise ForecastProductIncompleteError("Forecast product grid is empty.")
    indices = np.searchsorted(axis_values, np.asarray([sample_value], dtype=float))
    indices = np.clip(indices, 0, len(axis_values) - 1)
    previous = np.clip(indices - 1, 0, len(axis_values) - 1)
    choose_previous = np.abs(sample_value - axis_values[previous]) <= np.abs(sample_value - axis_values[indices])
    return int(np.where(choose_previous, previous, indices)[0])


def _format_deterministic_display_value(theme: str, forecast_year: int, value: float) -> str:
    if theme in {"onset", "cessation"}:
        start_of_year = pd.Timestamp(year=forecast_year, month=1, day=1, tz="UTC")
        resolved = start_of_year + pd.Timedelta(days=max(value - 1.0, 0.0))
        return resolved.strftime("%d %b")
    if theme in {"early_dry_spell", "late_dry_spell", "rainy_days"}:
        return f"{value:.1f} day(s)"
    if theme == "rainfall_amount":
        return f"{value:.1f} mm"
    return f"{value:.1f}"


def _deterministic_interpretation(theme: str, value: float) -> str:
    if theme == "onset":
        return "Earlier values indicate earlier onset timing; larger values indicate later onset timing."
    if theme == "cessation":
        return "Earlier values indicate earlier cessation timing; larger values indicate later cessation timing."
    if theme == "early_dry_spell":
        return "Smaller values indicate shorter early-season dry spells; larger values indicate longer dry spells."
    if theme == "late_dry_spell":
        return "Smaller values indicate shorter late-season dry spells; larger values indicate longer dry spells."
    if theme == "rainfall_amount":
        return "Smaller values indicate lower seasonal rainfall totals; larger values indicate higher totals."
    if theme == "rainy_days":
        return "Smaller values indicate fewer rainy days; larger values indicate more rainy days."
    return "Deterministic forecast value sampled from the nearest grid cell."


def _build_probability_preview_png(data_path: Path, theme: str) -> bytes:
    prepared = _prepare_probability_preview_payload(data_path, theme)
    rgba = _probability_preview_rgba(prepared)
    return _encode_png(rgba)


def _build_deterministic_preview_png(data_path: Path, theme: str) -> bytes:
    with xr.open_dataset(data_path) as dataset:
        data_var = dataset[list(dataset.data_vars)[0]].isel(T=0)
        values = np.asarray(data_var.values, dtype=float)
    spec = _theme_spec(theme)
    rgba = _continuous_grid_rgba(values, spec.deterministic_color_ramp)
    return _encode_png(rgba)


def _prepare_probability_preview_payload(data_path: Path, theme: str) -> PreparedProbabilityProduct:
    spec = _theme_spec(theme)
    with xr.open_dataset(data_path) as dataset:
        data_var = dataset[list(dataset.data_vars)[0]].isel(T=0)
        category_codes = tuple(str(item) for item in data_var.coords["probability"].values.tolist())
        latitudes = np.asarray(dataset["Y"].values, dtype=float)
        longitudes = np.asarray(dataset["X"].values, dtype=float)
        probabilities = _clean_probability_grid(np.asarray(data_var.values, dtype=float))
        valid_time = _to_utc_datetime(data_var.coords["T"].item())
    return PreparedProbabilityProduct(
        theme=theme,
        theme_label=spec.theme_label,
        season_profile=None,
        season_label=None,
        subseason=None,
        subseason_label=None,
        product_id="preview",
        forecast_year=2025,
        valid_time=valid_time,
        generated_at=datetime.now(UTC),
        refresh_interval_seconds=1800,
        freshness_threshold_hours=18,
        source_label="preview",
        source_run_id="preview",
        generation_backend="preview",
        source_artifact_type="final_netcdf",
        is_low_resolution_fallback=False,
        category_codes=category_codes,
        legend=tuple(_probability_legend(spec)),
        latitudes=latitudes,
        longitudes=longitudes,
        probabilities=probabilities,
        mask_zone=GHANA_PRODUCT_MASK_ZONE,
        preview_url=None,
    )


def _probability_preview_rgba(prepared: PreparedProbabilityProduct) -> np.ndarray:
    dominant, confidence = _dominant_probability_layers(prepared.probabilities)
    legend_lookup = {item["category_code"]: item for item in prepared.legend}
    rgba = np.zeros((dominant.shape[0], dominant.shape[1], 4), dtype=np.uint8)
    for idx, category_code in enumerate(prepared.category_codes):
        color = legend_lookup[category_code]["color"]
        rgb = tuple(int(color[offset : offset + 2], 16) for offset in (1, 3, 5))
        mask = dominant == idx
        rgba[..., 0] = np.where(mask, rgb[0], rgba[..., 0])
        rgba[..., 1] = np.where(mask, rgb[1], rgba[..., 1])
        rgba[..., 2] = np.where(mask, rgb[2], rgba[..., 2])
    alpha = np.where(np.isfinite(confidence), np.clip(confidence * 255.0, 96, 235), 0.0).astype(np.uint8)
    rgba[..., 3] = alpha
    return _flip_latitude_if_needed(rgba, prepared.latitudes)


def _continuous_grid_rgba(values: np.ndarray, color_ramp: tuple[tuple[float, str], ...]) -> np.ndarray:
    rgba = np.zeros((values.shape[0], values.shape[1], 4), dtype=np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return rgba
    lower_bound = float(np.nanmin(finite))
    upper_bound = float(np.nanmax(finite))
    normalized = np.full(values.shape, 0.0, dtype=float)
    if not math.isclose(lower_bound, upper_bound):
        normalized = (values - lower_bound) / (upper_bound - lower_bound)
    normalized = np.clip(normalized, 0.0, 1.0)
    red, green, blue = _interpolate_color_channels(normalized, color_ramp)
    rgba[..., 0] = red
    rgba[..., 1] = green
    rgba[..., 2] = blue
    rgba[..., 3] = np.where(np.isfinite(values), 248, 0).astype(np.uint8)
    return np.flipud(rgba)


def _flip_latitude_if_needed(rgba: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    if latitudes.size >= 2 and latitudes[0] < latitudes[-1]:
        return np.flipud(rgba)
    return rgba


def _manifest_preview_url(
    theme: str,
    view_mode: ViewMode,
    manifest: dict[str, Any],
    *,
    season_profile: str | None = None,
    subseason: str | None = None,
) -> str | None:
    preview_path = Path(str(manifest.get("preview_path") or ""))
    if not preview_path.exists():
        return None
    return _asset_query_path(
        view_mode,
        "preview.png",
        theme=theme,
        season_profile=season_profile,
        subseason=subseason,
    )


def _dominant_probability_layers(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe = np.where(np.isfinite(probabilities), probabilities, -np.inf)
    dominant = np.argmax(safe, axis=0)
    confidence = np.max(safe, axis=0)
    confidence = np.where(np.isfinite(confidence), confidence, np.nan)
    return dominant, confidence


def _to_utc_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _render_probability_tile_png(
    settings: Settings,
    prepared: PreparedProbabilityProduct,
    *,
    z: int,
    x: int,
    y: int,
) -> bytes:
    longitudes = _tile_pixel_longitudes(z, x)
    latitudes = _tile_pixel_latitudes(z, y)
    sampled_probabilities, lat_inside, lon_inside = _nearest_sample_probability_grid(
        prepared.probabilities,
        prepared.latitudes,
        prepared.longitudes,
        latitudes,
        longitudes,
    )
    dominant_sampled, confidence_sampled = _dominant_probability_layers(sampled_probabilities)
    geometry_mask = _tile_geometry_mask(settings, prepared.mask_zone, z=z, x=x, y=y)
    valid_mask = np.isfinite(confidence_sampled) & lat_inside[:, None] & lon_inside[None, :] & geometry_mask
    legend_lookup = {item["category_code"]: item for item in prepared.legend}
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    for idx, category_code in enumerate(prepared.category_codes):
        color = legend_lookup[category_code]["color"]
        rgb = tuple(int(color[offset : offset + 2], 16) for offset in (1, 3, 5))
        mask = (dominant_sampled == idx) & valid_mask
        rgba[..., 0] = np.where(mask, rgb[0], rgba[..., 0])
        rgba[..., 1] = np.where(mask, rgb[1], rgba[..., 1])
        rgba[..., 2] = np.where(mask, rgb[2], rgba[..., 2])
    alpha_min = FALLBACK_PROBABILITY_TILE_ALPHA_MIN if prepared.is_low_resolution_fallback else FINAL_PROBABILITY_TILE_ALPHA_MIN
    alpha_max = FALLBACK_PROBABILITY_TILE_ALPHA_MAX if prepared.is_low_resolution_fallback else FINAL_PROBABILITY_TILE_ALPHA_MAX
    alpha = np.where(np.isfinite(confidence_sampled), np.clip(confidence_sampled * 255.0, alpha_min, alpha_max), 0.0)
    rgba[..., 3] = np.where(valid_mask, alpha, 0).astype(np.uint8)
    return _encode_png(rgba)


def _render_deterministic_tile_png(
    settings: Settings,
    prepared: PreparedDeterministicProduct,
    *,
    z: int,
    x: int,
    y: int,
) -> bytes:
    longitudes = _tile_pixel_longitudes(z, x)
    latitudes = _tile_pixel_latitudes(z, y)
    sampled, lat_inside, lon_inside = _nearest_sample_grid(
        prepared.values,
        prepared.latitudes,
        prepared.longitudes,
        latitudes,
        longitudes,
    )
    geometry_mask = _tile_geometry_mask(settings, prepared.mask_zone, z=z, x=x, y=y)
    valid_mask = np.isfinite(sampled) & lat_inside[:, None] & lon_inside[None, :] & geometry_mask
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    if np.any(valid_mask):
        if math.isclose(prepared.lower_bound, prepared.upper_bound):
            normalized = np.full(sampled.shape, 0.5, dtype=float)
        else:
            normalized = (sampled - prepared.lower_bound) / (prepared.upper_bound - prepared.lower_bound)
        normalized = np.clip(normalized, 0.0, 1.0)
        red, green, blue = _interpolate_color_channels(normalized, prepared.color_ramp)
        rgba[..., 0] = red
        rgba[..., 1] = green
        rgba[..., 2] = blue
        alpha = FALLBACK_DETERMINISTIC_TILE_ALPHA if prepared.is_low_resolution_fallback else FINAL_DETERMINISTIC_TILE_ALPHA
        rgba[..., 3] = np.where(valid_mask, alpha, 0).astype(np.uint8)
    return _encode_png(rgba)


def _tile_geometry_mask(settings: Settings, native_zone: str, *, z: int, x: int, y: int) -> np.ndarray:
    normalized_zone = str(native_zone).strip().lower()
    if normalized_zone not in {GHANA_PRODUCT_MASK_ZONE, "north", "south"}:
        return np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)

    mask_rows = _tile_geometry_mask_cached(
        str(settings.seasonal_map.district_geojson_path),
        float(settings.seasonal_map.northern_latitude_threshold),
        normalized_zone,
        int(z),
        int(x),
        int(y),
    )
    return np.asarray(mask_rows, dtype=bool)


@lru_cache(maxsize=64)
def _tile_geometry_mask_cached(
    geojson_path: str,
    threshold: float,
    native_zone: str,
    z: int,
    x: int,
    y: int,
) -> tuple[tuple[bool, ...], ...]:
    features = _load_district_zone_features(geojson_path, threshold)
    longitudes = _tile_sample_longitudes(z, x, TILE_GEOMETRY_MASK_SIZE).tolist()
    latitudes = _tile_sample_latitudes(z, y, TILE_GEOMETRY_MASK_SIZE).tolist()
    mask: list[tuple[bool, ...]] = []
    for latitude in latitudes:
        row = [
            _point_matches_any_zone_feature(float(longitude), float(latitude), native_zone, features)
            for longitude in longitudes
        ]
        mask.append(tuple(row))
    return _upsample_tile_geometry_mask(tuple(mask), TILE_SIZE)


def _tile_sample_longitudes(z: int, x: int, sample_size: int) -> np.ndarray:
    pixel_positions = (np.arange(sample_size, dtype=float) + 0.5) / sample_size
    return ((x + pixel_positions) / (2**z)) * 360.0 - 180.0


def _tile_sample_latitudes(z: int, y: int, sample_size: int) -> np.ndarray:
    pixel_positions = (np.arange(sample_size, dtype=float) + 0.5) / sample_size
    mercator = math.pi * (1 - 2 * ((y + pixel_positions) / (2**z)))
    return np.degrees(np.arctan(np.sinh(mercator)))


def _upsample_tile_geometry_mask(
    coarse_mask: tuple[tuple[bool, ...], ...],
    target_size: int,
) -> tuple[tuple[bool, ...], ...]:
    if not coarse_mask:
        return tuple()
    row_scale = max(1, target_size // len(coarse_mask))
    col_scale = max(1, target_size // len(coarse_mask[0]))
    expanded_rows: list[tuple[bool, ...]] = []
    for row in coarse_mask:
        expanded_row = tuple(value for value in row for _ in range(col_scale))
        expanded_rows.extend([expanded_row[:target_size]] * row_scale)
    return tuple(expanded_rows[:target_size])


def _point_matches_any_zone_feature(
    longitude: float,
    latitude: float,
    native_zone: str,
    features: tuple[dict[str, Any], ...],
) -> bool:
    for feature in features:
        if native_zone != GHANA_PRODUCT_MASK_ZONE and native_zone not in feature["zones"]:
            continue
        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat = feature["bbox"]
        if longitude < bbox_min_lon or longitude > bbox_max_lon or latitude < bbox_min_lat or latitude > bbox_max_lat:
            continue
        if _point_in_geometry(longitude, latitude, feature["geometry"]):
            return True
    return False


def _nearest_indices(axis_values: np.ndarray, sample_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis_values, dtype=float)
    finite_axis_indices = np.flatnonzero(np.isfinite(axis))
    if finite_axis_indices.size == 0:
        raise ForecastProductIncompleteError("Forecast product grid coordinates are empty.")

    ordered_indices = finite_axis_indices[np.argsort(axis[finite_axis_indices])]
    ordered_values = axis[ordered_indices]
    samples = np.asarray(sample_values, dtype=float)
    inside = np.isfinite(samples) & (samples >= ordered_values[0]) & (samples <= ordered_values[-1])

    if ordered_values.size == 1:
        return np.full(samples.shape, int(ordered_indices[0]), dtype=int), inside

    upper_positions = np.searchsorted(ordered_values, samples, side="left")
    upper_positions = np.clip(upper_positions, 0, ordered_values.size - 1)
    lower_positions = np.clip(upper_positions - 1, 0, ordered_values.size - 1)
    choose_lower = np.abs(samples - ordered_values[lower_positions]) <= np.abs(samples - ordered_values[upper_positions])
    nearest_positions = np.where(choose_lower, lower_positions, upper_positions)
    return ordered_indices[nearest_positions].astype(int), inside


def _tile_pixel_longitudes(z: int, x: int) -> np.ndarray:
    pixel_positions = (np.arange(TILE_SIZE, dtype=float) + 0.5) / TILE_SIZE
    return ((x + pixel_positions) / (2**z)) * 360.0 - 180.0


def _tile_pixel_latitudes(z: int, y: int) -> np.ndarray:
    pixel_positions = (np.arange(TILE_SIZE, dtype=float) + 0.5) / TILE_SIZE
    mercator = math.pi * (1 - 2 * ((y + pixel_positions) / (2**z)))
    return np.degrees(np.arctan(np.sinh(mercator)))


def _nearest_sample_probability_grid(
    probabilities: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sample_latitudes: np.ndarray,
    sample_longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_layers: list[np.ndarray] = []
    lat_inside: np.ndarray | None = None
    lon_inside: np.ndarray | None = None
    for category_values in probabilities:
        sampled, current_lat_inside, current_lon_inside = _nearest_sample_grid(
            category_values,
            latitudes,
            longitudes,
            sample_latitudes,
            sample_longitudes,
        )
        sampled_layers.append(sampled)
        lat_inside = current_lat_inside
        lon_inside = current_lon_inside

    if not sampled_layers:
        raise ForecastProductIncompleteError("Probability product does not contain any probability categories.")
    sampled_probabilities = np.stack(sampled_layers, axis=0)
    clipped = np.where(np.isfinite(sampled_probabilities), np.clip(sampled_probabilities, 0.0, None), 0.0)
    totals = np.sum(clipped, axis=0)
    normalized = np.divide(
        clipped,
        totals[None, :, :],
        out=np.full_like(clipped, np.nan, dtype=float),
        where=totals[None, :, :] > 0.0,
    )
    assert lat_inside is not None
    assert lon_inside is not None
    return normalized, lat_inside, lon_inside


def _nearest_sample_grid(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sample_latitudes: np.ndarray,
    sample_longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ForecastProductIncompleteError("Nearest-neighbor tile sampling expects a two-dimensional forecast grid.")
    lat_indices, lat_inside = _nearest_indices(latitudes, sample_latitudes)
    lon_indices, lon_inside = _nearest_indices(longitudes, sample_longitudes)
    sampled = values[np.ix_(lat_indices, lon_indices)]
    return sampled, lat_inside, lon_inside


def _bilinear_sample_probability_grid(
    probabilities: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sample_latitudes: np.ndarray,
    sample_longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_layers: list[np.ndarray] = []
    lat_inside: np.ndarray | None = None
    lon_inside: np.ndarray | None = None
    for category_values in probabilities:
        sampled, current_lat_inside, current_lon_inside = _bilinear_sample_grid(
            category_values,
            latitudes,
            longitudes,
            sample_latitudes,
            sample_longitudes,
        )
        sampled_layers.append(sampled)
        lat_inside = current_lat_inside
        lon_inside = current_lon_inside

    if not sampled_layers:
        raise ForecastProductIncompleteError("Probability product does not contain any probability categories.")
    sampled_probabilities = np.stack(sampled_layers, axis=0)
    clipped = np.where(np.isfinite(sampled_probabilities), np.clip(sampled_probabilities, 0.0, None), 0.0)
    totals = np.sum(clipped, axis=0)
    normalized = np.divide(
        clipped,
        totals[None, :, :],
        out=np.full_like(clipped, np.nan, dtype=float),
        where=totals[None, :, :] > 0.0,
    )
    assert lat_inside is not None
    assert lon_inside is not None
    return normalized, lat_inside, lon_inside


def _bilinear_sample_grid(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sample_latitudes: np.ndarray,
    sample_longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ForecastProductIncompleteError("Bilinear tile sampling expects a two-dimensional forecast grid.")
    lat_lower, lat_upper, lat_weight, lat_inside = _axis_bilinear_bounds(latitudes, sample_latitudes)
    lon_lower, lon_upper, lon_weight, lon_inside = _axis_bilinear_bounds(longitudes, sample_longitudes)

    v00 = values[np.ix_(lat_lower, lon_lower)]
    v01 = values[np.ix_(lat_lower, lon_upper)]
    v10 = values[np.ix_(lat_upper, lon_lower)]
    v11 = values[np.ix_(lat_upper, lon_upper)]
    wy = lat_weight[:, None]
    wx = lon_weight[None, :]
    sampled = _weighted_bilinear_average(
        (v00, (1.0 - wy) * (1.0 - wx)),
        (v01, (1.0 - wy) * wx),
        (v10, wy * (1.0 - wx)),
        (v11, wy * wx),
    )
    return sampled, lat_inside, lon_inside


def _axis_bilinear_bounds(axis_values: np.ndarray, sample_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis_values, dtype=float)
    finite_axis_indices = np.flatnonzero(np.isfinite(axis))
    if finite_axis_indices.size == 0:
        raise ForecastProductIncompleteError("Forecast product grid coordinates are empty.")
    ordered_indices = finite_axis_indices[np.argsort(axis[finite_axis_indices])]
    ordered_values = axis[ordered_indices]
    samples = np.asarray(sample_values, dtype=float)
    inside = np.isfinite(samples) & (samples >= ordered_values[0]) & (samples <= ordered_values[-1])

    if ordered_values.size == 1:
        only = np.full(samples.shape, int(ordered_indices[0]), dtype=int)
        return only, only, np.zeros(samples.shape, dtype=float), inside

    upper_positions = np.searchsorted(ordered_values, samples, side="left")
    upper_positions = np.clip(upper_positions, 1, ordered_values.size - 1)
    lower_positions = upper_positions - 1
    lower_values = ordered_values[lower_positions]
    upper_values = ordered_values[upper_positions]
    denominator = upper_values - lower_values
    weight = np.divide(
        samples - lower_values,
        denominator,
        out=np.zeros(samples.shape, dtype=float),
        where=denominator != 0.0,
    )
    weight = np.clip(weight, 0.0, 1.0)
    return ordered_indices[lower_positions].astype(int), ordered_indices[upper_positions].astype(int), weight, inside


def _weighted_bilinear_average(*corners: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    numerator = np.zeros(corners[0][0].shape, dtype=float)
    denominator = np.zeros(corners[0][0].shape, dtype=float)
    for values, weights in corners:
        finite = np.isfinite(values)
        numerator += np.where(finite, values * weights, 0.0)
        denominator += np.where(finite, weights, 0.0)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan, dtype=float), where=denominator > 0.0)


def _interpolate_color_channels(
    normalized: np.ndarray,
    color_ramp: tuple[tuple[float, str], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    stops = np.asarray([stop for stop, _ in color_ramp], dtype=float)
    reds = np.asarray([int(color[1:3], 16) for _, color in color_ramp], dtype=float)
    greens = np.asarray([int(color[3:5], 16) for _, color in color_ramp], dtype=float)
    blues = np.asarray([int(color[5:7], 16) for _, color in color_ramp], dtype=float)
    return (
        np.interp(safe_normalized, stops, reds).astype(np.uint8),
        np.interp(safe_normalized, stops, greens).astype(np.uint8),
        np.interp(safe_normalized, stops, blues).astype(np.uint8),
    )


def _encode_png(rgba: np.ndarray) -> bytes:
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("PNG encoder expects an RGBA array.")
    raw = b"".join(b"\x00" + rgba[row_index].tobytes() for row_index in range(height))
    compressed = zlib.compress(raw, level=6)
    header = struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        ]
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return b"".join(
        [
            struct.pack("!I", len(payload)),
            chunk_type,
            payload,
            struct.pack("!I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF),
        ]
    )
