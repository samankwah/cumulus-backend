"""Cumulus-managed seasonal forecast artifact products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
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
from cumulus.settings import ForecastProductSourceConfig, Settings
from cumulus.utils.io import ensure_directory

TILE_SIZE = 256
ViewMode = Literal["probability", "deterministic"]
SEASON_BASED_THEMES = frozenset({"onset", "cessation", "early_dry_spell", "late_dry_spell"})
SUBSEASON_BASED_THEMES = frozenset({"rainfall_amount", "rainy_days"})
SUBSEASON_DISPLAY_ORDER = ("MAM", "AMJ", "MJJ", "JJA", "JAS", "SON")
DETERMINISTIC_REFERENCE_COLOR_RAMP = (
    (0.0, "#440154"),
    (0.25, "#3b528b"),
    (0.5, "#21918c"),
    (0.75, "#8fd744"),
    (1.0, "#fde725"),
)


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
    lower_bound: float
    upper_bound: float
    legend_ticks: tuple[float, ...]
    color_ramp: tuple[tuple[float, str], ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    values: np.ndarray
    unit: str
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
    category_codes: tuple[str, ...]
    legend: tuple[dict[str, Any], ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    probabilities: np.ndarray
    preview_url: str | None


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
    dominant, confidence = _dominant_probability_layers(prepared.probabilities)
    return _render_probability_tile_png(prepared, dominant, confidence, z=z, x=x, y=y)


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
    return _render_deterministic_tile_png(prepared, z=z, x=x, y=y)


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
    with xr.open_dataset(manifest["data_path"]) as dataset:
        data_var = dataset[list(dataset.data_vars)[0]].isel(T=0)
        category_codes = tuple(str(item) for item in data_var.coords["probability"].values.tolist())
        probabilities = np.asarray(data_var.values, dtype=float)
        latitudes = np.asarray(dataset["Y"].values, dtype=float)
        longitudes = np.asarray(dataset["X"].values, dtype=float)
        valid_time = _to_utc_datetime(data_var.coords["T"].item())
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
        category_codes=category_codes,
        legend=tuple(_probability_legend(spec)),
        latitudes=latitudes,
        longitudes=longitudes,
        probabilities=probabilities,
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
        lower_bound=round(lower_bound, 1),
        upper_bound=round(upper_bound, 1),
        legend_ticks=tuple(round(float(item), 1) for item in np.linspace(lower_bound, upper_bound, num=5)),
        color_ramp=spec.deterministic_color_ramp,
        latitudes=latitudes,
        longitudes=longitudes,
        values=values,
        unit=spec.deterministic_unit,
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
    source_path = source.probability_path if view_mode == "probability" else source.deterministic_path
    if source_path is None or not source_path.exists():
        raise ForecastProductArtifactsNotAvailableError(f"Forecast source file is missing: {source_path}")

    product_dir = _product_scope_path(
        settings,
        source.theme,
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
    if source.preview_path is not None and source.preview_path.exists():
        temp_preview_path = preview_path.with_suffix(".png.tmp")
        if temp_preview_path.exists():
            temp_preview_path.unlink()
        shutil.copyfile(source.preview_path, temp_preview_path)
        temp_preview_path.replace(preview_path)
    else:
        if view_mode == "probability":
            preview_bytes = _build_probability_preview_png(copied_path, source.theme)
        else:
            preview_bytes = _build_deterministic_preview_png(copied_path, source.theme)
        preview_path.write_bytes(preview_bytes)

    generated_at = datetime.now(UTC)
    suffix = ""
    if season_profile is not None:
        suffix = f"{suffix}_{season_profile}"
    if subseason is not None:
        suffix = f"{suffix}_{subseason.lower()}"
    product_id = f"{source.theme}_{view_mode}{suffix}_{source.forecast_year}_{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "product_id": product_id,
        "theme": source.theme,
        "view_mode": view_mode,
        "season_profile": season_profile,
        "subseason": subseason,
        "title": source.title,
        "forecast_year": source.forecast_year,
        "generated_at": generated_at.isoformat(),
        "source_label": settings.forecast_products.source_label,
        "source_run_id": product_id,
        "generation_backend": settings.forecast_products.generation_backend,
        "refresh_interval_seconds": settings.forecast_products.refresh_interval_seconds,
        "freshness_threshold_hours": settings.forecast_products.freshness_threshold_hours,
        "data_path": str(copied_path),
        "preview_path": str(preview_path),
    }
    manifest_path = product_dir / "active.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


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
    if not manifest_path.exists():
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
        elif _has_source_artifacts(source):
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
    data_path = Path(str(manifest["data_path"]))
    if not data_path.exists() or data_path.stat().st_size == 0:
        if not materialize_missing:
            raise ForecastProductArtifactsNotAvailableError(
                _artifacts_not_available_message(
                    normalized_theme,
                    view_mode,
                    season_profile=selection.season_profile,
                    subseason=selection.subseason,
                )
            )
        if _has_source_artifacts(source):
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
    seasons = list(settings.seasonal_map.profiles) if requires_season else []
    subseasons = _supported_subseasons(settings) if requires_subseason else []
    enabled = _theme_is_enabled(settings, normalized_theme)
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
    return _view_mode_is_available(
        settings,
        selection.theme,
        "probability",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    ) and _view_mode_is_available(
        settings,
        selection.theme,
        "deterministic",
        season_profile=selection.season_profile,
        subseason=selection.subseason,
    )


def _apply_selection_spatial_mask(
    settings: Settings,
    selection: ForecastProductSelection,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if selection.season_profile is None:
        return values

    profile = settings.seasonal_map.profiles.get(selection.season_profile)
    if profile is None:
        return values

    cell_mask = _district_zone_cell_mask(settings, profile.native_zone, latitudes, longitudes)
    if not np.any(cell_mask):
        raise ForecastProductArtifactsNotAvailableError(
            f"No forecast cells overlap the native zone for season_profile='{selection.season_profile}'."
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
            f"No valid forecast cells remain after applying the native zone for season_profile='{selection.season_profile}'."
    )
    return masked


def _district_zone_cell_mask(
    settings: Settings,
    native_zone: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    normalized_zone = str(native_zone).strip().lower()
    if normalized_zone not in {"north", "south"}:
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
    mask: list[tuple[bool, ...]] = []
    for latitude in latitude_key:
        row: list[bool] = []
        for longitude in longitude_key:
            row.append(_point_matches_any_zone_feature(longitude, latitude, native_zone, features))
        mask.append(tuple(row))
    return tuple(mask)


@lru_cache(maxsize=4)
def _load_district_zone_features(geojson_path: str, threshold: float) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        min_lon, min_lat, max_lon, max_lat = _geometry_bbox(geometry)
        zones: set[str] = set()
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


def _point_matches_any_zone_feature(
    longitude: float,
    latitude: float,
    native_zone: str,
    features: tuple[dict[str, Any], ...],
) -> bool:
    for feature in features:
        if native_zone not in feature["zones"]:
            continue
        min_lon, min_lat, max_lon, max_lat = feature["bbox"]
        if longitude < min_lon or longitude > max_lon or latitude < min_lat or latitude > max_lat:
            continue
        if _point_in_geometry(longitude, latitude, feature["geometry"]):
            return True
    return False


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
    canonical = _canonical_manifest_path(
        settings,
        theme,
        view_mode,
        season_profile=season_profile,
        subseason=subseason,
    )
    if canonical.exists():
        return True
    legacy = _legacy_manifest_path(settings, theme, view_mode)
    if legacy.exists():
        return True
    return _has_source_file(_source_for_theme(settings, theme), view_mode)


def _refreshable_selections(settings: Settings, theme: str | None = None) -> list[ForecastProductSelection]:
    requested = [_normalize_theme(theme)] if theme is not None else list(THEME_SPECS)
    selections: list[ForecastProductSelection] = []
    for candidate in requested:
        if _theme_is_enabled(settings, candidate):
            selections.extend(_all_selections_for_theme(settings, candidate))
    return [selection for selection in selections if _selection_has_complete_artifacts(settings, selection)]


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


def _bounds_payload(latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, float]:
    return {
        "latitude_min": round(float(np.nanmin(latitudes)), 4),
        "latitude_max": round(float(np.nanmax(latitudes)), 4),
        "longitude_min": round(float(np.nanmin(longitudes)), 4),
        "longitude_max": round(float(np.nanmax(longitudes)), 4),
    }


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
        probabilities = np.asarray(data_var.values, dtype=float)
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
        category_codes=category_codes,
        legend=tuple(_probability_legend(spec)),
        latitudes=latitudes,
        longitudes=longitudes,
        probabilities=probabilities,
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
    rgba[..., 3] = np.where(np.isfinite(values), 220, 0).astype(np.uint8)
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
    prepared: PreparedProbabilityProduct,
    dominant: np.ndarray,
    confidence: np.ndarray,
    *,
    z: int,
    x: int,
    y: int,
) -> bytes:
    longitudes = _tile_pixel_longitudes(z, x)
    latitudes = _tile_pixel_latitudes(z, y)
    lon_indices, lon_inside = _nearest_indices(prepared.longitudes, longitudes)
    lat_indices, lat_inside = _nearest_indices(prepared.latitudes, latitudes)
    dominant_sampled = dominant[np.ix_(lat_indices, lon_indices)]
    confidence_sampled = confidence[np.ix_(lat_indices, lon_indices)]
    valid_mask = np.isfinite(confidence_sampled) & lat_inside[:, None] & lon_inside[None, :]
    legend_lookup = {item["category_code"]: item for item in prepared.legend}
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    for idx, category_code in enumerate(prepared.category_codes):
        color = legend_lookup[category_code]["color"]
        rgb = tuple(int(color[offset : offset + 2], 16) for offset in (1, 3, 5))
        mask = (dominant_sampled == idx) & valid_mask
        rgba[..., 0] = np.where(mask, rgb[0], rgba[..., 0])
        rgba[..., 1] = np.where(mask, rgb[1], rgba[..., 1])
        rgba[..., 2] = np.where(mask, rgb[2], rgba[..., 2])
    alpha = np.where(np.isfinite(confidence_sampled), np.clip(confidence_sampled * 255.0, 96, 235), 0.0)
    rgba[..., 3] = np.where(valid_mask, alpha, 0).astype(np.uint8)
    return _encode_png(rgba)


def _render_deterministic_tile_png(prepared: PreparedDeterministicProduct, *, z: int, x: int, y: int) -> bytes:
    longitudes = _tile_pixel_longitudes(z, x)
    latitudes = _tile_pixel_latitudes(z, y)
    lon_indices, lon_inside = _nearest_indices(prepared.longitudes, longitudes)
    lat_indices, lat_inside = _nearest_indices(prepared.latitudes, latitudes)
    sampled = prepared.values[np.ix_(lat_indices, lon_indices)]
    valid_mask = np.isfinite(sampled) & lat_inside[:, None] & lon_inside[None, :]
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
        rgba[..., 3] = np.where(valid_mask, 220, 0).astype(np.uint8)
    return _encode_png(rgba)


def _nearest_indices(axis_values: np.ndarray, sample_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(axis_values, sample_values)
    indices = np.clip(indices, 0, len(axis_values) - 1)
    previous = np.clip(indices - 1, 0, len(axis_values) - 1)
    choose_previous = np.abs(sample_values - axis_values[previous]) <= np.abs(sample_values - axis_values[indices])
    nearest = np.where(choose_previous, previous, indices)
    inside = (sample_values >= axis_values.min()) & (sample_values <= axis_values.max())
    return nearest.astype(int), inside


def _tile_pixel_longitudes(z: int, x: int) -> np.ndarray:
    pixel_positions = (np.arange(TILE_SIZE, dtype=float) + 0.5) / TILE_SIZE
    return ((x + pixel_positions) / (2**z)) * 360.0 - 180.0


def _tile_pixel_latitudes(z: int, y: int) -> np.ndarray:
    pixel_positions = (np.arange(TILE_SIZE, dtype=float) + 0.5) / TILE_SIZE
    mercator = math.pi * (1 - 2 * ((y + pixel_positions) / (2**z)))
    return np.degrees(np.arctan(np.sinh(mercator)))


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
