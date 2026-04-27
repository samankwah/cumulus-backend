"""Artifact-backed Ghana seasonal advisory map generation and serving."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from cumulus.api.errors import (
    InvalidSeasonalModeError,
    InvalidSubseasonForProfileError,
    SeasonalMapArtifactsNotAvailableError,
    SeasonalProbabilityProductIncompleteError,
    SubseasonNotAllowedError,
    SubseasonRequiredError,
)
from cumulus.data.extractors import extract_locations
from cumulus.modeling.predictor import predict_dataframe
from cumulus.services.source_resolution import (
    ResolvedForecastSource,
    normalize_forecast_source_id,
    open_source_dataset,
    resolve_forecast_source,
)
from cumulus.settings import SeasonalProfileConfig, Settings
from cumulus.utils.io import ensure_directory, read_json, write_json


logger = logging.getLogger(__name__)

SEASONAL_THEMES = (
    "onset",
    "cessation",
    "early_dry_spell",
    "late_dry_spell",
    "rainfall_amount",
    "rainy_days",
)
REGIME_BOUND_THEMES = (
    "onset",
    "cessation",
    "early_dry_spell",
    "late_dry_spell",
)
SEASONAL_ONLY_THEMES = REGIME_BOUND_THEMES
SEASONAL_MODES = ("seasonal", "calendar")
CALENDAR_CAPABLE_THEMES = ("rainfall_amount", "rainy_days")
CALENDAR_SUBSEASON_MONTHS = {
    "MAM": (3, 4, 5),
    "AMJ": (4, 5, 6),
    "MJJ": (5, 6, 7),
    "JJA": (6, 7, 8),
    "JAS": (7, 8, 9),
    "SON": (9, 10, 11),
}

THEME_LABELS = {
    "onset": "Onset Date",
    "cessation": "Cessation Date",
    "early_dry_spell": "Early-Season Dry Spell",
    "late_dry_spell": "Late-Season Dry Spell",
    "rainfall_amount": "Seasonal Rainfall Total",
    "rainy_days": "Number of Rainy Days",
}
MODE_LABELS = {"seasonal": "Seasonal", "calendar": "Calendar"}

WASS2S_PROBABILITY_FAMILY_COLORS = {
    "teal": "#2f8f86",
    "neutral": "#b8b9b4",
    "brown": "#b47a34",
}


@dataclass(frozen=True)
class SeasonalRefreshCombination:
    theme: str
    season_profile: str
    mode: str
    subseason: str | None = None

    def to_payload(self) -> dict[str, str | None]:
        return {
            "theme": self.theme,
            "season_profile": self.season_profile,
            "mode": self.mode,
            "subseason": self.subseason,
        }


def generate_seasonal_map_product(
    settings: Settings,
    theme: str,
    season_profile: str,
    *,
    mode: str,
    subseason: str | None = None,
    forecast_source: str | None = None,
    resolved_source_override: ResolvedForecastSource | None = None,
) -> dict[str, Any]:
    normalized_theme = _resolve_theme(theme)
    normalized_profile, profile = _resolve_profile_config(settings, season_profile)
    normalized_mode, normalized_subseason = _resolve_mode_and_subseason(
        normalized_theme,
        profile,
        mode,
        subseason,
    )
    resolved_source = resolved_source_override or resolve_forecast_source(settings, forecast_source)
    district_catalog, predicted, metadata = _prepare_generation_inputs(settings, resolved_source)
    return _build_seasonal_map_product(
        settings,
        normalized_theme=normalized_theme,
        normalized_profile=normalized_profile,
        profile=profile,
        normalized_mode=normalized_mode,
        normalized_subseason=normalized_subseason,
        resolved_source=resolved_source,
        district_catalog=district_catalog,
        predicted=predicted,
        metadata=metadata,
    )


def _prepare_generation_inputs(
    settings: Settings,
    resolved_source: ResolvedForecastSource,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    district_catalog = _load_district_catalog(
        str(settings.seasonal_map.district_geojson_path),
        settings.seasonal_map.northern_latitude_threshold,
    )
    if not district_catalog:
        raise ValueError(f"No district features were found in {settings.seasonal_map.district_geojson_path}.")

    logger.info(
        "seasonal_map.prepare_inputs forecast_source=%s districts=%s",
        resolved_source.source_id,
        len(district_catalog),
    )
    dataset = open_source_dataset(settings, resolved_source)
    locations = pd.DataFrame(
        [
            {
                "location_id": district["location_id"],
                "latitude": district["latitude"],
                "longitude": district["longitude"],
            }
            for district in district_catalog
        ]
    )
    extracted = extract_locations(dataset, locations, list(dataset.data_vars))
    predicted, metadata = predict_dataframe(extracted, settings, forecast_source=resolved_source.source_id)
    predicted = predicted.sort_values(["location_id", "time"]).reset_index(drop=True)
    return district_catalog, predicted, metadata


def _build_seasonal_map_product(
    settings: Settings,
    *,
    normalized_theme: str,
    normalized_profile: str,
    profile: SeasonalProfileConfig,
    normalized_mode: str,
    normalized_subseason: str | None,
    resolved_source: ResolvedForecastSource,
    district_catalog: list[dict[str, Any]],
    predicted: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    forecast_cycle = _build_forecast_cycle(predicted)
    probability_district_items: list[dict[str, Any]] = []
    deterministic_district_items: list[dict[str, Any]] = []
    district_metrics: list[dict[str, Any]] = []
    grouped = {
        str(location_id): frame.reset_index(drop=True)
        for location_id, frame in predicted.groupby("location_id", sort=False)
    }
    for district in district_catalog:
        if _theme_uses_regime_footprint(normalized_theme) and not _district_matches_profile_footprint(district, profile):
            continue
        frame = grouped.get(district["location_id"], pd.DataFrame(columns=["time", "rainfall_corrected_mm"]))
        raw_metrics = _derive_raw_metrics(frame, district, profile, settings)
        district_metrics.append({"region_name": district["region_name"], **raw_metrics})
        coverage_note = _district_coverage_note(
            district=district,
            profile=profile,
            theme=normalized_theme,
        )
        probability_district_items.append(
            _serialize_probability_area_item(
                geography_type="district",
                geography_name=district["geography_name"],
                region_name=district["region_name"],
                location_id=district["location_id"],
                coverage_count=1,
                coverage_note=coverage_note,
                raw_metrics=raw_metrics,
                theme=normalized_theme,
                profile=profile,
                mode=normalized_mode,
                subseason=normalized_subseason,
                generated_at=generated_at,
            )
        )
        deterministic_district_items.append(
            _serialize_deterministic_area_item(
                geography_type="district",
                geography_name=district["geography_name"],
                region_name=district["region_name"],
                location_id=district["location_id"],
                coverage_count=1,
                coverage_note=coverage_note,
                raw_metrics=raw_metrics,
                theme=normalized_theme,
                profile=profile,
                mode=normalized_mode,
                subseason=normalized_subseason,
                generated_at=generated_at,
            )
        )

    probability_region_items, deterministic_region_items = _build_region_items(
        district_metrics,
        theme=normalized_theme,
        profile=profile,
        mode=normalized_mode,
        subseason=normalized_subseason,
        generated_at=generated_at,
    )
    product_id = (
        f"seasonal_{resolved_source.source_id}_{normalized_profile}_{normalized_theme}_{normalized_mode}"
        f"{f'_{normalized_subseason.lower()}' if normalized_subseason else ''}_{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_id = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ensure_directory(
        _artifact_scope_path(
            settings,
            resolved_source.source_id,
            normalized_profile,
            normalized_theme,
            normalized_mode,
            normalized_subseason,
        )
        / "runs"
        / run_id
    )
    product_path = run_dir / "product.json"
    manifest_path = run_dir / "manifest.json"
    pointer_path = _active_pointer_path(
        settings,
        resolved_source.source_id,
        normalized_profile,
        normalized_theme,
        normalized_mode,
        normalized_subseason,
    )

    base_payload = {
        "product_id": product_id,
        "theme": normalized_theme,
        "season_profile": normalized_profile,
        "mode": normalized_mode,
        "subseason": normalized_subseason,
        "mode_label": MODE_LABELS[normalized_mode],
        "subseason_label": normalized_subseason,
        "generated_at": generated_at.isoformat(),
        "forecast_cycle": forecast_cycle,
        "forecast_source": resolved_source.source_id,
        "forecast_source_label": _forecast_source_label(resolved_source.source_id),
        "source_run_id": resolved_source.source_run_id,
        "refresh_interval_seconds": int(settings.seasonal_map.refresh_interval_minutes * 60),
        "freshness_threshold_hours": int(settings.seasonal_map.freshness_threshold_hours),
        "district_count": len(probability_district_items),
        "region_count": len(probability_region_items),
        "legend": _build_probability_legend(normalized_theme, normalized_mode, profile, normalized_subseason),
        "district_items": probability_district_items,
        "region_items": probability_region_items,
        "deterministic_legend": _build_deterministic_legend(
            normalized_theme,
            normalized_mode,
            profile,
            normalized_subseason,
            generated_at,
        ),
        "deterministic_district_items": deterministic_district_items,
        "deterministic_region_items": deterministic_region_items,
    }
    product_payload = {
        **base_payload,
        "refresh_status": "fresh",
        "is_stale": False,
    }
    manifest = {
        **base_payload,
        "product_path": str(product_path),
        "manifest_path": str(manifest_path),
        "model_version": str(metadata.get("model_version", "unknown")),
    }
    write_json(product_path, product_payload)
    write_json(manifest_path, manifest)
    write_json(
        pointer_path,
        {
            "product_id": product_id,
            "product_path": str(product_path),
            "manifest_path": str(manifest_path),
            "mode": normalized_mode,
            "subseason": normalized_subseason,
        },
    )
    if normalized_mode == "seasonal" and normalized_subseason is None:
        write_json(
            _legacy_active_pointer_path(
                settings,
                resolved_source.source_id,
                normalized_profile,
                normalized_theme,
            ),
            {
                "product_id": product_id,
                "product_path": str(product_path),
                "manifest_path": str(manifest_path),
                "mode": normalized_mode,
                "subseason": normalized_subseason,
            },
        )
    clear_seasonal_map_cache()
    logger.info(
        "seasonal_map.generate_success theme=%s season_profile=%s mode=%s subseason=%s forecast_source=%s product_id=%s",
        normalized_theme,
        normalized_profile,
        normalized_mode,
        normalized_subseason,
        resolved_source.source_id,
        product_id,
    )
    return manifest


def generate_all_seasonal_map_products(
    settings: Settings,
    *,
    forecast_source: str | None = None,
) -> list[dict[str, Any]]:
    summary = refresh_seasonal_map_products(settings, forecast_source=forecast_source, include_manifests=True)
    return [item["manifest"] for item in summary["succeeded"]]


def refresh_seasonal_map_products(
    settings: Settings,
    *,
    theme: str | None = None,
    season_profile: str | None = None,
    mode: str | None = None,
    subseason: str | None = None,
    forecast_source: str | None = None,
    resolved_source_override: ResolvedForecastSource | None = None,
    include_manifests: bool = False,
) -> dict[str, Any]:
    resolved_source = resolved_source_override or resolve_forecast_source(settings, forecast_source)
    combinations = _build_refresh_combinations(
        settings,
        theme=theme,
        season_profile=season_profile,
        mode=mode,
        subseason=subseason,
    )
    attempted = [item.to_payload() for item in combinations]
    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    district_catalog, predicted, metadata = _prepare_generation_inputs(settings, resolved_source)

    logger.info(
        "seasonal_map.refresh_start forecast_source=%s combinations=%s theme=%s season_profile=%s mode=%s subseason=%s",
        resolved_source.source_id,
        len(combinations),
        theme,
        season_profile,
        mode,
        subseason,
    )
    for combination in combinations:
        try:
            normalized_profile, profile = _resolve_profile_config(settings, combination.season_profile)
            normalized_mode, normalized_subseason = _resolve_mode_and_subseason(
                combination.theme,
                profile,
                combination.mode,
                combination.subseason,
            )
            manifest = _build_seasonal_map_product(
                settings,
                normalized_theme=combination.theme,
                normalized_profile=normalized_profile,
                profile=profile,
                normalized_mode=normalized_mode,
                normalized_subseason=normalized_subseason,
                resolved_source=resolved_source,
                district_catalog=district_catalog,
                predicted=predicted,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception(
                "seasonal_map.refresh_failed theme=%s season_profile=%s mode=%s subseason=%s forecast_source=%s",
                combination.theme,
                combination.season_profile,
                combination.mode,
                combination.subseason,
                resolved_source.source_id,
            )
            failed.append({**combination.to_payload(), "error": str(exc)})
            continue
        success_item = {
            **combination.to_payload(),
            "product_id": manifest["product_id"],
            "generated_at": manifest["generated_at"],
            "active_pointer_path": str(
                _active_pointer_path(
                    settings,
                    resolved_source.source_id,
                    combination.season_profile,
                    combination.theme,
                    combination.mode,
                    combination.subseason,
                )
            ),
            "legacy_active_pointer_path": (
                str(
                    _legacy_active_pointer_path(
                        settings,
                        resolved_source.source_id,
                        combination.season_profile,
                        combination.theme,
                    )
                )
                if combination.mode == "seasonal" and combination.subseason is None
                else None
            ),
        }
        if include_manifests:
            success_item["manifest"] = manifest
        succeeded.append(success_item)

    summary = {
        "forecast_source": resolved_source.source_id,
        "forecast_source_label": _forecast_source_label(resolved_source.source_id),
        "requested_theme": _resolve_theme(theme) if theme is not None else None,
        "requested_season_profile": (
            _resolve_profile_config(settings, season_profile)[0] if season_profile is not None else None
        ),
        "requested_mode": str(mode).strip().lower() if mode is not None else None,
        "requested_subseason": str(subseason).strip().upper() if subseason is not None else None,
        "attempted_count": len(attempted),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
    }
    logger.info(
        "seasonal_map.refresh_complete forecast_source=%s attempted=%s succeeded=%s failed=%s",
        resolved_source.source_id,
        summary["attempted_count"],
        summary["succeeded_count"],
        summary["failed_count"],
    )
    return summary


def get_active_seasonal_map_product(
    settings: Settings,
    theme: str,
    season_profile: str,
    *,
    mode: str,
    subseason: str | None = None,
    forecast_source: str | None = None,
) -> dict[str, Any]:
    payload = _load_active_product_payload(
        settings,
        theme,
        season_profile,
        mode=mode,
        subseason=subseason,
        forecast_source=forecast_source,
    )
    return _as_probability_product_payload(payload)


def get_active_deterministic_seasonal_map_product(
    settings: Settings,
    theme: str,
    season_profile: str,
    *,
    mode: str,
    subseason: str | None = None,
    forecast_source: str | None = None,
) -> dict[str, Any]:
    payload = _load_active_product_payload(
        settings,
        theme,
        season_profile,
        mode=mode,
        subseason=subseason,
        forecast_source=forecast_source,
    )
    return _as_deterministic_product_payload(payload)


def _load_active_product_payload(
    settings: Settings,
    theme: str,
    season_profile: str,
    *,
    mode: str,
    subseason: str | None = None,
    forecast_source: str | None = None,
) -> dict[str, Any]:
    normalized_theme = _resolve_theme(theme)
    normalized_profile, profile = _resolve_profile_config(settings, season_profile)
    normalized_mode, normalized_subseason = _resolve_mode_and_subseason(
        normalized_theme,
        profile,
        mode,
        subseason,
    )
    source_id = (
        normalize_forecast_source_id(forecast_source)
        or normalize_forecast_source_id(settings.default_forecast_source)
        or "configured"
    )
    pointer_path = _resolve_active_pointer_path(
        settings,
        source_id,
        normalized_profile,
        normalized_theme,
        normalized_mode,
        normalized_subseason,
    )
    try:
        pointer = read_json(pointer_path)
    except FileNotFoundError as exc:
        pointer = {}
        fallback_product_path = _discover_latest_product_path(
            settings,
            source_id,
            normalized_profile,
            normalized_theme,
            normalized_mode,
            normalized_subseason,
        )
        if fallback_product_path is None:
            raise SeasonalMapArtifactsNotAvailableError(
                (
                    "No active seasonal map product is available for "
                    f"theme={normalized_theme} season_profile={normalized_profile} mode={normalized_mode}"
                    f"{f' subseason={normalized_subseason}' if normalized_subseason else ''} under {settings.seasonal_map.artifact_dir}."
                )
            ) from exc
        pointer["product_path"] = str(fallback_product_path)
    product_path = _resolve_product_path(
        pointer,
        settings=settings,
        source_id=source_id,
        season_profile=normalized_profile,
        theme=normalized_theme,
        mode=normalized_mode,
        subseason=normalized_subseason,
    )
    payload = _normalize_product_payload(
        _read_json_cached(str(product_path.resolve())),
        source_id=source_id,
        theme=normalized_theme,
        season_profile=normalized_profile,
        mode=normalized_mode,
        subseason=normalized_subseason,
    )
    generated_at = datetime.fromisoformat(str(payload["generated_at"]))
    is_stale = datetime.now(UTC) - generated_at > timedelta(hours=settings.seasonal_map.freshness_threshold_hours)
    return {
        **payload,
        "refresh_status": "stale" if is_stale else "fresh",
        "is_stale": is_stale,
    }


def list_supported_season_profiles(settings: Settings) -> list[str]:
    return list(settings.seasonal_map.profiles.keys())


def get_seasonal_map_options(settings: Settings) -> dict[str, Any]:
    return {
        "themes": {
            theme: {
                "modes": list(SEASONAL_MODES if theme in CALENDAR_CAPABLE_THEMES else ("seasonal",)),
                "subseasons": [],
            }
            for theme in SEASONAL_THEMES
        },
        "profiles": {
            profile_id: {
                "label": profile.label,
                "calendar_subseasons": list(profile.calendar_subseasons),
            }
            for profile_id, profile in settings.seasonal_map.profiles.items()
        },
    }


def clear_seasonal_map_cache() -> None:
    _read_json_cached.cache_clear()
    _load_district_catalog.cache_clear()


def _build_refresh_combinations(
    settings: Settings,
    *,
    theme: str | None = None,
    season_profile: str | None = None,
    mode: str | None = None,
    subseason: str | None = None,
) -> list[SeasonalRefreshCombination]:
    requested_theme = _resolve_theme(theme) if theme is not None else None
    requested_mode = _normalize_mode_filter(mode)
    requested_subseason = str(subseason).strip().upper() if subseason is not None else None
    if requested_subseason is not None and requested_mode != "calendar":
        raise SubseasonNotAllowedError("Sub-season filters require mode=calendar.")

    profiles: list[tuple[str, SeasonalProfileConfig]]
    if season_profile is None:
        profiles = list(settings.seasonal_map.profiles.items())
    else:
        profiles = [_resolve_profile_config(settings, season_profile)]

    theme_ids = [requested_theme] if requested_theme is not None else list(SEASONAL_THEMES)
    combinations: list[SeasonalRefreshCombination] = []
    for profile_id, profile in profiles:
        for theme_id in theme_ids:
            combinations.extend(
                _combinations_for_theme_profile(
                    theme_id,
                    profile_id,
                    profile,
                    requested_theme=requested_theme,
                    requested_mode=requested_mode,
                    requested_subseason=requested_subseason,
                )
            )

    if not combinations:
        raise ValueError("No valid seasonal refresh combinations matched the supplied filters.")
    return combinations


def _normalize_mode_filter(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SEASONAL_MODES:
        supported = ", ".join(SEASONAL_MODES)
        raise InvalidSeasonalModeError(f"Unsupported mode '{mode}'. Expected one of: {supported}.")
    return normalized_mode


def _combinations_for_theme_profile(
    theme: str,
    season_profile: str,
    profile: SeasonalProfileConfig,
    *,
    requested_theme: str | None,
    requested_mode: str | None,
    requested_subseason: str | None,
) -> list[SeasonalRefreshCombination]:
    combinations: list[SeasonalRefreshCombination] = []
    include_seasonal = requested_mode == "seasonal" or (requested_mode is None and theme in SEASONAL_ONLY_THEMES)
    include_calendar = requested_mode == "calendar" or (requested_mode is None and theme in CALENDAR_CAPABLE_THEMES)

    if include_seasonal and requested_subseason is None:
        combinations.append(SeasonalRefreshCombination(theme=theme, season_profile=season_profile, mode="seasonal"))

    if not include_calendar:
        return combinations
    if theme not in CALENDAR_CAPABLE_THEMES:
        if requested_theme == theme and requested_mode == "calendar":
            raise InvalidSeasonalModeError(f"Calendar mode is not supported for theme={theme}.")
        return combinations

    candidate_subseasons = [requested_subseason] if requested_subseason is not None else list(profile.calendar_subseasons)
    if requested_subseason is not None and requested_subseason not in profile.calendar_subseasons:
        if requested_theme == theme and requested_mode == "calendar" and len(candidate_subseasons) == 1:
            _resolve_mode_and_subseason(theme, profile, "calendar", requested_subseason)
        return combinations
    for candidate_subseason in candidate_subseasons:
        normalized_mode, normalized_subseason = _resolve_mode_and_subseason(
            theme,
            profile,
            "calendar",
            candidate_subseason,
        )
        combinations.append(
            SeasonalRefreshCombination(
                theme=theme,
                season_profile=season_profile,
                mode=normalized_mode,
                subseason=normalized_subseason,
            )
        )
    return combinations


@lru_cache(maxsize=64)
def _read_json_cached(path: str) -> Any:
    return read_json(Path(path))


@lru_cache(maxsize=4)
def _load_district_catalog(path: str, northern_latitude_threshold: float) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", [])
    seen_ids: dict[str, int] = {}
    districts: list[dict[str, Any]] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        geography_name = str(properties.get("display_name") or properties.get("api_district") or "Unknown district")
        region_name = str(properties.get("region") or "Unknown region")
        api_district = str(properties.get("api_district") or properties.get("display_name") or "Unknown district")
        location_id = _slugify(api_district)
        next_suffix = seen_ids.get(location_id, 0)
        seen_ids[location_id] = next_suffix + 1
        if next_suffix:
            location_id = f"{location_id}-{next_suffix + 1}"
        latitude, longitude = _representative_point(geometry)
        districts.append(
            {
                "location_id": location_id,
                "geography_name": geography_name,
                "region_name": region_name,
                "latitude": latitude,
                "longitude": longitude,
                "regime_zone": _district_regime_zone(latitude, northern_latitude_threshold),
            }
        )
    return districts


def _resolve_theme(theme: str) -> str:
    normalized = str(theme).strip().lower()
    if normalized not in SEASONAL_THEMES:
        supported = ", ".join(SEASONAL_THEMES)
        raise ValueError(f"Unsupported theme '{theme}'. Expected one of: {supported}.")
    return normalized


def _resolve_profile_config(settings: Settings, season_profile: str) -> tuple[str, SeasonalProfileConfig]:
    normalized = str(season_profile).strip().lower()
    config = settings.seasonal_map.profiles.get(normalized)
    if config is None:
        supported = ", ".join(settings.seasonal_map.profiles)
        raise ValueError(f"Unsupported season_profile '{season_profile}'. Expected one of: {supported}.")
    return normalized, config


def _resolve_mode_and_subseason(
    theme: str,
    profile: SeasonalProfileConfig,
    mode: str,
    subseason: str | None,
) -> tuple[str, str | None]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SEASONAL_MODES:
        supported = ", ".join(SEASONAL_MODES)
        raise InvalidSeasonalModeError(f"Unsupported mode '{mode}'. Expected one of: {supported}.")

    normalized_subseason = str(subseason).strip().upper() if subseason else None
    if theme not in CALENDAR_CAPABLE_THEMES and normalized_mode != "seasonal":
        raise InvalidSeasonalModeError(f"Calendar mode is not supported for theme={theme}.")
    if normalized_mode == "seasonal" and normalized_subseason:
        raise SubseasonNotAllowedError(f"Sub-season is not supported for theme={theme} when mode=seasonal.")
    if normalized_mode == "calendar" and theme in CALENDAR_CAPABLE_THEMES and not normalized_subseason:
        raise SubseasonRequiredError(f"Sub-season is required for theme={theme} when mode=calendar.")
    if normalized_mode != "calendar":
        return normalized_mode, None
    if normalized_subseason not in profile.calendar_subseasons:
        allowed = ", ".join(profile.calendar_subseasons)
        raise InvalidSubseasonForProfileError(
            f"Sub-season '{normalized_subseason}' is not valid for profile={profile.label}. Expected one of: {allowed}."
        )
    return normalized_mode, normalized_subseason


def _build_forecast_cycle(predicted: pd.DataFrame) -> str:
    if predicted.empty or "time" not in predicted.columns:
        return "unavailable"
    latest = pd.to_datetime(predicted["time"], utc=True).max()
    return latest.strftime("%d %b %Y %H:%M UTC")


def _derive_raw_metrics(
    frame: pd.DataFrame,
    district: dict[str, Any],
    profile: SeasonalProfileConfig,
    settings: Settings,
) -> dict[str, float]:
    rainfall = frame.get("rainfall_corrected_mm", pd.Series(dtype=float)).fillna(0.0).astype(float)
    rainfall_values = rainfall.tolist() or [0.0]
    rainfall_year = _rainfall_year(frame)
    seasonal_rainfall = _build_synthetic_rainfall_series(
        rainfall_values,
        year=rainfall_year,
        district=district,
        profile=profile,
        settings=settings,
    )
    onset_date = _detect_onset(seasonal_rainfall, profile, settings)
    cessation_date = _detect_cessation(seasonal_rainfall, onset_date, profile)
    if cessation_date <= onset_date:
        fallback_index = min(len(seasonal_rainfall.index) - 1, seasonal_rainfall.index.get_loc(onset_date) + 60)
        cessation_date = seasonal_rainfall.index[fallback_index]

    season_slice = seasonal_rainfall.loc[onset_date:cessation_date]
    early_window_end = min(cessation_date, onset_date + pd.Timedelta(days=49))
    late_window_start = min(cessation_date, onset_date + pd.Timedelta(days=50))
    early_dry_spell = float(
        _longest_dry_spell(
            seasonal_rainfall.loc[onset_date:early_window_end],
            settings.seasonal_map.dry_day_threshold_mm,
        )
    )
    late_dry_spell = float(
        _longest_dry_spell(
            seasonal_rainfall.loc[late_window_start:cessation_date],
            settings.seasonal_map.dry_day_threshold_mm,
        )
    )
    rainfall_amount = float(round(float(season_slice.sum()), 1))
    rainy_days = float(round(float((season_slice >= settings.seasonal_map.rainy_day_threshold_mm).sum()), 1))
    season_length_days = float((cessation_date.date() - onset_date.date()).days)
    onset_reference_date = date(rainfall_year, profile.onset_reference_month, profile.onset_reference_day)
    cessation_reference_date = date(rainfall_year, profile.cessation_reference_month, profile.cessation_reference_day)
    zone_alignment = _profile_alignment_factor(district["regime_zone"], profile.native_zone)
    north_south = max(-0.14, min(0.14, (7.4 - float(district["latitude"])) * 0.025))

    rainfall_normal_mm = profile.rainfall_normal_mm * zone_alignment * (1.0 + north_south * 0.22)
    rainy_days_normal = profile.rainy_days_normal * zone_alignment * (1.0 + north_south * 0.12)
    raw_metrics = {
        "onset_offset_days": float((onset_date.date() - onset_reference_date).days),
        "cessation_offset_days": float((cessation_date.date() - cessation_reference_date).days),
        "early_dry_spell_days": float(round(early_dry_spell, 1)),
        "late_dry_spell_days": float(round(late_dry_spell, 1)),
        "rainfall_amount_mm": rainfall_amount,
        "rainfall_normal_mm": float(round(rainfall_normal_mm, 1)),
        "rainy_days_count": rainy_days,
        "rainy_days_normal": float(round(rainy_days_normal, 1)),
        "season_length_days": float(round(season_length_days, 1)),
    }
    for subseason in profile.calendar_subseasons:
        calendar_slice = _calendar_window_slice(seasonal_rainfall, subseason)
        calendar_rainfall = float(round(float(calendar_slice.sum()), 1))
        calendar_rainy_days = float(
            round(float((calendar_slice >= settings.seasonal_map.rainy_day_threshold_mm).sum()), 1)
        )
        rainfall_normal = float(
            round(profile.calendar_rainfall_normals_mm[subseason] * zone_alignment * (1.0 + north_south * 0.22), 1)
        )
        rainy_days_normal = float(
            round(profile.calendar_rainy_days_normals[subseason] * zone_alignment * (1.0 + north_south * 0.12), 1)
        )
        raw_metrics[f"calendar_rainfall_amount_mm_{subseason.lower()}"] = calendar_rainfall
        raw_metrics[f"calendar_rainfall_normal_mm_{subseason.lower()}"] = rainfall_normal
        raw_metrics[f"calendar_rainy_days_count_{subseason.lower()}"] = calendar_rainy_days
        raw_metrics[f"calendar_rainy_days_normal_{subseason.lower()}"] = rainy_days_normal
    return raw_metrics


def _build_synthetic_rainfall_series(
    rainfall_values: list[float],
    *,
    year: int,
    district: dict[str, Any],
    profile: SeasonalProfileConfig,
    settings: Settings,
) -> pd.Series:
    index = pd.date_range(
        start=pd.Timestamp(date(year, 1, 1), tz="UTC"),
        end=pd.Timestamp(date(year, 12, 31), tz="UTC"),
        freq="D",
    )
    onset_start = pd.Timestamp(date(year, profile.onset_search_start_month, profile.onset_search_start_day), tz="UTC")
    cessation_start = pd.Timestamp(date(year, profile.cessation_search_start_month, profile.cessation_search_start_day), tz="UTC")
    latitude = float(district["latitude"])
    longitude = float(district["longitude"])
    north_south = max(-0.16, min(0.16, (7.2 - latitude) * 0.03))
    east_west = max(-0.08, min(0.08, (longitude + 0.8) * 0.02))
    zone_alignment = _profile_alignment_factor(district["regime_zone"], profile.native_zone)
    base_scale = profile.rainfall_factor * zone_alignment * (1.0 + north_south + east_west)
    dry_day_cap = settings.seasonal_map.dry_day_threshold_mm * 0.22

    values: list[float] = []
    for position, timestamp in enumerate(index):
        repeated_value = rainfall_values[position % len(rainfall_values)]
        weight = _seasonal_weight(timestamp, onset_start, cessation_start)
        value = max(0.0, repeated_value * base_scale * weight)
        values.append(
            round(
                _apply_intraseasonal_structure(
                    value=value,
                    timestamp=timestamp,
                    onset_start=onset_start,
                    cessation_start=cessation_start,
                    latitude=latitude,
                    longitude=longitude,
                    profile=profile,
                    dry_day_cap=dry_day_cap,
                ),
                3,
            )
        )

    if max(values, default=0.0) <= 0:
        values = [0.0 for _ in index]
    return pd.Series(values, index=index, dtype=float)


def _seasonal_weight(timestamp: pd.Timestamp, onset_start: pd.Timestamp, cessation_start: pd.Timestamp) -> float:
    if timestamp < onset_start - pd.Timedelta(days=20):
        return 0.3
    if timestamp < onset_start:
        days = max(0, (timestamp - (onset_start - pd.Timedelta(days=20))).days)
        return 0.45 + (days / 20.0) * 0.45
    if timestamp <= cessation_start:
        span_days = max((cessation_start - onset_start).days, 1)
        progress = min(1.0, max(0.0, (timestamp - onset_start).days / span_days))
        return 0.95 + (1.0 - abs(progress - 0.5) * 2.0) * 0.28
    days_after = max(0, (timestamp - cessation_start).days)
    return max(0.14, 0.92 - days_after * 0.02)


def _apply_intraseasonal_structure(
    *,
    value: float,
    timestamp: pd.Timestamp,
    onset_start: pd.Timestamp,
    cessation_start: pd.Timestamp,
    latitude: float,
    longitude: float,
    profile: SeasonalProfileConfig,
    dry_day_cap: float,
) -> float:
    season_day = (timestamp - onset_start).days
    if season_day < 0:
        return value

    phase_shift = int(abs(latitude * 7 + longitude * 11)) % 7
    early_dry_start = 18 + phase_shift
    early_dry_length = max(
        3,
        min(
            14,
            profile.early_dry_spell_moderate_days
            + int(abs(latitude - 7.0) * 1.6)
            + (int(abs(longitude) * 10) % 3),
        ),
    )
    if early_dry_start <= season_day < early_dry_start + early_dry_length:
        return min(value * 0.08, dry_day_cap)

    late_dry_start = max(54, int((cessation_start - onset_start).days * 0.62)) + (phase_shift // 2)
    late_dry_length = max(
        4,
        min(
            18,
            profile.late_dry_spell_moderate_days
            + int(abs(latitude - 7.0) * 1.3)
            + (int(abs(longitude) * 10) % 4),
        ),
    )
    if late_dry_start <= season_day < late_dry_start + late_dry_length:
        return min(value * 0.18, dry_day_cap)

    return value


def _calendar_window_slice(series: pd.Series, subseason: str) -> pd.Series:
    months = CALENDAR_SUBSEASON_MONTHS[subseason]
    mask = series.index.month.isin(months)
    return series.loc[mask]


def _detect_onset(series: pd.Series, profile: SeasonalProfileConfig, settings: Settings) -> pd.Timestamp:
    start_date = pd.Timestamp(date(series.index[0].year, profile.onset_search_start_month, profile.onset_search_start_day), tz="UTC")
    start_idx = series.index.get_indexer([start_date], method="nearest")[0]
    values = series.tolist()
    window_days = max(1, profile.onset_window_days)

    for index in range(start_idx, len(values) - window_days + 1):
        window = values[index : index + window_days]
        if profile.onset_requires_consecutive_days and len(window) != window_days:
            continue
        if sum(window) < profile.onset_threshold_mm:
            continue
        guard_slice = pd.Series(
            values[index : min(len(values), index + profile.onset_guard_window_days)],
            dtype=float,
        )
        guard_spell = _longest_dry_spell(guard_slice, settings.seasonal_map.dry_day_threshold_mm)
        if guard_spell <= profile.onset_guard_max_dry_spell_days:
            return series.index[index]

    best_index = start_idx
    best_total = float("-inf")
    for index in range(start_idx, len(values) - window_days + 1):
        total = float(sum(values[index : index + window_days]))
        if total > best_total:
            best_total = total
            best_index = index
    return series.index[best_index]


def _detect_cessation(series: pd.Series, onset_date: pd.Timestamp, profile: SeasonalProfileConfig) -> pd.Timestamp:
    start_date = pd.Timestamp(date(series.index[0].year, profile.cessation_search_start_month, profile.cessation_search_start_day), tz="UTC")
    start_idx = max(series.index.get_indexer([start_date], method="nearest")[0], series.index.get_loc(onset_date))
    balance = float(profile.cessation_soil_water_mm)
    values = series.tolist()
    for index in range(start_idx, len(values)):
        balance = min(
            float(profile.cessation_soil_water_mm),
            balance + float(values[index]) - float(profile.cessation_et_mm_per_day),
        )
        if balance <= 0:
            return series.index[index]
    return series.index[-1]


def _build_region_items(
    district_metrics: list[dict[str, Any]],
    *,
    theme: str,
    profile: SeasonalProfileConfig,
    mode: str,
    subseason: str | None,
    generated_at: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in district_metrics:
        grouped[str(metric["region_name"])].append(metric)

    probability_items: list[dict[str, Any]] = []
    deterministic_items: list[dict[str, Any]] = []
    for region_name, metrics in sorted(grouped.items()):
        raw_metrics = {
            "onset_offset_days": _average(metrics, "onset_offset_days"),
            "cessation_offset_days": _average(metrics, "cessation_offset_days"),
            "early_dry_spell_days": _average(metrics, "early_dry_spell_days"),
            "late_dry_spell_days": _average(metrics, "late_dry_spell_days"),
            "rainfall_amount_mm": _average(metrics, "rainfall_amount_mm"),
            "rainfall_normal_mm": _average(metrics, "rainfall_normal_mm"),
            "rainy_days_count": _average(metrics, "rainy_days_count"),
            "rainy_days_normal": _average(metrics, "rainy_days_normal"),
            "season_length_days": _average(metrics, "season_length_days"),
        }
        for calendar_subseason in profile.calendar_subseasons:
            suffix = calendar_subseason.lower()
            raw_metrics[f"calendar_rainfall_amount_mm_{suffix}"] = _average(metrics, f"calendar_rainfall_amount_mm_{suffix}")
            raw_metrics[f"calendar_rainfall_normal_mm_{suffix}"] = _average(metrics, f"calendar_rainfall_normal_mm_{suffix}")
            raw_metrics[f"calendar_rainy_days_count_{suffix}"] = _average(metrics, f"calendar_rainy_days_count_{suffix}")
            raw_metrics[f"calendar_rainy_days_normal_{suffix}"] = _average(metrics, f"calendar_rainy_days_normal_{suffix}")
        coverage_note = _region_coverage_note(
            region_name=region_name,
            coverage_count=len(metrics),
            profile=profile,
            theme=theme,
        )
        probability_items.append(
            _serialize_probability_area_item(
                geography_type="region",
                geography_name=region_name,
                region_name=region_name,
                location_id=_slugify(region_name),
                coverage_count=len(metrics),
                coverage_note=coverage_note,
                raw_metrics=raw_metrics,
                theme=theme,
                profile=profile,
                mode=mode,
                subseason=subseason,
                generated_at=generated_at,
            )
        )
        deterministic_items.append(
            _serialize_deterministic_area_item(
                geography_type="region",
                geography_name=region_name,
                region_name=region_name,
                location_id=_slugify(region_name),
                coverage_count=len(metrics),
                coverage_note=coverage_note,
                raw_metrics=raw_metrics,
                theme=theme,
                profile=profile,
                mode=mode,
                subseason=subseason,
                generated_at=generated_at,
            )
        )
    return probability_items, deterministic_items


def _serialize_probability_area_item(
    *,
    geography_type: str,
    geography_name: str,
    region_name: str,
    location_id: str,
    coverage_count: int,
    coverage_note: str,
    raw_metrics: dict[str, float],
    theme: str,
    profile: SeasonalProfileConfig,
    mode: str,
    subseason: str | None,
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "location_id": location_id,
        "geography_type": geography_type,
        "geography_name": geography_name,
        "region_name": region_name,
        "coverage_count": coverage_count,
        "coverage_note": coverage_note,
        "metric": _build_probability_metric(theme, raw_metrics, profile, mode, subseason, generated_at),
    }


def _serialize_deterministic_area_item(
    *,
    geography_type: str,
    geography_name: str,
    region_name: str,
    location_id: str,
    coverage_count: int,
    coverage_note: str,
    raw_metrics: dict[str, float],
    theme: str,
    profile: SeasonalProfileConfig,
    mode: str,
    subseason: str | None,
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "location_id": location_id,
        "geography_type": geography_type,
        "geography_name": geography_name,
        "region_name": region_name,
        "coverage_count": coverage_count,
        "coverage_note": coverage_note,
        "metric": _build_deterministic_metric(theme, raw_metrics, profile, mode, subseason, generated_at),
    }


def _build_deterministic_metric(
    theme: str,
    raw_metrics: dict[str, float],
    profile: SeasonalProfileConfig,
    mode: str,
    subseason: str | None,
    generated_at: datetime,
) -> dict[str, Any]:
    if theme == "onset":
        return _build_deterministic_onset_metric(raw_metrics["onset_offset_days"], profile, generated_at)
    if theme == "cessation":
        return _build_deterministic_cessation_metric(raw_metrics["cessation_offset_days"], profile, generated_at)
    if theme == "early_dry_spell":
        return _build_deterministic_dry_spell_metric(
            theme="early_dry_spell",
            value=raw_metrics["early_dry_spell_days"],
            moderate_days=profile.early_dry_spell_moderate_days,
            high_days=profile.early_dry_spell_high_days,
        )
    if theme == "late_dry_spell":
        return _build_deterministic_dry_spell_metric(
            theme="late_dry_spell",
            value=raw_metrics["late_dry_spell_days"],
            moderate_days=profile.late_dry_spell_moderate_days,
            high_days=profile.late_dry_spell_high_days,
        )
    if theme == "rainfall_amount":
        return _build_deterministic_rainfall_amount_metric(
            raw_metrics["rainfall_amount_mm"]
            if mode == "seasonal"
            else raw_metrics[f"calendar_rainfall_amount_mm_{str(subseason).lower()}"],
            raw_metrics["rainfall_normal_mm"]
            if mode == "seasonal"
            else raw_metrics[f"calendar_rainfall_normal_mm_{str(subseason).lower()}"],
            profile.rainfall_band_pct,
            profile,
            mode=mode,
            subseason=subseason,
        )
    return _build_deterministic_rainy_days_metric(
        raw_metrics["rainy_days_count"]
        if mode == "seasonal"
        else raw_metrics[f"calendar_rainy_days_count_{str(subseason).lower()}"],
        raw_metrics["rainy_days_normal"]
        if mode == "seasonal"
        else raw_metrics[f"calendar_rainy_days_normal_{str(subseason).lower()}"],
        profile.rainy_days_band,
        profile,
        mode=mode,
        subseason=subseason,
    )


def _build_probability_metric(
    theme: str,
    raw_metrics: dict[str, float],
    profile: SeasonalProfileConfig,
    mode: str,
    subseason: str | None,
    generated_at: datetime,
) -> dict[str, Any]:
    legend = _build_probability_legend(theme, mode, profile, subseason)
    if theme == "onset":
        band = max(float(profile.onset_normal_band_days), 1.0)
        categories = _probability_categories_from_scores(
            legend,
            _soft_scores(
                raw_metrics["onset_offset_days"],
                centers=(-band * 1.6, 0.0, band * 1.6),
                scale=max(band * 0.9, 1.0),
            ),
        )
    elif theme == "cessation":
        band = max(float(profile.cessation_normal_band_days), 1.0)
        categories = _probability_categories_from_scores(
            legend,
            _soft_scores(
                raw_metrics["cessation_offset_days"],
                centers=(-band * 1.6, 0.0, band * 1.6),
                scale=max(band * 0.9, 1.0),
            ),
        )
    elif theme == "early_dry_spell":
        categories = _probability_categories_from_scores(
            legend,
            _soft_scores(
                raw_metrics["early_dry_spell_days"],
                centers=(
                    max(profile.early_dry_spell_moderate_days - 2.0, 1.0),
                    (profile.early_dry_spell_moderate_days + profile.early_dry_spell_high_days) / 2.0,
                    profile.early_dry_spell_high_days + 2.5,
                ),
                scale=max((profile.early_dry_spell_high_days - profile.early_dry_spell_moderate_days) * 0.9, 1.0),
            ),
        )
    elif theme == "late_dry_spell":
        categories = _probability_categories_from_scores(
            legend,
            _soft_scores(
                raw_metrics["late_dry_spell_days"],
                centers=(
                    max(profile.late_dry_spell_moderate_days - 2.0, 1.0),
                    (profile.late_dry_spell_moderate_days + profile.late_dry_spell_high_days) / 2.0,
                    profile.late_dry_spell_high_days + 2.5,
                ),
                scale=max((profile.late_dry_spell_high_days - profile.late_dry_spell_moderate_days) * 0.9, 1.0),
            ),
        )
    elif theme == "rainfall_amount":
        normal_value = (
            raw_metrics["rainfall_normal_mm"]
            if mode == "seasonal"
            else raw_metrics[f"calendar_rainfall_normal_mm_{str(subseason).lower()}"]
        )
        value = (
            raw_metrics["rainfall_amount_mm"]
            if mode == "seasonal"
            else raw_metrics[f"calendar_rainfall_amount_mm_{str(subseason).lower()}"]
        )
        deviation_pct = ((value - normal_value) / max(normal_value, 1.0)) * 100.0
        band = max(float(profile.rainfall_band_pct), 1.0)
        categories = _probability_categories_from_scores(
            legend,
            _soft_scores(
                deviation_pct,
                centers=(-band * 1.5, 0.0, band * 1.5),
                scale=max(band * 0.85, 1.0),
            ),
        )
    else:
        normal_value = (
            raw_metrics["rainy_days_normal"]
            if mode == "seasonal"
            else raw_metrics[f"calendar_rainy_days_normal_{str(subseason).lower()}"]
        )
        value = (
            raw_metrics["rainy_days_count"]
            if mode == "seasonal"
            else raw_metrics[f"calendar_rainy_days_count_{str(subseason).lower()}"]
        )
        band = max(float(profile.rainy_days_band), 1.0)
        categories = _probability_categories_from_scores(
            legend,
            _soft_scores(
                value - normal_value,
                centers=(-band * 1.5, 0.0, band * 1.5),
                scale=max(band * 0.85, 1.0),
            ),
        )

    dominant = max(categories, key=lambda item: float(item["percentage"]))
    deterministic_metric = _build_deterministic_metric(theme, raw_metrics, profile, mode, subseason, generated_at)
    return {
        "theme": theme,
        "theme_label": THEME_LABELS[theme],
        "category_code": dominant["category_code"],
        "category_label": dominant["label"],
        "dominant_category_code": dominant["category_code"],
        "dominant_category_label": dominant["label"],
        "dominant_percentage": dominant["percentage"],
        "display_value": f"{int(round(float(dominant['percentage'])))}%",
        "unit": "percent",
        "criteria_note": deterministic_metric["criteria_note"],
        "interpretation": (
            f"{profile.label} probability favours {dominant['label'].lower()} at {float(dominant['percentage']):.1f}%."
        ),
        "color": dominant["color"],
        "category_probabilities": categories,
    }


def _probability_categories_from_scores(
    legend: list[dict[str, str]],
    scores: tuple[float, float, float],
) -> list[dict[str, Any]]:
    percentages = _normalize_percentages(list(scores))
    return [
        {
            "category_code": item["category_code"],
            "label": item["label"],
            "hint": item["hint"],
            "color": item["color"],
            "percentage": percentages[index],
        }
        for index, item in enumerate(legend)
    ]


def _soft_scores(value: float, *, centers: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    resolved_scale = max(scale, 0.5)
    scores = tuple(math.exp(-((value - center) ** 2) / (2.0 * (resolved_scale**2))) for center in centers)
    return scores if any(score > 0 for score in scores) else (1.0, 1.0, 1.0)


def _normalize_percentages(scores: list[float]) -> list[float]:
    total = sum(scores)
    if total <= 0:
        return [33.4, 33.3, 33.3]
    percentages = [round((score / total) * 100.0, 1) for score in scores]
    delta = round(100.0 - sum(percentages), 1)
    percentages[0] = round(percentages[0] + delta, 1)
    return percentages


def _build_deterministic_onset_metric(
    offset_days: float,
    profile: SeasonalProfileConfig,
    generated_at: datetime,
) -> dict[str, Any]:
    classified = _classify_onset(offset_days, profile, generated_at)
    return {
        "theme": "onset",
        "theme_label": THEME_LABELS["onset"],
        "value": round(offset_days, 1),
        "display_value": classified["display_value"],
        "unit": classified["unit"],
        "criteria_note": classified["criteria_note"],
        "interpretation": f"{profile.label} detected onset date resolves to {classified['display_value']}.",
        "legend_label": _deterministic_date_legend_label(
            date(generated_at.year, profile.onset_reference_month, profile.onset_reference_day),
            profile.onset_normal_band_days,
            offset_days,
        ),
        "color": classified["color"],
    }


def _build_deterministic_cessation_metric(
    offset_days: float,
    profile: SeasonalProfileConfig,
    generated_at: datetime,
) -> dict[str, Any]:
    classified = _classify_cessation(offset_days, profile, generated_at)
    return {
        "theme": "cessation",
        "theme_label": THEME_LABELS["cessation"],
        "value": round(offset_days, 1),
        "display_value": classified["display_value"],
        "unit": classified["unit"],
        "criteria_note": classified["criteria_note"],
        "interpretation": f"{profile.label} detected cessation date resolves to {classified['display_value']}.",
        "legend_label": _deterministic_date_legend_label(
            date(generated_at.year, profile.cessation_reference_month, profile.cessation_reference_day),
            profile.cessation_normal_band_days,
            offset_days,
        ),
        "color": classified["color"],
    }


def _build_deterministic_dry_spell_metric(*, theme: str, value: float, moderate_days: int, high_days: int) -> dict[str, Any]:
    classified = _classify_dry_spell(
        theme=theme,
        value=value,
        moderate_days=moderate_days,
        high_days=high_days,
    )
    whole_days = int(round(value))
    return {
        "theme": theme,
        "theme_label": THEME_LABELS[theme],
        "value": round(value, 1),
        "display_value": f"{whole_days} day(s)",
        "unit": "days",
        "criteria_note": classified["criteria_note"],
        "interpretation": f"{THEME_LABELS[theme]} resolves to {whole_days} day(s) for this geography.",
        "legend_label": classified["category_label"],
        "color": classified["color"],
    }


def _build_deterministic_rainfall_amount_metric(
    value: float,
    normal_value: float,
    band_pct: float,
    profile: SeasonalProfileConfig,
    *,
    mode: str,
    subseason: str | None,
) -> dict[str, Any]:
    classified = _classify_rainfall_amount(value, normal_value, band_pct, profile, mode=mode, subseason=subseason)
    return {
        "theme": "rainfall_amount",
        "theme_label": THEME_LABELS["rainfall_amount"],
        "value": round(value, 1),
        "display_value": f"{value:.1f} mm",
        "unit": "mm",
        "criteria_note": classified["criteria_note"],
        "interpretation": (
            f"Detected rainfall total is {value:.1f} mm against a normal of {normal_value:.1f} mm."
            if mode == "seasonal"
            else f"Detected {subseason} rainfall total is {value:.1f} mm against a normal of {normal_value:.1f} mm."
        ),
        "legend_label": classified["category_label"],
        "color": classified["color"],
    }


def _build_deterministic_rainy_days_metric(
    value: float,
    normal_value: float,
    band: float,
    profile: SeasonalProfileConfig,
    *,
    mode: str,
    subseason: str | None,
) -> dict[str, Any]:
    classified = _classify_rainy_days(value, normal_value, band, profile, mode=mode, subseason=subseason)
    whole_days = int(round(value))
    return {
        "theme": "rainy_days",
        "theme_label": THEME_LABELS["rainy_days"],
        "value": round(value, 1),
        "display_value": f"{whole_days} day(s)",
        "unit": "days",
        "criteria_note": classified["criteria_note"],
        "interpretation": (
            f"Detected rainy-day count is {whole_days} day(s) against a normal of {normal_value:.1f} day(s)."
            if mode == "seasonal"
            else f"Detected {subseason} rainy-day count is {whole_days} day(s) against a normal of {normal_value:.1f} day(s)."
        ),
        "legend_label": classified["category_label"],
        "color": classified["color"],
    }


def _deterministic_date_legend_label(reference_date: date, band_days: int, offset_days: float) -> str:
    lower = reference_date - timedelta(days=band_days)
    upper = reference_date + timedelta(days=band_days)
    if offset_days < -band_days:
        return f"Before {lower.strftime('%d %b')}"
    if offset_days > band_days:
        return f"After {upper.strftime('%d %b')}"
    return f"{lower.strftime('%d %b')} to {upper.strftime('%d %b')}"


def _classify_onset(offset_days: float, profile: SeasonalProfileConfig, generated_at: datetime) -> dict[str, Any]:
    reference_date = date(generated_at.year, profile.onset_reference_month, profile.onset_reference_day)
    resolved_date = reference_date + timedelta(days=int(round(offset_days)))
    if offset_days < -profile.onset_normal_band_days:
        category_code = "early"
        category_label = "Early"
        interpretation = f"{profile.label} onset is arriving earlier than the Ghana WMO timing band."
    elif offset_days > profile.onset_normal_band_days:
        category_code = "late"
        category_label = "Late"
        interpretation = f"{profile.label} onset is lagging the Ghana WMO timing band."
    else:
        category_code = "normal"
        category_label = "Near-Normal"
        interpretation = f"{profile.label} onset stays within the Ghana WMO timing band."
    onset_phrase = (
        "20 mm in 3 consecutive days"
        if profile.onset_requires_consecutive_days
        else f"at least {profile.onset_threshold_mm:.0f} mm in up to {profile.onset_window_days} days"
    )
    return {
        "theme": "onset",
        "theme_label": THEME_LABELS["onset"],
        "category_code": category_code,
        "category_label": category_label,
        "numeric_value": round(offset_days, 1),
        "display_value": resolved_date.strftime("%d %b"),
        "unit": "days_from_reference",
        "criteria_note": (
            f"Detected from {date(generated_at.year, profile.onset_search_start_month, profile.onset_search_start_day).strftime('%d %b')} "
            f"with end-search on 15 May using {onset_phrase}, and no dry spell longer than "
            f"{profile.onset_guard_max_dry_spell_days} days in the next {profile.onset_guard_window_days} days."
        ),
        "interpretation": interpretation,
        "color": _legend_color("onset", category_code),
    }


def _classify_cessation(
    offset_days: float,
    profile: SeasonalProfileConfig,
    generated_at: datetime,
) -> dict[str, Any]:
    reference_date = date(generated_at.year, profile.cessation_reference_month, profile.cessation_reference_day)
    resolved_date = reference_date + timedelta(days=int(round(offset_days)))
    if offset_days < -profile.cessation_normal_band_days:
        category_code = "early"
        category_label = "Early"
        interpretation = f"{profile.label} cessation is happening earlier than the Ghana WMO timing band."
    elif offset_days > profile.cessation_normal_band_days:
        category_code = "late"
        category_label = "Late"
        interpretation = f"{profile.label} cessation extends later than the Ghana WMO timing band."
    else:
        category_code = "normal"
        category_label = "Normal"
        interpretation = f"{profile.label} cessation stays within the Ghana WMO timing band."
    return {
        "theme": "cessation",
        "theme_label": THEME_LABELS["cessation"],
        "category_code": category_code,
        "category_label": category_label,
        "numeric_value": round(offset_days, 1),
        "display_value": resolved_date.strftime("%d %b"),
        "unit": "days_from_reference",
        "criteria_note": (
            f"Detected from {date(generated_at.year, profile.cessation_search_start_month, profile.cessation_search_start_day).strftime('%d %b')} "
            f"using soil water balance depletion from {profile.cessation_soil_water_mm:.0f} mm with "
            f"{profile.cessation_et_mm_per_day:.0f} mm/day evapotranspiration."
        ),
        "interpretation": interpretation,
        "color": _legend_color("cessation", category_code),
    }


def _classify_dry_spell(*, theme: str, value: float, moderate_days: int, high_days: int) -> dict[str, Any]:
    whole_days = int(value + 0.5)
    if value >= high_days:
        category_code = "high"
        category_label = "Long"
        interpretation = f"{THEME_LABELS[theme]} pressure is elevated under the selected Ghana regime."
    elif value >= moderate_days:
        category_code = "moderate"
        category_label = "Near-Normal"
        interpretation = f"{THEME_LABELS[theme]} pressure is building and should be monitored."
    else:
        category_code = "low"
        category_label = "Short"
        interpretation = f"{THEME_LABELS[theme]} pressure remains limited in this regime run."
    window_note = "Longest dry run from onset to day 50." if theme == "early_dry_spell" else "Longest dry run from day 51 to cessation."
    return {
        "theme": theme,
        "theme_label": THEME_LABELS[theme],
        "category_code": category_code,
        "category_label": category_label,
        "numeric_value": whole_days,
        "display_value": f"{whole_days} day(s)",
        "unit": "days",
        "criteria_note": (
            f"{window_note} Onset criterion uses cumulative rainfall >= 20 mm, "
            f"dry-day threshold < 1 mm, end-search 15 May, and nbjour: 50. "
            f"Short below {moderate_days} days; Near-Normal from {moderate_days} to under {high_days}; "
            f"Long from {high_days} days upward."
        ),
        "interpretation": interpretation,
        "color": _legend_color(theme, category_code),
    }


def _classify_rainfall_amount(
    value: float,
    normal_value: float,
    band_pct: float,
    profile: SeasonalProfileConfig,
    *,
    mode: str,
    subseason: str | None,
) -> dict[str, Any]:
    lower = normal_value * (1 - band_pct / 100)
    upper = normal_value * (1 + band_pct / 100)
    if value < lower:
        category_code = "below_normal"
        category_label = "BELOW-AVERAGE"
        interpretation = (
            f"{profile.label} rainfall is below the expected onset-to-cessation total."
            if mode == "seasonal"
            else f"{profile.label} rainfall is below the expected {subseason} reporting window total."
        )
    elif value > upper:
        category_code = "above_normal"
        category_label = "ABOVE-AVERAGE"
        interpretation = (
            f"{profile.label} rainfall is above the expected onset-to-cessation total."
            if mode == "seasonal"
            else f"{profile.label} rainfall is above the expected {subseason} reporting window total."
        )
    else:
        category_code = "near_normal"
        category_label = "NEAR-AVERAGE"
        interpretation = (
            f"{profile.label} rainfall stays within the expected onset-to-cessation band."
            if mode == "seasonal"
            else f"{profile.label} rainfall stays within the expected {subseason} reporting window band."
        )
    criteria_note = (
        f"Seasonal rainfall is summed from detected onset to cessation. Categories compare against a "
        f"{profile.label.lower()} normal of {normal_value:.1f} mm with a +/- {band_pct:.0f}% band."
        if mode == "seasonal"
        else f"Calendar rainfall is summed only within {subseason}. Categories compare against a "
        f"{profile.label.lower()} {subseason} normal of {normal_value:.1f} mm with a +/- {band_pct:.0f}% band."
    )
    return {
        "theme": "rainfall_amount",
        "theme_label": THEME_LABELS["rainfall_amount"],
        "category_code": category_code,
        "category_label": category_label,
        "numeric_value": round(value, 1),
        "display_value": f"{value:.1f} mm",
        "unit": "mm",
        "criteria_note": criteria_note,
        "interpretation": interpretation,
        "color": _legend_color("rainfall_amount", category_code),
    }


def _classify_rainy_days(
    value: float,
    normal_value: float,
    band: float,
    profile: SeasonalProfileConfig,
    *,
    mode: str,
    subseason: str | None,
) -> dict[str, Any]:
    lower = normal_value - band
    upper = normal_value + band
    if value < lower:
        category_code = "fewer"
        category_label = "BELOW-AVERAGE"
        interpretation = (
            f"{profile.label} rainy days are below the expected onset-to-cessation count."
            if mode == "seasonal"
            else f"{profile.label} rainy days are below the expected {subseason} reporting window count."
        )
    elif value > upper:
        category_code = "more"
        category_label = "ABOVE-AVERAGE"
        interpretation = (
            f"{profile.label} rainy days are above the expected onset-to-cessation count."
            if mode == "seasonal"
            else f"{profile.label} rainy days are above the expected {subseason} reporting window count."
        )
    else:
        category_code = "normal"
        category_label = "NEAR-AVERAGE"
        interpretation = (
            f"{profile.label} rainy days stay within the expected onset-to-cessation band."
            if mode == "seasonal"
            else f"{profile.label} rainy days stay within the expected {subseason} reporting window band."
        )
    criteria_note = (
        f"Rainy days are counted between detected onset and cessation. Categories compare against a "
        f"{profile.label.lower()} normal of {normal_value:.1f} day(s) with a +/- {band:.1f} band."
        if mode == "seasonal"
        else f"Rainy days are counted only within {subseason}. Categories compare against a "
        f"{profile.label.lower()} {subseason} normal of {normal_value:.1f} day(s) with a +/- {band:.1f} band."
    )
    return {
        "theme": "rainy_days",
        "theme_label": THEME_LABELS["rainy_days"],
        "category_code": category_code,
        "category_label": category_label,
        "numeric_value": round(value, 1),
        "display_value": f"{int(round(value))} day(s)",
        "unit": "days",
        "criteria_note": criteria_note,
        "interpretation": interpretation,
        "color": _legend_color("rainy_days", category_code),
    }


def _probability_reverse_scale(theme: str) -> bool:
    return theme in {"onset", "early_dry_spell", "late_dry_spell"}


def _legend_entry(
    *,
    category_code: str,
    label: str,
    hint: str,
    color: str,
    display_order: int,
    reverse_probability_scale: bool,
) -> dict[str, Any]:
    return {
        "category_code": category_code,
        "label": label,
        "hint": hint,
        "color": color,
        "family_label": label,
        "display_order": display_order,
        "reverse_probability_scale": reverse_probability_scale,
    }


def _build_probability_legend(
    theme: str,
    mode: str,
    profile: SeasonalProfileConfig,
    subseason: str | None = None,
) -> list[dict[str, Any]]:
    reverse_probability_scale = _probability_reverse_scale(theme)
    onset_phrase = (
        "20 mm in 3 consecutive days"
        if profile.onset_requires_consecutive_days
        else f"20 mm in up to {profile.onset_window_days} days"
    )
    if theme == "onset":
        return [
            _legend_entry(
                category_code="early",
                label="Early",
                hint=(
                    f"Probability of onset arriving earlier than the {profile.label.lower()} WMO band after "
                    f"{date(2026, profile.onset_search_start_month, profile.onset_search_start_day).strftime('%d %b')}."
                ),
                color=WASS2S_PROBABILITY_FAMILY_COLORS["teal"],
                display_order=0,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="normal",
                label="Near-Normal",
                hint=f"Probability of onset staying within the Ghana WMO timing band using {onset_phrase}.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["neutral"],
                display_order=1,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="late",
                label="Late",
                hint="Probability of onset arriving later than the Ghana WMO timing band.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["brown"],
                display_order=2,
                reverse_probability_scale=reverse_probability_scale,
            ),
        ]
    if theme == "cessation":
        return [
            _legend_entry(
                category_code="early",
                label="Early",
                hint="Probability of soil water balance depleting earlier than the regime timing band.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["brown"],
                display_order=0,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="normal",
                label="Near-Normal",
                hint="Probability of soil water balance depleting within the regime timing band.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["neutral"],
                display_order=1,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="late",
                label="Late",
                hint="Probability of soil water balance depleting later than the regime timing band.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["teal"],
                display_order=2,
                reverse_probability_scale=reverse_probability_scale,
            ),
        ]
    if theme == "early_dry_spell":
        return [
            _legend_entry(
                category_code="low",
                label="Short",
                hint="Probability that the longest dry run from onset to day 50 remains short.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["teal"],
                display_order=0,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="moderate",
                label="Near-Normal",
                hint="Probability that the longest dry run from onset to day 50 stays near the normal range.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["neutral"],
                display_order=1,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="high",
                label="Long",
                hint="Probability that the longest dry run from onset to day 50 becomes long.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["brown"],
                display_order=2,
                reverse_probability_scale=reverse_probability_scale,
            ),
        ]
    if theme == "late_dry_spell":
        return [
            _legend_entry(
                category_code="low",
                label="Short",
                hint="Probability that the longest dry run from day 51 to cessation remains short.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["teal"],
                display_order=0,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="moderate",
                label="Near-Normal",
                hint="Probability that the longest dry run from day 51 to cessation stays near the normal range.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["neutral"],
                display_order=1,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="high",
                label="Long",
                hint="Probability that the longest dry run from day 51 to cessation becomes long.",
                color=WASS2S_PROBABILITY_FAMILY_COLORS["brown"],
                display_order=2,
                reverse_probability_scale=reverse_probability_scale,
            ),
        ]
    if theme == "rainfall_amount":
        return [
            _legend_entry(
                category_code="below_normal",
                label="BELOW-AVERAGE",
                hint=(
                    "Probability that seasonal rainfall falls below the regime normal."
                    if mode == "seasonal"
                    else f"Probability that {subseason} rainfall falls below the regime calendar normal."
                ),
                color=WASS2S_PROBABILITY_FAMILY_COLORS["brown"],
                display_order=0,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="near_normal",
                label="NEAR-AVERAGE",
                hint=(
                    "Probability that seasonal rainfall stays within the regime normal band."
                    if mode == "seasonal"
                    else f"Probability that {subseason} rainfall stays within the regime calendar band."
                ),
                color=WASS2S_PROBABILITY_FAMILY_COLORS["neutral"],
                display_order=1,
                reverse_probability_scale=reverse_probability_scale,
            ),
            _legend_entry(
                category_code="above_normal",
                label="ABOVE-AVERAGE",
                hint=(
                    "Probability that seasonal rainfall rises above the regime normal."
                    if mode == "seasonal"
                    else f"Probability that {subseason} rainfall rises above the regime calendar normal."
                ),
                color=WASS2S_PROBABILITY_FAMILY_COLORS["teal"],
                display_order=2,
                reverse_probability_scale=reverse_probability_scale,
            ),
        ]
    return [
        _legend_entry(
            category_code="fewer",
            label="BELOW-AVERAGE",
            hint=(
                "Probability that the rainy-day count stays below the regime normal."
                if mode == "seasonal"
                else f"Probability that the rainy-day count in {subseason} stays below the regime calendar normal."
            ),
            color=WASS2S_PROBABILITY_FAMILY_COLORS["brown"],
            display_order=0,
            reverse_probability_scale=reverse_probability_scale,
        ),
        _legend_entry(
            category_code="normal",
            label="NEAR-AVERAGE",
            hint=(
                "Probability that the rainy-day count stays within the regime normal band."
                if mode == "seasonal"
                else f"Probability that the rainy-day count in {subseason} stays within the regime calendar band."
            ),
            color=WASS2S_PROBABILITY_FAMILY_COLORS["neutral"],
            display_order=1,
            reverse_probability_scale=reverse_probability_scale,
        ),
        _legend_entry(
            category_code="more",
            label="ABOVE-AVERAGE",
            hint=(
                "Probability that the rainy-day count rises above the regime normal."
                if mode == "seasonal"
                else f"Probability that the rainy-day count in {subseason} rises above the regime calendar normal."
            ),
            color=WASS2S_PROBABILITY_FAMILY_COLORS["teal"],
            display_order=2,
            reverse_probability_scale=reverse_probability_scale,
        ),
    ]


def _build_deterministic_legend(
    theme: str,
    mode: str,
    profile: SeasonalProfileConfig,
    subseason: str | None,
    generated_at: datetime,
) -> list[dict[str, str]]:
    if theme == "onset":
        reference_date = date(generated_at.year, profile.onset_reference_month, profile.onset_reference_day)
        lower = (reference_date - timedelta(days=profile.onset_normal_band_days)).strftime("%d %b")
        upper = (reference_date + timedelta(days=profile.onset_normal_band_days)).strftime("%d %b")
        return [
            {"category_code": "early", "label": f"Before {lower}", "hint": "Detected onset date before the timing band.", "color": "#1f8a5b"},
            {"category_code": "normal", "label": f"{lower} to {upper}", "hint": "Detected onset date within the timing band.", "color": "#c9962b"},
            {"category_code": "late", "label": f"After {upper}", "hint": "Detected onset date after the timing band.", "color": "#c65a46"},
        ]
    if theme == "cessation":
        reference_date = date(generated_at.year, profile.cessation_reference_month, profile.cessation_reference_day)
        lower = (reference_date - timedelta(days=profile.cessation_normal_band_days)).strftime("%d %b")
        upper = (reference_date + timedelta(days=profile.cessation_normal_band_days)).strftime("%d %b")
        return [
            {"category_code": "early", "label": f"Before {lower}", "hint": "Detected cessation date before the timing band.", "color": "#c65a46"},
            {"category_code": "normal", "label": f"{lower} to {upper}", "hint": "Detected cessation date within the timing band.", "color": "#c9962b"},
            {"category_code": "late", "label": f"After {upper}", "hint": "Detected cessation date after the timing band.", "color": "#1f8a5b"},
        ]
    if theme == "early_dry_spell":
        return [
            {"category_code": "low", "label": f"0 to {profile.early_dry_spell_moderate_days - 1} day(s)", "hint": "Detected early-season dry spell stays short.", "color": "#2f8f6e"},
            {"category_code": "moderate", "label": f"{profile.early_dry_spell_moderate_days} to {profile.early_dry_spell_high_days - 1} day(s)", "hint": "Detected early-season dry spell stays near the normal range.", "color": "#c98b37"},
            {"category_code": "high", "label": f"{profile.early_dry_spell_high_days}+ day(s)", "hint": "Detected early-season dry spell becomes long.", "color": "#c45143"},
        ]
    if theme == "late_dry_spell":
        return [
            {"category_code": "low", "label": f"0 to {profile.late_dry_spell_moderate_days - 1} day(s)", "hint": "Detected late-season dry spell stays short.", "color": "#2f8f6e"},
            {"category_code": "moderate", "label": f"{profile.late_dry_spell_moderate_days} to {profile.late_dry_spell_high_days - 1} day(s)", "hint": "Detected late-season dry spell stays near the normal range.", "color": "#c98b37"},
            {"category_code": "high", "label": f"{profile.late_dry_spell_high_days}+ day(s)", "hint": "Detected late-season dry spell becomes long.", "color": "#c45143"},
        ]
    if theme == "rainfall_amount":
        normal_value = float(profile.rainfall_normal_mm)
        lower = normal_value * (1 - profile.rainfall_band_pct / 100.0)
        upper = normal_value * (1 + profile.rainfall_band_pct / 100.0)
        label_prefix = "Seasonal" if mode == "seasonal" else str(subseason)
        return [
            {"category_code": "below_normal", "label": f"< {lower:.1f} mm", "hint": f"{label_prefix} rainfall total below the normal band.", "color": "#c55a45"},
            {"category_code": "near_normal", "label": f"{lower:.1f} to {upper:.1f} mm", "hint": f"{label_prefix} rainfall total within the normal band.", "color": "#c9962b"},
            {"category_code": "above_normal", "label": f"> {upper:.1f} mm", "hint": f"{label_prefix} rainfall total above the normal band.", "color": "#1f8a5b"},
        ]
    lower_days = profile.rainy_days_normal - profile.rainy_days_band
    upper_days = profile.rainy_days_normal + profile.rainy_days_band
    label_prefix = "Seasonal" if mode == "seasonal" else str(subseason)
    return [
        {"category_code": "fewer", "label": f"< {lower_days:.1f} day(s)", "hint": f"{label_prefix} rainy-day count below the normal band.", "color": "#c55a45"},
        {"category_code": "normal", "label": f"{lower_days:.1f} to {upper_days:.1f} day(s)", "hint": f"{label_prefix} rainy-day count within the normal band.", "color": "#c9962b"},
        {"category_code": "more", "label": f"> {upper_days:.1f} day(s)", "hint": f"{label_prefix} rainy-day count above the normal band.", "color": "#1f8a5b"},
    ]


def _legend_color(theme: str, category_code: str) -> str:
    for item in _build_probability_legend(theme, "seasonal", _fallback_profile_for_legend()):
        if item["category_code"] == category_code:
            return item["color"]
    return "#75857b"


def _build_legend(
    theme: str,
    mode: str,
    profile: SeasonalProfileConfig,
    subseason: str | None = None,
) -> list[dict[str, Any]]:
    return _build_probability_legend(theme, mode, profile, subseason)


@lru_cache(maxsize=1)
def _fallback_profile_for_legend() -> SeasonalProfileConfig:
    return SeasonalProfileConfig(
        label="Fallback Profile",
        native_zone="south",
        onset_search_start_month=2,
        onset_search_start_day=1,
        onset_reference_month=2,
        onset_reference_day=12,
        cessation_search_start_month=7,
        cessation_search_start_day=1,
        cessation_reference_month=7,
        cessation_reference_day=18,
        rainfall_normal_mm=500.0,
        rainy_days_normal=40.0,
    )


def _longest_dry_spell(series: pd.Series, dry_day_threshold_mm: float) -> int:
    longest = 0
    current = 0
    for value in series.fillna(0.0).astype(float):
        if value < dry_day_threshold_mm:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _rainfall_year(frame: pd.DataFrame) -> int:
    if "time" in frame.columns and not frame.empty:
        return int(pd.to_datetime(frame["time"], utc=True).dt.year.max())
    return datetime.now(UTC).year


def _theme_uses_regime_footprint(theme: str) -> bool:
    return theme in REGIME_BOUND_THEMES


def _district_regime_zone(latitude: float, northern_latitude_threshold: float) -> str:
    return "north" if float(latitude) >= northern_latitude_threshold else "south"


def _district_matches_profile_footprint(district: dict[str, Any], profile: SeasonalProfileConfig) -> bool:
    return str(district.get("regime_zone")) == profile.native_zone


def _district_coverage_note(
    *,
    district: dict[str, Any],
    profile: SeasonalProfileConfig,
    theme: str,
) -> str:
    latitude = float(district["latitude"])
    regime_zone = str(district["regime_zone"])
    if _theme_uses_regime_footprint(theme):
        return (
            f"District classification is published only inside the {profile.label} agro-ecological footprint. "
            f"This district falls in the {regime_zone} zone at a representative latitude of {latitude:.2f} N."
        )
    return (
        f"District classification remains nationwide. The {profile.label} selection changes the seasonal criteria, "
        f"normals, and labels, while this district remains visible at {latitude:.2f} N in the {regime_zone} zone."
    )


def _region_coverage_note(
    *,
    region_name: str,
    coverage_count: int,
    profile: SeasonalProfileConfig,
    theme: str,
) -> str:
    if _theme_uses_regime_footprint(theme):
        return (
            f"Regional classification for {region_name} aggregates {coverage_count} in-footprint district outputs "
            f"that match the {profile.label} agro-ecological footprint."
        )
    return (
        f"Regional classification for {region_name} aggregates all {coverage_count} district outputs returned for "
        f"the region. The {profile.label} selection only changes the seasonal criteria, normals, and labels."
    )


def _profile_alignment_factor(regime_zone: str, native_zone: str) -> float:
    return 1.05 if regime_zone == native_zone else 0.92


def _slugify(value: str) -> str:
    normalized = "".join(char if char.isalnum() else "-" for char in value.lower())
    return "-".join(piece for piece in normalized.split("-") if piece)


def _representative_point(geometry: dict[str, Any]) -> tuple[float, float]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    polygons: list[list[list[float]]] = []
    if geometry_type == "Polygon":
        polygons = [coordinates[0]] if coordinates else []
    elif geometry_type == "MultiPolygon":
        polygons = [polygon[0] for polygon in coordinates if polygon]

    if not polygons:
        return 0.0, 0.0

    ring = max(polygons, key=lambda item: abs(_polygon_area(item)))
    centroid = _polygon_centroid(ring)
    if centroid is not None:
        return round(centroid[1], 4), round(centroid[0], 4)
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return round((min(ys) + max(ys)) / 2, 4), round((min(xs) + max(xs)) / 2, 4)


def _polygon_area(ring: list[list[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    total = 0.0
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        total += x1 * y2 - x2 * y1
    return total / 2


def _polygon_centroid(ring: list[list[float]]) -> tuple[float, float] | None:
    area = _polygon_area(ring)
    if area == 0:
        return None
    cx = 0.0
    cy = 0.0
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        factor = x1 * y2 - x2 * y1
        cx += (x1 + x2) * factor
        cy += (y1 + y2) * factor
    return cx / (6 * area), cy / (6 * area)


def _average(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items]
    return round(sum(values) / len(values), 2) if values else 0.0


def _forecast_source_label(source_id: str) -> str:
    return "Configured Forecast Feed" if source_id == "configured" else source_id.upper()


def _artifact_scope_path(
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
    mode: str,
    subseason: str | None,
) -> Path:
    scope = settings.seasonal_map.artifact_dir / source_id / season_profile / theme / mode
    if subseason:
        return scope / subseason.lower()
    return scope


def _legacy_artifact_scope_path(
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
) -> Path:
    return settings.seasonal_map.artifact_dir / source_id / season_profile / theme


def _active_pointer_path(
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
    mode: str,
    subseason: str | None,
) -> Path:
    suffix = f"{source_id}_{season_profile}_{theme}_{mode}"
    if subseason:
        suffix = f"{suffix}_{subseason.lower()}"
    return settings.seasonal_map.artifact_dir / f"active_{suffix}.json"


def _legacy_active_pointer_path(
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
) -> Path:
    return settings.seasonal_map.artifact_dir / f"active_{source_id}_{season_profile}_{theme}.json"


def _resolve_active_pointer_path(
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
    mode: str,
    subseason: str | None,
) -> Path:
    canonical = _active_pointer_path(settings, source_id, season_profile, theme, mode, subseason)
    if canonical.exists():
        return canonical
    canonical_pattern = f"active_*_{season_profile}_{theme}_{mode}{f'_{subseason.lower()}' if subseason else ''}.json"
    canonical_match = _unique_glob_match(settings.seasonal_map.artifact_dir, canonical_pattern)
    if canonical_match is not None:
        return canonical_match
    if mode == "seasonal" and subseason is None:
        legacy = _legacy_active_pointer_path(settings, source_id, season_profile, theme)
        if legacy.exists():
            return legacy
        legacy_match = _unique_glob_match(settings.seasonal_map.artifact_dir, f"active_*_{season_profile}_{theme}.json")
        if legacy_match is not None:
            return legacy_match
    return canonical


def _candidate_scope_paths(
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
    mode: str,
    subseason: str | None,
) -> list[Path]:
    candidates: list[Path] = []
    canonical = _artifact_scope_path(settings, source_id, season_profile, theme, mode, subseason)
    candidates.append(canonical)
    canonical_pattern = f"*/{season_profile}/{theme}/{mode}"
    if subseason:
        canonical_pattern = f"{canonical_pattern}/{subseason.lower()}"
    canonical_match = _unique_glob_match(settings.seasonal_map.artifact_dir, canonical_pattern)
    if canonical_match is not None:
        candidates.append(canonical_match)
    if mode == "seasonal" and subseason is None:
        legacy = _legacy_artifact_scope_path(settings, source_id, season_profile, theme)
        candidates.append(legacy)
        legacy_match = _unique_glob_match(settings.seasonal_map.artifact_dir, f"*/{season_profile}/{theme}")
        if legacy_match is not None:
            candidates.append(legacy_match)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _unique_glob_match(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    return None


def _discover_latest_product_path(
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
    mode: str,
    subseason: str | None,
) -> Path | None:
    candidates: list[Path] = []
    for scope in _candidate_scope_paths(settings, source_id, season_profile, theme, mode, subseason):
        runs_dir = scope / "runs"
        if not runs_dir.exists():
            continue
        candidates.extend(sorted(runs_dir.glob("*/product.json")))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.parent.name)


def _resolve_product_path(
    pointer: dict[str, Any],
    *,
    settings: Settings,
    source_id: str,
    season_profile: str,
    theme: str,
    mode: str,
    subseason: str | None,
) -> Path:
    product_path = pointer.get("product_path")
    if product_path:
        resolved = Path(str(product_path)).resolve()
        if resolved.exists():
            return resolved
    manifest_path = pointer.get("manifest_path")
    if manifest_path:
        manifest_product = Path(str(manifest_path)).resolve().parent / "product.json"
        if manifest_product.exists():
            return manifest_product
    product_id = pointer.get("product_id")
    if product_id:
        for scope in _candidate_scope_paths(settings, source_id, season_profile, theme, mode, subseason):
            candidate = scope / "runs" / str(product_id) / "product.json"
            if candidate.exists():
                return candidate
    discovered = _discover_latest_product_path(settings, source_id, season_profile, theme, mode, subseason)
    if discovered is not None:
        return discovered
    raise SeasonalMapArtifactsNotAvailableError("Seasonal map active pointer is incomplete.")


def _normalize_product_payload(
    payload: dict[str, Any],
    *,
    source_id: str,
    theme: str,
    season_profile: str,
    mode: str,
    subseason: str | None,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["theme"] = str(normalized.get("theme") or theme)
    normalized["season_profile"] = str(normalized.get("season_profile") or season_profile)
    normalized["mode"] = mode
    normalized["subseason"] = subseason
    normalized["mode_label"] = MODE_LABELS[mode]
    normalized["subseason_label"] = subseason
    normalized.setdefault("forecast_source", source_id)
    normalized.setdefault("forecast_source_label", _forecast_source_label(str(normalized["forecast_source"])))
    normalized.setdefault("source_run_id", "unknown")
    normalized.setdefault("refresh_interval_seconds", 1800)
    normalized.setdefault("freshness_threshold_hours", 18)
    normalized.setdefault("legend", [])
    normalized.setdefault("district_items", [])
    normalized.setdefault("region_items", [])
    normalized.setdefault("deterministic_legend", [])
    normalized.setdefault("deterministic_district_items", [])
    normalized.setdefault("deterministic_region_items", [])
    normalized.setdefault("district_count", len(normalized["district_items"]))
    normalized.setdefault("region_count", len(normalized["region_items"]))
    normalized["legend"] = _upgrade_legacy_legend_items(normalized["legend"], theme)
    normalized["deterministic_legend"] = _upgrade_legacy_legend_items(
        normalized.get("deterministic_legend", []),
        theme,
    )
    return normalized


def _upgrade_legacy_legend_items(items: Any, theme: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    reverse_probability_scale = _probability_reverse_scale(theme)
    upgraded: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        upgraded.append(
            {
                **item,
                "family_label": str(item.get("family_label") or label),
                "display_order": int(item.get("display_order", index)),
                "reverse_probability_scale": bool(item.get("reverse_probability_scale", reverse_probability_scale)),
            }
        )
    return upgraded


def _as_probability_product_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if _has_explicit_probability_payload(payload):
        return payload

    raise SeasonalProbabilityProductIncompleteError(
        "Published probability product is unavailable or incomplete for this selection. "
        "Explicit category percentages are required for probability mode."
    )


def _as_deterministic_product_payload(payload: dict[str, Any]) -> dict[str, Any]:
    deterministic_legend = payload.get("deterministic_legend")
    deterministic_district_items = payload.get("deterministic_district_items")
    deterministic_region_items = payload.get("deterministic_region_items")
    if deterministic_legend and deterministic_district_items is not None and deterministic_region_items is not None:
        return {
            **payload,
            "legend": deterministic_legend,
            "district_items": deterministic_district_items,
            "region_items": deterministic_region_items,
        }

    legacy_legend = payload.get("legend", [])
    return {
        **payload,
        "legend": legacy_legend,
        "district_items": [_legacy_deterministic_area_item(item) for item in payload.get("district_items", [])],
        "region_items": [_legacy_deterministic_area_item(item) for item in payload.get("region_items", [])],
    }


def _has_explicit_probability_payload(payload: dict[str, Any]) -> bool:
    legend = payload.get("legend")
    district_items = payload.get("district_items")
    region_items = payload.get("region_items")
    return (
        _is_probability_area_item_list(district_items, legend)
        and _is_probability_area_item_list(region_items, legend)
    )


def _is_probability_area_item_list(items: Any, legend: Any) -> bool:
    if not isinstance(items, list):
        return False
    return all(
        isinstance(item, dict) and _is_explicit_probability_metric(item.get("metric"), legend)
        for item in items
    )


def _is_explicit_probability_metric(metric: Any, legend: Any) -> bool:
    if not isinstance(metric, dict):
        return False
    categories = metric.get("category_probabilities")
    if not isinstance(categories, list) or not categories:
        return False
    if not all(_is_probability_category(item) for item in categories):
        return False
    if not isinstance(metric.get("dominant_percentage"), (int, float)):
        return False
    dominant = max(categories, key=lambda item: float(item["percentage"]))
    if str(metric.get("dominant_category_code")) != str(dominant["category_code"]):
        return False
    if str(metric.get("dominant_category_label")) != str(dominant["label"]):
        return False
    if str(metric.get("category_code")) != str(dominant["category_code"]):
        return False
    if str(metric.get("category_label")) != str(dominant["label"]):
        return False
    if not math.isclose(float(metric["dominant_percentage"]), float(dominant["percentage"]), abs_tol=0.1):
        return False
    total = sum(float(item["percentage"]) for item in categories)
    if not math.isclose(total, 100.0, abs_tol=0.2):
        return False
    if isinstance(legend, list) and legend:
        legend_codes = [str(item.get("category_code")) for item in legend if isinstance(item, dict)]
        category_codes = [str(item["category_code"]) for item in categories]
        if legend_codes != category_codes:
            return False
    return True


def _is_probability_category(category: Any) -> bool:
    return (
        isinstance(category, dict)
        and isinstance(category.get("category_code"), str)
        and isinstance(category.get("label"), str)
        and isinstance(category.get("hint"), str)
        and isinstance(category.get("color"), str)
        and isinstance(category.get("percentage"), (int, float))
    )


def _legacy_deterministic_area_item(item: dict[str, Any]) -> dict[str, Any]:
    metric = dict(item.get("metric") or {})
    upgraded_metric = {
        "theme": metric.get("theme"),
        "theme_label": metric.get("theme_label"),
        "value": metric.get("numeric_value"),
        "display_value": metric.get("display_value", ""),
        "unit": metric.get("unit"),
        "criteria_note": metric.get("criteria_note", ""),
        "interpretation": metric.get("interpretation", ""),
        "legend_label": metric.get("category_label", ""),
        "color": metric.get("color", "#75857b"),
    }
    return {**item, "metric": upgraded_metric}
