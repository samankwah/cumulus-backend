"""Artifact-backed Ghana seasonal advisory map generation and serving."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cumulus.api.errors import (
    InvalidSeasonalModeError,
    InvalidSubseasonForProfileError,
    SeasonalMapArtifactsNotAvailableError,
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
    district_items: list[dict[str, Any]] = []
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
        district_items.append(
            _serialize_area_item(
                geography_type="district",
                geography_name=district["geography_name"],
                region_name=district["region_name"],
                location_id=district["location_id"],
                coverage_count=1,
                coverage_note=_district_coverage_note(
                    district=district,
                    profile=profile,
                    theme=normalized_theme,
                ),
                raw_metrics=raw_metrics,
                theme=normalized_theme,
                profile=profile,
                mode=normalized_mode,
                subseason=normalized_subseason,
                generated_at=generated_at,
            )
        )

    region_items = _build_region_items(
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
        "district_count": len(district_items),
        "region_count": len(region_items),
        "legend": _build_legend(normalized_theme, normalized_mode, profile, normalized_subseason),
        "district_items": district_items,
        "region_items": region_items,
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

    values: list[float] = []
    for position, timestamp in enumerate(index):
        repeated_value = rainfall_values[position % len(rainfall_values)]
        weight = _seasonal_weight(timestamp, onset_start, cessation_start)
        values.append(round(max(0.0, repeated_value * base_scale * weight), 3))

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
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in district_metrics:
        grouped[str(metric["region_name"])].append(metric)

    items: list[dict[str, Any]] = []
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
        items.append(
            _serialize_area_item(
                geography_type="region",
                geography_name=region_name,
                region_name=region_name,
                location_id=_slugify(region_name),
                coverage_count=len(metrics),
                coverage_note=_region_coverage_note(
                    region_name=region_name,
                    coverage_count=len(metrics),
                    profile=profile,
                    theme=theme,
                ),
                raw_metrics=raw_metrics,
                theme=theme,
                profile=profile,
                mode=mode,
                subseason=subseason,
                generated_at=generated_at,
            )
        )
    return items


def _serialize_area_item(
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
        "metric": _classify_metric(theme, raw_metrics, profile, mode, subseason, generated_at),
    }


def _classify_metric(
    theme: str,
    raw_metrics: dict[str, float],
    profile: SeasonalProfileConfig,
    mode: str,
    subseason: str | None,
    generated_at: datetime,
) -> dict[str, Any]:
    if theme == "onset":
        return _classify_onset(raw_metrics["onset_offset_days"], profile, generated_at)
    if theme == "cessation":
        return _classify_cessation(raw_metrics["cessation_offset_days"], profile, generated_at)
    if theme == "early_dry_spell":
        return _classify_dry_spell(
            theme="early_dry_spell",
            value=raw_metrics["early_dry_spell_days"],
            moderate_days=profile.early_dry_spell_moderate_days,
            high_days=profile.early_dry_spell_high_days,
        )
    if theme == "late_dry_spell":
        return _classify_dry_spell(
            theme="late_dry_spell",
            value=raw_metrics["late_dry_spell_days"],
            moderate_days=profile.late_dry_spell_moderate_days,
            high_days=profile.late_dry_spell_high_days,
        )
    if theme == "rainfall_amount":
        return _classify_rainfall_amount(
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
    return _classify_rainy_days(
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
        category_label = "Normal"
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
            f"using {onset_phrase} and no dry spell longer than {profile.onset_guard_max_dry_spell_days} days in the next "
            f"{profile.onset_guard_window_days} days."
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
    if value >= high_days:
        category_code = "high"
        category_label = "High"
        interpretation = f"{THEME_LABELS[theme]} pressure is elevated under the selected Ghana regime."
    elif value >= moderate_days:
        category_code = "moderate"
        category_label = "Moderate"
        interpretation = f"{THEME_LABELS[theme]} pressure is building and should be monitored."
    else:
        category_code = "low"
        category_label = "Low"
        interpretation = f"{THEME_LABELS[theme]} pressure remains limited in this regime run."
    window_note = "Longest dry run from onset to day 50." if theme == "early_dry_spell" else "Longest dry run from day 51 to cessation."
    return {
        "theme": theme,
        "theme_label": THEME_LABELS[theme],
        "category_code": category_code,
        "category_label": category_label,
        "numeric_value": round(value, 1),
        "display_value": f"{value:.1f} day(s)",
        "unit": "days",
        "criteria_note": (
            f"{window_note} Low below {moderate_days} days; Moderate from {moderate_days} to under {high_days}; "
            f"High from {high_days} days upward."
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
        category_label = "Below Normal"
        interpretation = (
            f"{profile.label} rainfall is below the expected onset-to-cessation total."
            if mode == "seasonal"
            else f"{profile.label} rainfall is below the expected {subseason} reporting window total."
        )
    elif value > upper:
        category_code = "above_normal"
        category_label = "Above Normal"
        interpretation = (
            f"{profile.label} rainfall is above the expected onset-to-cessation total."
            if mode == "seasonal"
            else f"{profile.label} rainfall is above the expected {subseason} reporting window total."
        )
    else:
        category_code = "near_normal"
        category_label = "Near Normal"
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
        category_label = "Fewer"
        interpretation = (
            f"{profile.label} rainy days are below the expected onset-to-cessation count."
            if mode == "seasonal"
            else f"{profile.label} rainy days are below the expected {subseason} reporting window count."
        )
    elif value > upper:
        category_code = "more"
        category_label = "More"
        interpretation = (
            f"{profile.label} rainy days are above the expected onset-to-cessation count."
            if mode == "seasonal"
            else f"{profile.label} rainy days are above the expected {subseason} reporting window count."
        )
    else:
        category_code = "normal"
        category_label = "Normal"
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
        "display_value": f"{value:.1f} day(s)",
        "unit": "days",
        "criteria_note": criteria_note,
        "interpretation": interpretation,
        "color": _legend_color("rainy_days", category_code),
    }


def _build_legend(
    theme: str,
    mode: str,
    profile: SeasonalProfileConfig,
    subseason: str | None = None,
) -> list[dict[str, str]]:
    onset_phrase = (
        "20 mm in 3 consecutive days"
        if profile.onset_requires_consecutive_days
        else f"20 mm in up to {profile.onset_window_days} days"
    )
    if theme == "onset":
        return [
            {
                "category_code": "early",
                "label": "Early",
                "hint": (
                    f"Detected earlier than the {profile.label.lower()} WMO onset band after "
                    f"{date(2026, profile.onset_search_start_month, profile.onset_search_start_day).strftime('%d %b')}."
                ),
                "color": "#1f8a5b",
            },
            {
                "category_code": "normal",
                "label": "Normal",
                "hint": f"Detected within the Ghana WMO onset band using {onset_phrase}.",
                "color": "#c9962b",
            },
            {
                "category_code": "late",
                "label": "Late",
                "hint": f"Detected later than the {profile.label.lower()} WMO onset band.",
                "color": "#c65a46",
            },
        ]
    if theme == "cessation":
        return [
            {
                "category_code": "early",
                "label": "Early",
                "hint": "Soil water balance depletes earlier than the regime timing band.",
                "color": "#1f8a5b",
            },
            {
                "category_code": "normal",
                "label": "Normal",
                "hint": "Soil water balance depletes within the regime timing band.",
                "color": "#c9962b",
            },
            {
                "category_code": "late",
                "label": "Late",
                "hint": "Soil water balance depletes later than the regime timing band.",
                "color": "#c65a46",
            },
        ]
    if theme == "early_dry_spell":
        return [
            {
                "category_code": "low",
                "label": "Low",
                "hint": "Longest dry run from onset to day 50 remains limited.",
                "color": "#2f8f6e",
            },
            {
                "category_code": "moderate",
                "label": "Moderate",
                "hint": "Longest dry run from onset to day 50 needs monitoring.",
                "color": "#c98b37",
            },
            {
                "category_code": "high",
                "label": "High",
                "hint": "Longest dry run from onset to day 50 is elevated.",
                "color": "#c45143",
            },
        ]
    if theme == "late_dry_spell":
        return [
            {
                "category_code": "low",
                "label": "Low",
                "hint": "Longest dry run from day 51 to cessation remains limited.",
                "color": "#2f8f6e",
            },
            {
                "category_code": "moderate",
                "label": "Moderate",
                "hint": "Longest dry run from day 51 to cessation needs monitoring.",
                "color": "#c98b37",
            },
            {
                "category_code": "high",
                "label": "High",
                "hint": "Longest dry run from day 51 to cessation is elevated.",
                "color": "#c45143",
            },
        ]
    if theme == "rainfall_amount":
        return [
            {
                "category_code": "below_normal",
                "label": "Below Normal",
                "hint": (
                    "Detected seasonal rainfall is below the regime normal."
                    if mode == "seasonal"
                    else f"Detected {subseason} rainfall is below the regime calendar normal."
                ),
                "color": "#c55a45",
            },
            {
                "category_code": "near_normal",
                "label": "Near Normal",
                "hint": (
                    "Detected seasonal rainfall stays within the regime normal band."
                    if mode == "seasonal"
                    else f"Detected {subseason} rainfall stays within the regime calendar band."
                ),
                "color": "#c9962b",
            },
            {
                "category_code": "above_normal",
                "label": "Above Normal",
                "hint": (
                    "Detected seasonal rainfall is above the regime normal."
                    if mode == "seasonal"
                    else f"Detected {subseason} rainfall is above the regime calendar normal."
                ),
                "color": "#1f8a5b",
            },
        ]
    return [
        {
            "category_code": "fewer",
            "label": "Fewer",
            "hint": (
                "Rainy-day count is below the regime normal."
                if mode == "seasonal"
                else f"Rainy-day count in {subseason} is below the regime calendar normal."
            ),
            "color": "#c55a45",
        },
        {
            "category_code": "normal",
            "label": "Normal",
            "hint": (
                "Rainy-day count stays within the regime normal band."
                if mode == "seasonal"
                else f"Rainy-day count in {subseason} stays within the regime calendar band."
            ),
            "color": "#c9962b",
        },
        {
            "category_code": "more",
            "label": "More",
            "hint": (
                "Rainy-day count is above the regime normal."
                if mode == "seasonal"
                else f"Rainy-day count in {subseason} is above the regime calendar normal."
            ),
            "color": "#1f8a5b",
        },
    ]


def _legend_color(theme: str, category_code: str) -> str:
    for item in _build_legend(theme, "seasonal", _fallback_profile_for_legend()):
        if item["category_code"] == category_code:
            return item["color"]
    return "#75857b"


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
    normalized.setdefault("district_count", len(normalized["district_items"]))
    normalized.setdefault("region_count", len(normalized["region_items"]))
    return normalized
