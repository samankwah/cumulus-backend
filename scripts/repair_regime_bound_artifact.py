"""Backfill a regime-bound seasonal artifact after footprint rules change."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from cumulus.services.seasonal_map_service import (
    REGIME_BOUND_THEMES,
    _active_pointer_path,
    _artifact_scope_path,
    _classify_metric,
    _district_coverage_note,
    _district_matches_profile_footprint,
    _legacy_active_pointer_path,
    _load_district_catalog,
    _region_coverage_note,
    _resolve_profile_config,
    _slugify,
)
from cumulus.settings import get_settings
from cumulus.utils.io import ensure_directory, read_json, write_json


def _pointer_path(settings, source_id: str, season_profile: str, theme: str, mode: str):
    canonical = _active_pointer_path(settings, source_id, season_profile, theme, mode, None)
    if canonical.exists():
        return canonical
    return _legacy_active_pointer_path(settings, source_id, season_profile, theme)


def _raw_metrics_for_theme(theme: str, numeric_value: float) -> dict[str, float]:
    raw_metrics = {
        "onset_offset_days": 0.0,
        "cessation_offset_days": 0.0,
        "early_dry_spell_days": 0.0,
        "late_dry_spell_days": 0.0,
        "rainfall_amount_mm": 0.0,
        "rainfall_normal_mm": 0.0,
        "rainy_days_count": 0.0,
        "rainy_days_normal": 0.0,
        "season_length_days": 0.0,
    }
    if theme == "onset":
        raw_metrics["onset_offset_days"] = numeric_value
    elif theme == "cessation":
        raw_metrics["cessation_offset_days"] = numeric_value
    elif theme == "early_dry_spell":
        raw_metrics["early_dry_spell_days"] = numeric_value
    elif theme == "late_dry_spell":
        raw_metrics["late_dry_spell_days"] = numeric_value
    return raw_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair a regime-bound seasonal artifact footprint.")
    parser.add_argument("--season-profile", required=True)
    parser.add_argument("--theme", required=True)
    parser.add_argument("--source-id", default="configured")
    parser.add_argument("--mode", default="seasonal")
    args = parser.parse_args()

    theme = str(args.theme).strip().lower()
    if theme not in REGIME_BOUND_THEMES:
        raise SystemExit(f"theme must be one of: {', '.join(REGIME_BOUND_THEMES)}")
    if str(args.mode).strip().lower() != "seasonal":
        raise SystemExit("Only seasonal regime-bound artifacts are supported by this repair tool.")

    settings = get_settings()
    season_profile, profile = _resolve_profile_config(settings, args.season_profile)
    source_id = str(args.source_id).strip().lower()

    pointer_path = _pointer_path(settings, source_id, season_profile, theme, "seasonal")
    pointer = read_json(pointer_path)
    payload = read_json(Path(pointer["product_path"]))

    district_catalog = _load_district_catalog(
        str(settings.seasonal_map.district_geojson_path),
        settings.seasonal_map.northern_latitude_threshold,
    )
    districts_by_id = {district["location_id"]: district for district in district_catalog}

    filtered_district_items: list[dict[str, object]] = []
    for item in payload["district_items"]:
        district = districts_by_id.get(item["location_id"])
        if district is None or not _district_matches_profile_footprint(district, profile):
            continue
        filtered_district_items.append(
            {
                **item,
                "coverage_note": _district_coverage_note(district=district, profile=profile, theme=theme),
            }
        )

    region_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in filtered_district_items:
        region_groups[str(item["region_name"])].append(item)

    generated_at = datetime.fromisoformat(str(payload["generated_at"]))
    filtered_region_items: list[dict[str, object]] = []
    for region_name, items in sorted(region_groups.items()):
        numeric_values = [float(item["metric"]["numeric_value"]) for item in items if item["metric"]["numeric_value"] is not None]
        region_numeric_value = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
        filtered_region_items.append(
            {
                "location_id": _slugify(region_name),
                "geography_type": "region",
                "geography_name": region_name,
                "region_name": region_name,
                "coverage_count": len(items),
                "coverage_note": _region_coverage_note(
                    region_name=region_name,
                    coverage_count=len(items),
                    profile=profile,
                    theme=theme,
                ),
                "metric": _classify_metric(
                    theme,
                    _raw_metrics_for_theme(theme, region_numeric_value),
                    profile,
                    "seasonal",
                    None,
                    generated_at,
                ),
            }
        )

    repaired_at = datetime.now(UTC)
    product_id = f"{payload['product_id']}_footprintfix_{repaired_at.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_directory(
        _artifact_scope_path(settings, source_id, season_profile, theme, "seasonal", None) / "runs" / product_id
    )
    product_path = run_dir / "product.json"
    manifest_path = run_dir / "manifest.json"

    repaired_payload = {
        **payload,
        "product_id": product_id,
        "mode": "seasonal",
        "subseason": None,
        "mode_label": "Seasonal",
        "subseason_label": None,
        "district_count": len(filtered_district_items),
        "region_count": len(filtered_region_items),
        "district_items": filtered_district_items,
        "region_items": filtered_region_items,
    }
    repaired_manifest = {
        **repaired_payload,
        "product_path": str(product_path),
        "manifest_path": str(manifest_path),
    }

    write_json(product_path, repaired_payload)
    write_json(manifest_path, repaired_manifest)

    active_payload = {
        "product_id": product_id,
        "product_path": str(product_path),
        "manifest_path": str(manifest_path),
        "mode": "seasonal",
        "subseason": None,
    }
    write_json(_active_pointer_path(settings, source_id, season_profile, theme, "seasonal", None), active_payload)
    write_json(_legacy_active_pointer_path(settings, source_id, season_profile, theme), active_payload)

    print(
        json.dumps(
            {
                "product_id": product_id,
                "district_count": len(filtered_district_items),
                "region_count": len(filtered_region_items),
                "product_path": str(product_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
