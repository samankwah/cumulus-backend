from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from cumulus.main import app
from cumulus.services.seasonal_map_service import clear_seasonal_map_cache
from cumulus.settings import get_settings


def _configure_artifact_dir(monkeypatch, tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "seasonal-map"
    monkeypatch.setenv("CUMULUS_SEASONAL_MAP__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_DEFAULT_FORECAST_SOURCE", "configured")
    get_settings.cache_clear()
    clear_seasonal_map_cache()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _metric(theme: str) -> dict[str, object]:
    return {
        "theme": theme,
        "theme_label": "Onset Date" if theme == "onset" else "Seasonal Rainfall Total",
        "category_code": "normal" if theme == "onset" else "near_normal",
        "category_label": "Normal" if theme == "onset" else "Near Normal",
        "numeric_value": 0.0 if theme == "onset" else 320.4,
        "display_value": "24 Mar" if theme == "onset" else "320.4 mm",
        "unit": "days_from_reference" if theme == "onset" else "mm",
        "criteria_note": "Legacy compatibility payload.",
        "interpretation": "Legacy compatibility payload.",
        "color": "#c9962b",
    }


def _legacy_product(product_id: str, *, theme: str, season_profile: str) -> dict[str, object]:
    metric = _metric(theme)
    return {
        "product_id": product_id,
        "theme": theme,
        "season_profile": season_profile,
        "generated_at": "2026-04-24T13:14:44+00:00",
        "forecast_cycle": "24 Apr 2026 12:00 UTC",
        "forecast_source": "configured",
        "forecast_source_label": "Configured Forecast Feed",
        "source_run_id": "configured-legacy-run",
        "refresh_interval_seconds": 1800,
        "freshness_threshold_hours": 18,
        "district_count": 1,
        "region_count": 1,
        "legend": [{"category_code": metric["category_code"], "label": metric["category_label"], "hint": "Legacy hint", "color": metric["color"]}],
        "district_items": [
            {
                "location_id": "accra-metropolitan",
                "geography_type": "district",
                "geography_name": "Accra Metropolitan",
                "region_name": "Greater Accra",
                "coverage_count": 1,
                "coverage_note": "Legacy district coverage note.",
                "metric": metric,
            }
        ],
        "region_items": [
            {
                "location_id": "greater-accra",
                "geography_type": "region",
                "geography_name": "Greater Accra",
                "region_name": "Greater Accra",
                "coverage_count": 1,
                "coverage_note": "Legacy region coverage note.",
                "metric": metric,
            }
        ],
        "refresh_status": "fresh",
        "is_stale": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _active_url(theme: str, season_profile: str, mode: str, *, subseason: str | None = None) -> str:
    query = f"/seasonal-map/active?theme={theme}&season_profile={season_profile}&mode={mode}&forecast_source=configured"
    if subseason:
        query = f"{query}&subseason={subseason}"
    return query


def test_seasonal_mode_resolves_legacy_pointer_and_enriches_response(monkeypatch, tmp_path):
    artifact_dir = _configure_artifact_dir(monkeypatch, tmp_path)
    product_id = "seasonal_configured_northern_single_onset_20260424T131444Z"
    product_path = artifact_dir / "configured" / "northern_single" / "onset" / "runs" / product_id / "product.json"
    _write_json(product_path, _legacy_product(product_id, theme="onset", season_profile="northern_single"))
    _write_json(
        artifact_dir / "active_configured_northern_single_onset.json",
        {
            "product_id": product_id,
            "product_path": str(product_path),
        },
    )

    client = TestClient(app)
    response = client.get(_active_url("onset", "northern_single", "seasonal"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["product_id"] == product_id
    assert payload["mode"] == "seasonal"
    assert payload["subseason"] is None
    assert payload["mode_label"] == "Seasonal"
    assert payload["subseason_label"] is None


def test_seasonal_mode_prefers_mode_aware_pointer_over_legacy_pointer(monkeypatch, tmp_path):
    artifact_dir = _configure_artifact_dir(monkeypatch, tmp_path)
    legacy_id = "seasonal_configured_northern_single_onset_20260424T131444Z"
    legacy_path = artifact_dir / "configured" / "northern_single" / "onset" / "runs" / legacy_id / "product.json"
    _write_json(legacy_path, _legacy_product(legacy_id, theme="onset", season_profile="northern_single"))
    _write_json(
        artifact_dir / "active_configured_northern_single_onset.json",
        {
            "product_id": legacy_id,
            "product_path": str(legacy_path),
        },
    )

    new_id = "seasonal_configured_northern_single_onset_seasonal_20260425T000000Z"
    new_path = (
        artifact_dir / "configured" / "northern_single" / "onset" / "seasonal" / "runs" / new_id / "product.json"
    )
    new_payload = _legacy_product(new_id, theme="onset", season_profile="northern_single")
    new_payload["mode"] = "seasonal"
    new_payload["mode_label"] = "Seasonal"
    new_payload["subseason"] = None
    new_payload["subseason_label"] = None
    _write_json(new_path, new_payload)
    _write_json(
        artifact_dir / "active_configured_northern_single_onset_seasonal.json",
        {
            "product_id": new_id,
            "product_path": str(new_path),
            "mode": "seasonal",
            "subseason": None,
        },
    )

    client = TestClient(app)
    response = client.get(_active_url("onset", "northern_single", "seasonal"))

    assert response.status_code == 200
    assert response.json()["product_id"] == new_id


def test_seasonal_mode_discovers_legacy_run_directory_without_pointer(monkeypatch, tmp_path):
    artifact_dir = _configure_artifact_dir(monkeypatch, tmp_path)
    product_id = "seasonal_configured_southern_minor_rainfall_amount_20260424T131458Z"
    product_path = (
        artifact_dir / "configured" / "southern_minor" / "rainfall_amount" / "runs" / product_id / "product.json"
    )
    _write_json(product_path, _legacy_product(product_id, theme="rainfall_amount", season_profile="southern_minor"))

    client = TestClient(app)
    response = client.get(_active_url("rainfall_amount", "southern_minor", "seasonal"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["product_id"] == product_id
    assert payload["mode"] == "seasonal"
    assert payload["mode_label"] == "Seasonal"


def test_calendar_mode_does_not_fall_back_to_legacy_seasonal_artifacts(monkeypatch, tmp_path):
    artifact_dir = _configure_artifact_dir(monkeypatch, tmp_path)
    product_id = "seasonal_configured_southern_major_rainfall_amount_20260424T131458Z"
    product_path = (
        artifact_dir / "configured" / "southern_major" / "rainfall_amount" / "runs" / product_id / "product.json"
    )
    _write_json(product_path, _legacy_product(product_id, theme="rainfall_amount", season_profile="southern_major"))
    _write_json(
        artifact_dir / "active_configured_southern_major_rainfall_amount.json",
        {
            "product_id": product_id,
            "product_path": str(product_path),
        },
    )

    client = TestClient(app)
    response = client.get(_active_url("rainfall_amount", "southern_major", "calendar", subseason="MAM"))

    assert response.status_code == 503
    assert response.json()["error_code"] == "seasonal_map_artifacts_not_available"
