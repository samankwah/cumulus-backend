from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from cumulus.main import app
from cumulus.settings import get_settings


REPO_FORECAST_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "forecast_products"


def _configure_forecast_artifacts(monkeypatch, tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "forecast-products"
    for theme in ("onset", "early_dry_spell"):
      shutil.copytree(REPO_FORECAST_ARTIFACT_DIR / theme, artifact_dir / theme)
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    get_settings.cache_clear()
    return artifact_dir


def test_forecast_product_options_return_all_target_themes_with_readiness(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/products/options")

    assert response.status_code == 200
    payload = response.json()
    themes = {item["theme"]: item for item in payload}

    assert list(themes) == [
        "onset",
        "early_dry_spell",
        "cessation",
        "late_dry_spell",
        "rainfall_amount",
        "rainy_days",
    ]
    assert themes["onset"]["enabled"] is True
    assert themes["onset"]["requires_season"] is True
    assert themes["onset"]["seasons"] == ["northern_single", "southern_major", "southern_minor"]
    assert themes["early_dry_spell"]["enabled"] is True
    assert themes["cessation"]["enabled"] is False
    assert themes["cessation"]["reason"] == "artifacts_not_generated"
    assert themes["rainfall_amount"]["requires_subseason"] is True
    assert themes["rainfall_amount"]["enabled"] is False
    assert themes["rainfall_amount"]["subseasons"] == ["MAM", "AMJ", "MJJ", "JJA", "JAS", "SON"]
    assert themes["rainy_days"]["enabled"] is False


def test_active_probability_product_returns_absolute_asset_urls(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/probability/active?theme=onset&season_profile=northern_single")

    assert response.status_code == 200
    payload = response.json()
    assert payload["theme"] == "onset"
    assert payload["season_profile"] == "northern_single"
    assert payload["season_label"] == "Northern Uni Modal Seasonal"
    assert payload["tile_url"].startswith("http://testserver/forecast/probability/tiles/{z}/{x}/{y}.png?")
    assert "theme=onset" in payload["tile_url"]
    assert "season_profile=northern_single" in payload["tile_url"]
    assert payload["preview_url"].startswith("http://testserver/forecast/probability/preview.png?")
    assert "season_profile=northern_single" in payload["preview_url"]


def test_scoped_probability_product_requires_published_manifest_even_when_generic_exists(monkeypatch, tmp_path):
    artifact_dir = _configure_forecast_artifacts(monkeypatch, tmp_path)
    shutil.rmtree(artifact_dir / "onset" / "probability" / "southern_minor")
    client = TestClient(app)

    response = client.get("/forecast/probability/active?theme=onset&season_profile=southern_minor")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "forecast_product_artifacts_not_available"
    assert "theme=onset" in payload["detail"]
    assert "view_mode=probability" in payload["detail"]
    assert "season_profile=southern_minor" in payload["detail"]
    assert not (artifact_dir / "onset" / "probability" / "southern_minor" / "active.json").exists()


def test_scoped_deterministic_product_still_loads_when_manifest_exists(monkeypatch, tmp_path):
    artifact_dir = _configure_forecast_artifacts(monkeypatch, tmp_path)
    shutil.rmtree(artifact_dir / "onset" / "probability" / "southern_minor")
    client = TestClient(app)

    response = client.get("/forecast/deterministic/active?theme=onset&season_profile=southern_minor")

    assert response.status_code == 200
    payload = response.json()
    assert payload["theme"] == "onset"
    assert payload["season_profile"] == "southern_minor"
    assert payload["season_label"] == "Southern Minor Season"


def test_scoped_probability_sample_tile_and_preview_require_published_manifest(monkeypatch, tmp_path):
    artifact_dir = _configure_forecast_artifacts(monkeypatch, tmp_path)
    shutil.rmtree(artifact_dir / "onset" / "probability" / "southern_minor")
    client = TestClient(app)

    responses = [
        client.get(
            "/forecast/probability/sample?theme=onset&season_profile=southern_minor&latitude=5.6037&longitude=-0.187"
        ),
        client.get("/forecast/probability/tiles/6/31/29.png?theme=onset&season_profile=southern_minor"),
        client.get("/forecast/probability/preview.png?theme=onset&season_profile=southern_minor"),
    ]

    for response in responses:
        assert response.status_code == 503
        payload = response.json()
        assert payload["error_code"] == "forecast_product_artifacts_not_available"
        assert "view_mode=probability" in payload["detail"]
        assert "season_profile=southern_minor" in payload["detail"]
    assert not (artifact_dir / "onset" / "probability" / "southern_minor" / "active.json").exists()


def test_missing_generated_product_combination_returns_unavailable_error(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/deterministic/active?theme=cessation&season_profile=southern_major")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "forecast_product_artifacts_not_available"
    assert "season_profile=southern_major" in payload["detail"]


def test_probability_sample_endpoint_honors_season_selector(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get(
        "/forecast/probability/sample?theme=onset&season_profile=southern_major&latitude=5.6037&longitude=-0.187"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["theme"] == "onset"
    assert payload["season_profile"] == "southern_major"
    assert payload["season_label"] == "Southern Major Season"
    assert payload["subseason"] is None
    assert payload["nearest_latitude"] is not None
    assert payload["nearest_longitude"] is not None


def test_subseason_theme_requires_subseason_selector(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/probability/active?theme=rainfall_amount")

    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_forecast_product_selection"
