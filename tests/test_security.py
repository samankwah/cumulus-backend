from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from cumulus.main import app
from cumulus.settings import get_settings


@pytest.fixture()
def locked_down(monkeypatch):
    """Production defaults: legacy endpoints off, no unauthenticated admin, no admin key."""
    monkeypatch.setenv("CUMULUS_ENABLE_LEGACY_ENDPOINTS", "false")
    monkeypatch.setenv("CUMULUS_ALLOW_UNAUTHENTICATED_ADMIN", "false")
    monkeypatch.delenv("CUMULUS_ADMIN_API_KEY", raising=False)
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/predict"),
        ("post", "/advisory"),
        ("post", "/advisory/legacy"),
        ("post", "/forecast"),
        ("get", "/forecast/raster"),
        ("get", "/forecast/raster/sample?latitude=5.6&longitude=-0.2"),
        ("get", "/forecast/raster/tiles/6/31/30.png"),
        ("post", "/train"),
        ("post", "/nationwide/generate"),
        ("get", "/nationwide/run/active"),
        ("post", "/seasonal-map/generate?theme=onset&season_profile=northern_single&mode=probability"),
        ("get", "/seasonal-map/profiles"),
    ],
)
def test_legacy_endpoints_are_hidden_by_default(locked_down, method, path):
    assert getattr(locked_down, method)(path).status_code == 404


def test_admin_endpoints_are_hidden_without_a_configured_key(locked_down):
    assert locked_down.post("/forecast/products/refresh").status_code == 404
    assert locked_down.get("/health/details").status_code == 404


def test_admin_endpoints_require_the_configured_key(locked_down, monkeypatch):
    monkeypatch.setenv("CUMULUS_ADMIN_API_KEY", "s3cret")
    get_settings.cache_clear()
    assert locked_down.post("/forecast/products/refresh").status_code == 401
    assert locked_down.post("/forecast/products/refresh", headers={"X-API-Key": "wrong"}).status_code == 401
    assert locked_down.get("/health/details", headers={"X-API-Key": "wrong"}).status_code == 401
    details = locked_down.get("/health/details", headers={"X-API-Key": "s3cret"})
    assert details.status_code == 200 and "data_sources" in details.json()


def test_legacy_mutations_still_need_the_admin_key_when_enabled(locked_down, monkeypatch):
    monkeypatch.setenv("CUMULUS_ENABLE_LEGACY_ENDPOINTS", "true")
    monkeypatch.setenv("CUMULUS_ADMIN_API_KEY", "s3cret")
    get_settings.cache_clear()
    assert locked_down.post("/train", json={}).status_code == 401
    assert locked_down.post("/nationwide/generate").status_code == 401


def test_health_is_cheap_and_leaks_no_paths(locked_down):
    response = locked_down.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["active_forecast_path"] is None and body["station_path"] is None and body["data_sources"] == {}


def test_public_forecast_and_subseasonal_reads_stay_open(locked_down):
    # Not 404/401: the guards must not touch the endpoints the frontend uses.
    assert locked_down.get("/forecast/products/options").status_code == 200
    assert locked_down.get("/subseasonal/runs").status_code in {200, 503}


def test_serverless_runtime_never_generates_artifacts_on_read(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("CUMULUS_FORECAST_PRODUCTS__GENERATE_ON_READ", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().forecast_products.generate_on_read is False
    finally:
        get_settings.cache_clear()
    monkeypatch.delenv("VERCEL")
    get_settings.cache_clear()
    assert get_settings().forecast_products.generate_on_read is True
    get_settings.cache_clear()


def test_forecast_request_caps_locations():
    from pydantic import ValidationError

    from cumulus.schemas import ForecastRequest

    location = {"latitude": 5.6, "longitude": -0.2}
    with pytest.raises(ValidationError):
        ForecastRequest(locations=[location] * 501, forecast_source={})
    with pytest.raises(ValidationError):
        ForecastRequest(locations=[], forecast_source={})
