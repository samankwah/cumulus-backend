from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from cumulus.settings import DEFAULT_CORS_ALLOWED_ORIGINS, DEFAULT_FORECAST_PRODUCT_ARTIFACT_DIR, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_accepts_comma_separated_cors_allowed_origins(monkeypatch):
    monkeypatch.setenv(
        "CUMULUS_CORS_ALLOWED_ORIGINS",
        "https://frontend.example.com, http://localhost:3000",
    )

    settings = get_settings()

    assert settings.cors_allowed_origins == [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "https://seasonalforecast.netlify.app",
        "https://frontend.example.com",
    ]


def test_settings_accepts_json_cors_allowed_origins(monkeypatch):
    monkeypatch.setenv(
        "CUMULUS_CORS_ALLOWED_ORIGINS",
        '["https://frontend.example.com", "http://localhost:3000"]',
    )

    settings = get_settings()

    assert settings.cors_allowed_origins == [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "https://seasonalforecast.netlify.app",
        "https://frontend.example.com",
    ]


def test_settings_accepts_single_cors_allowed_origin(monkeypatch):
    monkeypatch.setenv("CUMULUS_CORS_ALLOWED_ORIGINS", "https://frontend.example.com")

    settings = get_settings()

    assert settings.cors_allowed_origins == [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "https://seasonalforecast.netlify.app",
        "https://frontend.example.com",
    ]


def test_settings_uses_default_cors_allowed_origins_for_blank_env(monkeypatch):
    monkeypatch.setenv("CUMULUS_CORS_ALLOWED_ORIGINS", "")

    settings = get_settings()

    assert settings.cors_allowed_origins == list(DEFAULT_CORS_ALLOWED_ORIGINS)


def test_serverless_settings_keep_committed_forecast_product_artifacts(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", raising=False)

    settings = get_settings()

    assert Path(settings.forecast_products.artifact_dir) == DEFAULT_FORECAST_PRODUCT_ARTIFACT_DIR


def test_runtime_dependencies_include_scipy_for_classic_netcdf_forecast_products():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependency_names = {
        dependency.split("[", 1)[0].split(">=", 1)[0].split("==", 1)[0].lower()
        for dependency in payload["project"]["dependencies"]
    }

    assert "scipy" in dependency_names
