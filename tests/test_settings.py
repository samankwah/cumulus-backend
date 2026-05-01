from __future__ import annotations

import pytest

from cumulus.settings import DEFAULT_CORS_ALLOWED_ORIGINS, get_settings


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
        "https://frontend.example.com",
        "http://localhost:3000",
    ]


def test_settings_accepts_json_cors_allowed_origins(monkeypatch):
    monkeypatch.setenv(
        "CUMULUS_CORS_ALLOWED_ORIGINS",
        '["https://frontend.example.com", "http://localhost:3000"]',
    )

    settings = get_settings()

    assert settings.cors_allowed_origins == [
        "https://frontend.example.com",
        "http://localhost:3000",
    ]


def test_settings_accepts_single_cors_allowed_origin(monkeypatch):
    monkeypatch.setenv("CUMULUS_CORS_ALLOWED_ORIGINS", "https://frontend.example.com")

    settings = get_settings()

    assert settings.cors_allowed_origins == ["https://frontend.example.com"]


def test_settings_uses_default_cors_allowed_origins_for_blank_env(monkeypatch):
    monkeypatch.setenv("CUMULUS_CORS_ALLOWED_ORIGINS", "")

    settings = get_settings()

    assert settings.cors_allowed_origins == list(DEFAULT_CORS_ALLOWED_ORIGINS)
