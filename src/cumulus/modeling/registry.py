"""Active model registry."""

from __future__ import annotations

from pathlib import Path

from cumulus.utils.io import read_json, write_json


REGISTRY_FILENAME = "active_model.json"


def save_active_model(
    registry_dir: Path,
    payload: dict[str, object],
    *,
    forecast_source: str | None = None,
) -> None:
    write_json(registry_dir / _registry_filename(forecast_source), payload)


def load_active_model(registry_dir: Path, *, forecast_source: str | None = None) -> dict[str, object]:
    source_registry = registry_dir / _registry_filename(forecast_source)
    if source_registry.exists():
        return read_json(source_registry)
    return read_json(registry_dir / REGISTRY_FILENAME)


def _registry_filename(forecast_source: str | None) -> str:
    if not forecast_source:
        return REGISTRY_FILENAME
    return f"active_model_{str(forecast_source).strip().lower().replace(' ', '_')}.json"
