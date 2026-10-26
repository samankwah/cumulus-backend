"""Per-run summary tables for every region and district, kept forever as the run was issued.

The partner API (``/v1``) serves these: one row per area with the whole-run and week-by-week
figures the drawer shows (rain amount and spread, rain days, spell days and chances, the chance of
an extreme day, onset). Rows are built with :func:`sample_area`, so a partner's number for a
district is exactly the one on the map.

Summaries are written at ingest to ``<archive_dir>/<model>/<run_id>.json.gz`` and never pruned,
unlike the run artifacts (``retention_runs``): insurers and analysts need the forecasts as issued.
Runs ingested before this existed are summarised on demand (and can be backfilled with
``python -m cumulus.subseasonal.summary``).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from cumulus.settings import Settings, get_settings
from cumulus.subseasonal.ingest import MANIFEST_FILE_NAME, for_model, list_run_ids, source_dir
from cumulus.subseasonal.metrics import rain_day

SCHEMA_VERSION = 2
LEVELS = ("region", "district")
SUMMARY_SUFFIX = ".json.gz"


def area_id(name: str, region: str | None = None) -> str:
    """A stable, URL-safe id: ``northern`` for a region, ``northern--tamale-metropolitan`` for a district."""
    return "--".join(_slug(part) for part in (region, name) if part)


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def archive_dir(settings: Settings) -> Path:
    return Path(settings.subseasonal.archive_dir) / settings.subseasonal.source_id


def summary_path(settings: Settings, run_id: str) -> Path:
    return archive_dir(settings) / f"{run_id}{SUMMARY_SUFFIX}"


def archived_run_ids(settings: Settings) -> list[str]:
    root = archive_dir(settings)
    if not root.is_dir():
        return []
    return sorted(path.name[: -len(SUMMARY_SUFFIX)] for path in root.glob(f"*{SUMMARY_SUFFIX}"))


def build_summary(settings: Settings, run_id: str) -> dict[str, Any]:
    """The summary table of one run, for every region and district."""
    from cumulus.services import subseasonal_service as service

    run = service.load_run(settings, run_id)
    manifest = run.manifest
    levels: dict[str, list[dict[str, Any]]] = {}
    for level in LEVELS:
        names, _matrix = run.area_matrices[level]
        rows = []
        for name in names:
            payload = service.sample_area(settings, level=level, name=name, run_id=run.run_id)
            rows.append(_row(level, payload, run.extreme_rain_mm))
        levels[level] = rows
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.run_id,
        "model": manifest.get("source_id", settings.subseasonal.source_id),
        "model_label": manifest.get("model_label", settings.subseasonal.model_label),
        "init_time": run.init_time.isoformat(),
        "valid_from": run.init_date.isoformat(),
        "valid_to": rain_day(run.init_date, run.day_count).isoformat(),
        "lead_days": run.day_count,
        "members": run.member_count,
        # The staging path on the ingest machine is of no use to readers of the archive.
        "provenance": {k: v for k, v in (manifest.get("provenance") or {}).items() if k != "source_path"},
        "thresholds": {
            "wet_day_mm": run.wet_threshold_mm,
            "dry_spell_days": run.dry_spell_min_days,
            "wet_spell_days": run.wet_spell_min_days,
            "extreme_rain_mm": run.extreme_rain_mm,
        },
        "observed_days": run.observed_days,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "levels": levels,
    }


def _row(level: str, payload: dict[str, Any], extreme_mm: float) -> dict[str, Any]:
    """One area's figures, flattened from its /subseasonal/area series."""
    region = payload.get("region")
    extreme_days = _extreme_days(payload, extreme_mm)
    name = payload["name"]
    totals = payload.get("member_totals") or []
    metrics = payload.get("metrics") or {}
    onset = payload.get("onset") or {}
    spell_total = payload.get("spell_chance_total") or {}
    return {
        "id": area_id(name, region if level == "district" else None),
        "name": name,
        "region": region,
        "cells": payload.get("cell_count"),
        "total": {
            "rain_mm": metrics.get("total_mm"),
            **_quantiles(totals),
            "rainy_days": metrics.get("rainy_days"),
            "dry_spell_days": metrics.get("dry_spell_days"),
            "wet_spell_days": metrics.get("wet_spell_days"),
            "max_day_mm": metrics.get("max_day_mm"),
            "extreme_days": metrics.get("extreme_days"),
            "dry_spell_chance": spell_total.get("dry"),
            "wet_spell_chance": spell_total.get("wet"),
            "chance_at_least_mm": payload.get("total_chance") or {},
        },
        "onset": {
            "date": onset.get("date"),
            "probability": onset.get("probability"),
            "provisional": onset.get("provisional"),
        }
        if onset.get("date")
        else None,
        "weeks": [_week(week, extreme_days) for week in payload.get("weeks") or []],
    }


def _extreme_days(payload: dict[str, Any], extreme_mm: float) -> list[float]:
    """Per day, the share (0-1) of members at or above the extreme amount, as in the drawer's count."""
    shares = []
    for day in payload.get("days") or []:
        members = day.get("members")
        amounts = [value for value in members if value is not None] if members else [day.get("value")]
        amounts = [value for value in amounts if value is not None]
        shares.append(sum(value >= extreme_mm for value in amounts) / len(amounts) if amounts else 0.0)
    return shares


def _week(week: dict[str, Any], extreme_days: list[float]) -> dict[str, Any]:
    spread = week.get("spread") or {}
    counts = week.get("day_counts") or {}
    spell = week.get("spell_chance") or {}
    return {
        "week": week["week"],
        "start_date": str(week["start_date"]),
        "end_date": str(week["end_date"]),
        "partial": bool(week.get("partial")),
        "rain_mm": week.get("value"),
        "p10_mm": spread.get("p10"),
        "p50_mm": spread.get("p50"),
        "p90_mm": spread.get("p90"),
        "rainy_days": (counts.get("rainy") or {}).get("mean"),
        "dry_spell_days": (counts.get("dry_spell") or {}).get("mean"),
        "wet_spell_days": (counts.get("wet_spell") or {}).get("mean"),
        "extreme_days": round(sum(extreme_days[week["start_day"] - 1 : week["end_day"]]), 1),
        "extreme_chance": week.get("extreme_prob"),
        "dry_spell_chance": spell.get("dry"),
        "wet_spell_chance": spell.get("wet"),
        "chance_at_least_mm": week.get("chance") or {},
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    """p10/p50/p90 of the members' totals; None for a single run."""
    if len(values) < 2:
        return {"p10_mm": None, "p50_mm": None, "p90_mm": None}
    ordered = sorted(float(value) for value in values)

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 1)

    return {"p10_mm": at(0.1), "p50_mm": at(0.5), "p90_mm": at(0.9)}


def write_summary(settings: Settings, run_id: str) -> Path:
    """Build and archive one run's summary; returns the file written."""
    summary = build_summary(settings, run_id)
    path = summary_path(settings, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # mtime=0 keeps the gzip bytes identical for identical content, so re-running makes no git change.
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(json.dumps(summary, separators=(",", ":"), default=str).encode("utf-8"))
    temporary.replace(path)
    return path


def load_summary(settings: Settings, run_id: str) -> dict[str, Any]:
    """The archived summary, or one built now (and kept in memory) for a run not yet archived."""
    path = summary_path(settings, run_id)
    if path.exists():
        return _read(str(path), path.stat().st_mtime_ns)
    if run_id not in list_run_ids(settings):
        from cumulus.api.errors import SubseasonalRunNotFoundError

        raise SubseasonalRunNotFoundError(f"Sub-seasonal run '{run_id}' is not available.")
    manifest_path = source_dir(settings) / run_id / MANIFEST_FILE_NAME
    return _built(settings.subseasonal.source_id, run_id, manifest_path.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _read(path: str, _mtime: int) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


@lru_cache(maxsize=8)
def _built(model: str, run_id: str, _manifest_mtime: int) -> dict[str, Any]:
    return build_summary(for_model(get_settings(), model), run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive the per-area summary of ingested 46-day runs.")
    parser.add_argument("--model", action="append", help="Model id (repeatable); default: every configured model.")
    parser.add_argument("--run-id", help="One run; default: every kept run without a summary.")
    parser.add_argument("--force", action="store_true", help="Rebuild summaries that already exist.")
    args = parser.parse_args(argv)
    settings = get_settings()
    models = args.model or list(settings.subseasonal.models) or [settings.subseasonal.source_id]
    for model in models:
        scoped = for_model(settings, model)
        run_ids = [args.run_id] if args.run_id else list_run_ids(scoped)
        for run_id in run_ids:
            if not args.force and summary_path(scoped, run_id).exists():
                continue
            print(f"{model}: {write_summary(scoped, run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
