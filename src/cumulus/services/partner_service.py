"""The open API (``/v1``): GMet's 46-day forecast, free for anyone, from the public to stakeholders
such as NADMO, VRA, MoFA, FAO, NCA and insurers.

Everything is read from the per-run summaries (``cumulus.subseasonal.summary``) or, for one area's
full series, from :func:`sample_area`, so partners get exactly the figures on the map. Responses
share one envelope: ``meta`` (which run, which model, when issued, attribution) and ``data``.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
import io
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr

from cumulus.api.errors import InvalidSubseasonalLayerError, SubseasonalAreaNotFoundError, SubseasonalRunNotFoundError
from cumulus.services import subseasonal_service as service
from cumulus.settings import Settings
from cumulus.subseasonal import metrics
from cumulus.subseasonal import summary as summaries
from cumulus.subseasonal.ingest import MANIFEST_FILE_NAME, list_run_ids, source_dir

logger = logging.getLogger(__name__)

ATTRIBUTION = "Delivered by the Ghana Meteorological Agency (GMet) · forecast models by the Cumulus team"
DISCLAIMER = (
    "Probabilistic guidance, not a guarantee. Day-to-day detail beyond about 10 days is indicative only; "
    "use weekly figures and chances for planning."
)
LEVELS = summaries.LEVELS
FORMATS = ("json", "csv", "geojson")
GRID_FORMATS = ("json", "csv", "netcdf")
# Amounts offered for "chance of at least" over a chosen period (mm).
PERIOD_AMOUNTS = (10.0, 25.0, 50.0, 100.0, 200.0)
# How long the admin's choice of official model is trusted before asking the frontend again.
OFFICIAL_MODEL_TTL_SECONDS = 300
OFFICIAL_MODEL_TIMEOUT_SECONDS = 2.0

_official_lock = threading.Lock()
_official_cache: dict[str, tuple[float, str | None]] = {}


# --------------------------------------------------------------------------- models and runs


def official_model(settings: Settings) -> str:
    """The model GMet marked as official: the admin's setting, else configuration, else the default."""
    config = settings.subseasonal
    candidates = [_remote_official_model(config.official_model_url), config.official_model, config.source_id]
    known = set(config.models) | {config.source_id}
    return next(model for model in candidates if model and model in known)


def _remote_official_model(url: str | None) -> str | None:
    if not url:
        return None
    now = time.monotonic()
    with _official_lock:
        cached = _official_cache.get(url)
        if cached and now - cached[0] < OFFICIAL_MODEL_TTL_SECONDS:
            return cached[1]
    value: str | None = None
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=OFFICIAL_MODEL_TIMEOUT_SECONDS) as response:
            value = json.loads(response.read().decode("utf-8")).get("officialModel") or None
    except (OSError, ValueError) as exc:
        logger.warning("official_model_fetch_failed url=%s error=%s", url, exc)
        # Keep the last good answer through a blip rather than flipping models.
        value = cached[1] if cached else None
    with _official_lock:
        _official_cache[url] = (now, value)
    return value


def scoped(settings: Settings, model: str | None) -> tuple[Settings, str, bool]:
    """(settings for the model, model id, whether it is the official one); default: the official model."""
    official = official_model(settings)
    model_id = (model or official).strip()
    return service.model_settings(settings, model_id), model_id, model_id == official


def _run_id(scoped_settings: Settings, run: str | None) -> str:
    if run is None:
        return service._active_run_id(scoped_settings)
    if run in list_run_ids(scoped_settings) or run in summaries.archived_run_ids(scoped_settings):
        return run
    raise SubseasonalRunNotFoundError(f"Run '{run}' is not available for this model.")


def _meta(summary: dict[str, Any], *, official: bool, active_run: str | None) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "model": summary["model"],
        "model_label": summary.get("model_label"),
        "official": official,
        "active": summary["run_id"] == active_run,
        "issued": summary["init_time"],
        "valid_from": summary["valid_from"],
        "valid_to": summary["valid_to"],
        "lead_days": summary.get("lead_days"),
        "members": summary.get("members"),
        "provenance": _public_provenance(summary.get("provenance") or {}),
        "thresholds": summary.get("thresholds") or {},
        "attribution": ATTRIBUTION,
        "disclaimer": DISCLAIMER,
    }


# Where the files sat on the modelling team's machines: internal, so never published.
PRIVATE_PROVENANCE_KEYS = ("source_path",)


def _public_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in provenance.items() if key not in PRIVATE_PROVENANCE_KEYS}


def _active_or_none(scoped_settings: Settings) -> str | None:
    try:
        return service._active_run_id(scoped_settings)
    except Exception:  # noqa: BLE001 - archive-only model: no live run
        return None


def latest_run(settings: Settings, model: str | None) -> dict[str, Any]:
    model_settings, _model_id, official = scoped(settings, model)
    run_id = _run_id(model_settings, None)
    summary = summaries.load_summary(model_settings, run_id)
    meta = _meta(summary, official=official, active_run=run_id)
    meta["is_stale"] = (datetime.now(UTC).date() - datetime.fromisoformat(summary["init_time"]).date()).days > int(
        model_settings.subseasonal.stale_after_days
    )
    return {"meta": meta, "data": None}


def list_partner_runs(settings: Settings, model: str | None, start: str | None, end: str | None) -> dict[str, Any]:
    """Every run partners can ask for: those still kept and every archived one, newest first."""
    model_settings, model_id, official = scoped(settings, model)
    live = set(list_run_ids(model_settings))
    archived = set(summaries.archived_run_ids(model_settings))
    active = _active_or_none(model_settings)
    runs = []
    for run_id in sorted(live | archived, reverse=True):
        issued = _issued(model_settings, run_id)
        day = issued[:10] if issued else None
        if day and ((start and day < start) or (end and day > end)):
            continue
        runs.append(
            {"run_id": run_id, "issued": issued, "active": run_id == active, "live": run_id in live, "archived": run_id in archived}
        )
    return {
        "meta": {"model": model_id, "official": official, "attribution": ATTRIBUTION, "disclaimer": DISCLAIMER},
        "data": runs,
    }


def _issued(model_settings: Settings, run_id: str) -> str | None:
    manifest_path = source_dir(model_settings) / run_id / MANIFEST_FILE_NAME
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))["init_time"]
        except (OSError, ValueError, KeyError):
            pass
    stamp = run_id.rsplit("_", 1)[-1]
    try:
        return datetime.strptime(stamp, "%Y%m%d%H").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return None


# --------------------------------------------------------------------------- areas and summaries


def _level(level: str) -> str:
    if level not in LEVELS:
        raise InvalidSubseasonalLayerError(f"Unknown level '{level}'. Choose one of: {', '.join(LEVELS)}.")
    return level


def run_summary(settings: Settings, model: str | None, run: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(summary, meta) for a run (default: the active run of the official model)."""
    model_settings, _model_id, official = scoped(settings, model)
    run_id = _run_id(model_settings, run)
    summary = summaries.load_summary(model_settings, run_id)
    return summary, _meta(summary, official=official, active_run=_active_or_none(model_settings))


def list_areas(settings: Settings, model: str | None, level: str) -> dict[str, Any]:
    summary, meta = run_summary(settings, model, None)
    rows = summary["levels"][_level(level)]
    return {
        "meta": meta,
        "data": [{"id": row["id"], "name": row["name"], "region": row["region"], "level": level} for row in rows],
    }


def area_summaries(settings: Settings, model: str | None, run: str | None, level: str) -> dict[str, Any]:
    summary, meta = run_summary(settings, model, run)
    return {"meta": meta, "data": summary["levels"][_level(level)]}


def _live(settings: Settings, model: str | None, run: str | None) -> tuple[Settings, dict[str, Any], dict[str, Any]]:
    """(model settings, summary, meta) for a run whose full data is still kept (not archive-only)."""
    summary, meta = run_summary(settings, model, run)
    model_settings, _model_id, _official = scoped(settings, model)
    if summary["run_id"] not in list_run_ids(model_settings):
        raise SubseasonalRunNotFoundError(
            f"Run '{summary['run_id']}' is archived: only its summary is kept. Use /v1/summary for archived runs."
        )
    return model_settings, summary, meta


def _series(
    settings: Settings,
    model: str | None,
    run: str | None,
    *,
    level: str | None,
    area: str | None,
    latitude: float | None,
    longitude: float | None,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """(meta, full series with members, area id) for one area (by id or name) or one point."""
    model_settings, summary, meta = _live(settings, model, run)
    if area is not None:
        level_key = _level(level or "district")
        row = _find_area(summary, level_key, area)
        series = service.sample_area(model_settings, level=level_key, name=row["name"], run_id=summary["run_id"])
        return meta, series, row["id"]
    if latitude is None or longitude is None:
        raise InvalidSubseasonalLayerError("Give an area (level and area) or a point (lat and lon).")
    series = service.sample_point(model_settings, latitude=latitude, longitude=longitude, run_id=summary["run_id"])
    return meta, series, None


def _without_members(series: dict[str, Any]) -> dict[str, Any]:
    series.pop("members", None)
    series.pop("member_totals", None)
    for item in (*series.get("days", []), *series.get("weeks", [])):
        item.pop("members", None)
    return series


def area_forecast(
    settings: Settings, model: str | None, run: str | None, level: str, area: str, include_members: bool = False
) -> dict[str, Any]:
    """One area's full daily and weekly series (as the drawer shows it), found by id or name."""
    meta, series, area_key = _series(settings, model, run, level=level, area=area, latitude=None, longitude=None)
    if not include_members:
        series = _without_members(series)
    return {"meta": meta, "data": {"id": area_key, **series}}


def point_forecast(
    settings: Settings, model: str | None, run: str | None, latitude: float, longitude: float, include_members: bool = False
) -> dict[str, Any]:
    """The full series of the grid cell nearest a point (as when the map is clicked)."""
    meta, series, _area = _series(settings, model, run, level=None, area=None, latitude=latitude, longitude=longitude)
    if not include_members:
        series = _without_members(series)
    return {"meta": meta, "data": series}


def period_rainfall(
    settings: Settings,
    model: str | None,
    run: str | None,
    start: date,
    end: date,
    *,
    level: str | None = None,
    area: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Rain over any dates inside the run, for an area or a point: mean, members' spread and chances."""
    meta, series, area_key = _series(settings, model, run, level=level, area=area, latitude=latitude, longitude=longitude)
    days = [day for day in series["days"] if start.isoformat() <= str(day["date"]) <= end.isoformat()]
    if start > end or not days or str(days[0]["date"]) != start.isoformat() or str(days[-1]["date"]) != end.isoformat():
        raise InvalidSubseasonalLayerError(
            f"Choose dates from {series['days'][0]['date']} to {series['days'][-1]['date']}, start on or before end."
        )
    wet_mm = float((series.get("thresholds") or {}).get("wet_day_mm") or 1.0)
    member_totals: list[float] = []
    member_rainy: list[int] = []
    for member in range(len(days[0].get("members") or [None])):
        amounts = [(day.get("members") or [day.get("value")])[member] for day in days]
        amounts = [value for value in amounts if value is not None]
        if amounts:
            member_totals.append(sum(amounts))
            member_rainy.append(sum(value >= wet_mm for value in amounts))
    place_keys = ("kind", "name", "region", "district", "latitude", "longitude", "nearest_latitude", "nearest_longitude", "cell_count")
    return {
        "meta": meta,
        "data": {
            "id": area_key,
            **{key: series.get(key) for key in place_keys},
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "days": len(days),
            "observed_days": sum(bool(day.get("observed")) for day in days),
            "rain_mm": round(sum(member_totals) / len(member_totals), 1) if member_totals else None,
            **summaries._quantiles(member_totals),
            "min_mm": round(min(member_totals), 1) if member_totals else None,
            "max_mm": round(max(member_totals), 1) if member_totals else None,
            "rainy_days": round(sum(member_rainy) / len(member_rainy), 1) if member_rainy else None,
            "chance_at_least_mm": {
                f"{amount:g}": round(sum(total >= amount for total in member_totals) / len(member_totals) * 100)
                for amount in PERIOD_AMOUNTS
            }
            if len(member_totals) > 1
            else {},
            "daily_mm": [
                {"date": str(day["date"]), "rain_mm": day.get("value"), "observed": bool(day.get("observed"))} for day in days
            ],
        },
    }


# --------------------------------------------------------------------------- maps and grids


def map_layer(settings: Settings, model: str | None, run: str | None, base_url: str, **selection: Any) -> dict[str, Any]:
    """A map layer as on the GMet map: title, legend, national stats and an XYZ tile URL (PNG)."""
    model_settings, summary, meta = _live(settings, model, run)
    layer = service.get_layer(model_settings, run_id=summary["run_id"], **selection)
    layer["tile_url"] = base_url.rstrip("/") + layer["tile_url"]
    for key in ("timeline", "timeline_unit", "progress"):
        layer.pop(key, None)
    return {"meta": meta, "data": layer}


def grid(
    settings: Settings,
    model: str | None,
    run: str | None,
    *,
    start: date | None = None,
    end: date | None = None,
    **selection: Any,
) -> dict[str, Any]:
    """The values behind a map layer on the forecast grid, or rain summed over chosen dates."""
    model_settings, summary, meta = _live(settings, model, run)
    prepared = service.load_run(model_settings, summary["run_id"])
    if start is not None or end is not None:
        first = (start - prepared.init_date).days + 1 if start else 1
        last = (end - prepared.init_date).days + 1 if end else prepared.day_count
        if not 1 <= first <= last <= prepared.day_count:
            raise InvalidSubseasonalLayerError(
                f"Choose dates from {metrics.rain_day(prepared.init_date, 1)} to "
                f"{metrics.rain_day(prepared.init_date, prepared.day_count)}, start on or before end."
            )
        window = prepared.daily[first - 1 : last]
        values = np.where(np.all(np.isnan(window), axis=0), np.nan, np.nansum(window, axis=0))
        info = {
            "layer": "rainfall",
            "title": "Rainfall over the chosen dates" + (" (ensemble mean)" if prepared.member_count > 1 else ""),
            "unit": "mm",
        }
    else:
        layer = selection.get("layer") or "rainfall"
        selected = service.resolve_selection(
            prepared,
            layer,
            selection.get("aggregation"),
            selection.get("index"),
            selection.get("stat"),
            selection.get("threshold"),
            selection.get("spell_days"),
            service._spell_range(model_settings, layer),
        )
        values = service.layer_field(prepared, selected)
        first, last = service._selection_days(prepared, selected)
        info = {
            "layer": selected.layer,
            "aggregation": selected.aggregation,
            "index": selected.index,
            "title": service._selection_title(prepared, selected),
            "unit": service._legend_for(prepared, selected).unit,
        }
    weights = prepared.land_weights
    step = abs(float(prepared.latitudes[1] - prepared.latitudes[0])) if prepared.latitudes.size > 1 else None
    return {
        "meta": meta,
        "data": {
            **info,
            "start_date": metrics.rain_day(prepared.init_date, first).isoformat(),
            "end_date": metrics.rain_day(prepared.init_date, last).isoformat(),
            "resolution_degrees": round(step, 4) if step else None,
            "latitudes": [round(float(value), 4) for value in prepared.latitudes],
            "longitudes": [round(float(value), 4) for value in prepared.longitudes],
            "values": [[round(float(value), 2) if np.isfinite(value) else None for value in row] for row in np.asarray(values)],
            "in_ghana": None if weights is None else (np.asarray(weights).reshape(np.shape(values)) > 0).tolist(),
        },
    }


def grid_csv(payload: dict[str, Any]) -> str:
    """One row per grid cell: latitude, longitude, value and whether the cell is in Ghana."""
    meta, data = payload["meta"], payload["data"]
    buffer = io.StringIO()
    buffer.write(
        f"# {data['title']} ({data['unit']}), {data['start_date']} to {data['end_date']}. Run {meta['run_id']}. {ATTRIBUTION}.\n"
    )
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["latitude", "longitude", "value", "in_ghana"])
    mask = data.get("in_ghana")
    for i, latitude in enumerate(data["latitudes"]):
        for j, longitude in enumerate(data["longitudes"]):
            writer.writerow([latitude, longitude, data["values"][i][j], "" if mask is None else int(mask[i][j])])
    return buffer.getvalue()


def grid_netcdf(payload: dict[str, Any]) -> bytes:
    """The grid as a small NetCDF file (classic format) for GIS and analysis tools."""
    meta, data = payload["meta"], payload["data"]
    values = np.array([[np.nan if value is None else value for value in row] for row in data["values"]], dtype=np.float32)
    variables: dict[str, Any] = {"value": (("latitude", "longitude"), values, {"units": data["unit"], "long_name": data["title"]})}
    if data.get("in_ghana") is not None:
        variables["in_ghana"] = (("latitude", "longitude"), np.array(data["in_ghana"], dtype=np.int8))
    dataset = xr.Dataset(
        variables,
        coords={"latitude": data["latitudes"], "longitude": data["longitudes"]},
        attrs={
            "run_id": meta["run_id"],
            "model": meta["model"],
            "issued": str(meta["issued"]),
            "start_date": data["start_date"],
            "end_date": data["end_date"],
            "attribution": ATTRIBUTION,
            "disclaimer": DISCLAIMER,
        },
    )
    return bytes(dataset.to_netcdf(engine="scipy"))


def _find_area(summary: dict[str, Any], level: str, area: str) -> dict[str, Any]:
    wanted = area.strip().casefold()
    for row in summary["levels"][level]:
        if wanted in (row["id"], row["name"].casefold()):
            return row
    raise SubseasonalAreaNotFoundError(f"No {level} '{area}'. List them with /v1/areas?level={level}.")


def alerts(
    settings: Settings,
    model: str | None,
    run: str | None,
    level: str,
    extreme_chance: float,
    dry_spell_chance: float,
    wet_spell_chance: float | None,
) -> dict[str, Any]:
    """Area-weeks at or above the chosen chances: a short list for disaster and water managers."""
    summary, meta = run_summary(settings, model, run)
    rules = [("extreme_rain", "extreme_chance", extreme_chance), ("dry_spell", "dry_spell_chance", dry_spell_chance)]
    if wet_spell_chance is not None:
        rules.append(("wet_spell", "wet_spell_chance", wet_spell_chance))
    found = []
    for row in summary["levels"][_level(level)]:
        for week in row["weeks"]:
            for kind, key, threshold in rules:
                value = week.get(key)
                if value is not None and value >= threshold:
                    found.append(
                        {
                            "id": row["id"],
                            "name": row["name"],
                            "region": row["region"],
                            "kind": kind,
                            "chance": value,
                            "week": week["week"],
                            "start_date": week["start_date"],
                            "end_date": week["end_date"],
                            "rain_mm": week.get("rain_mm"),
                        }
                    )
    found.sort(key=lambda item: (item["start_date"], -item["chance"], item["name"]))
    meta["criteria"] = {key: threshold for _kind, key, threshold in rules}
    return {"meta": meta, "data": found}


# --------------------------------------------------------------------------- formats

CSV_COLUMNS = (
    "id",
    "name",
    "region",
    "period",
    "start_date",
    "end_date",
    "rain_mm",
    "p10_mm",
    "p50_mm",
    "p90_mm",
    "rainy_days",
    "dry_spell_days",
    "wet_spell_days",
    "extreme_days",
    "extreme_chance",
    "dry_spell_chance",
    "wet_spell_chance",
    "onset_date",
    "onset_probability",
)


def summary_csv(payload: dict[str, Any]) -> str:
    """One row per area and period (each week, then the whole run), with the run in a comment header."""
    meta = payload["meta"]
    buffer = io.StringIO()
    buffer.write(f"# {meta['model_label'] or meta['model']} run {meta['run_id']}, issued {meta['issued']}. {ATTRIBUTION}.\n")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in payload["data"]:
        onset = row.get("onset") or {}
        base = {
            "id": row["id"],
            "name": row["name"],
            "region": row["region"],
            "onset_date": onset.get("date"),
            "onset_probability": onset.get("probability"),
        }
        for week in row["weeks"]:
            writer.writerow({**base, **week, "period": f"week {week['week']}"})
        total = row["total"]
        writer.writerow(
            {
                **base,
                **total,
                "period": "whole run",
                "start_date": meta["valid_from"],
                "end_date": meta["valid_to"],
                "extreme_chance": None,
            }
        )
    return buffer.getvalue()


def summary_geojson(settings: Settings, payload: dict[str, Any], level: str) -> dict[str, Any]:
    """Districts as GeoJSON features carrying their summary (whole run plus week 1 and 2, flattened)."""
    if level != "district":
        raise InvalidSubseasonalLayerError("GeoJSON is available for districts only (level=district).")
    rows = {row["id"]: row for row in payload["data"]}
    features = []
    for feature in _district_features(str(settings.seasonal_map.district_geojson_path)):
        properties = feature.get("properties") or {}
        row = rows.get(summaries.area_id(properties.get("display_name", ""), properties.get("region")))
        if row is None:
            continue
        flat: dict[str, Any] = {"id": row["id"], "name": row["name"], "region": row["region"]}
        flat.update({f"total_{key}": value for key, value in row["total"].items() if not isinstance(value, dict)})
        for week in row["weeks"][:2]:
            prefix = f"week{week['week']}_"
            flat.update({prefix + key: value for key, value in week.items() if not isinstance(value, dict) and key != "week"})
        onset = row.get("onset") or {}
        flat["onset_date"] = onset.get("date")
        flat["onset_probability"] = onset.get("probability")
        features.append({"type": "Feature", "geometry": feature.get("geometry"), "properties": flat})
    return {"type": "FeatureCollection", "meta": payload["meta"], "features": features}


_GEOJSON_CACHE: dict[str, list[dict[str, Any]]] = {}


def _district_features(path: str) -> list[dict[str, Any]]:
    if path not in _GEOJSON_CACHE:
        _GEOJSON_CACHE[path] = json.loads(Path(path).read_text(encoding="utf-8")).get("features", [])
    return _GEOJSON_CACHE[path]
