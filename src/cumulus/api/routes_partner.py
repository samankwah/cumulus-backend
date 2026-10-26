"""The open API, mounted at /v1 with its own docs (/v1/docs): GMet's 46-day forecast, free for
anyone (the public, NADMO, VRA, MoFA, FAO, NCA, insurers...). No key is needed."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from cumulus.api.errors import install_exception_handlers
from cumulus.api.security import identify_caller
from cumulus.services import partner_service
from cumulus.settings import get_settings

DESCRIPTION = """
GMet's 46-day rainfall forecast for partners, by region and district of Ghana.

* **Free and open:** no key or sign-up. Call it from a server, a spreadsheet or a web page (CORS is open).
* **Partners** may send a key from GMet in `X-API-Key` so their use is counted as theirs; it gives the same data.
* **Fair use:** up to 120 calls a minute per address (600 with a partner key); past that you get `429`
  with `Retry-After`. Cached answers do not count, so poll `/v1/runs/latest` with `If-None-Match`.
* **Terms:** credit "Delivered by GMet · forecast models by the Cumulus team" and keep the disclaimer in each response.
* **Runs:** by default every route answers from the newest run of the model GMet marks as official;
  pass `model=` for another model and `run=` for an earlier run (archived runs keep their summary).
* **Updates:** a new run is published a few times a week. Poll `/v1/runs/latest` (it sends an `ETag`
  so an unchanged run costs a `304`), or ask GMet to register a webhook.
* **Figures** are the ones on the GMet map: amounts in mm, chances in % of ensemble members.
* **Places:** a region or district (`/v1/areas`), or any point (`lat`, `lon`): the nearest grid cell.
* **Maps:** `/v1/map` gives a layer's legend and PNG tile URL; `/v1/grid` its values (JSON, CSV or NetCDF).
"""

app = FastAPI(
    title="GMet 46-day forecast · open API",
    version="1",
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url=None,
)
install_exception_handlers(app)

MODEL_QUERY = Query(default=None, description="Model id (see /subseasonal/models); default: GMet's official model.")
RUN_QUERY = Query(default=None, description="Run id (see /v1/runs); default: the newest run.")
LEVEL_QUERY = Query(default="district", description="region or district.")
MEMBERS_QUERY = Query(default=False, description="Include each ensemble member's amounts.")
LAT_QUERY = Query(default=None, ge=-90, le=90, description="Latitude of a point (with lon), instead of an area.")
LON_QUERY = Query(default=None, ge=-180, le=180, description="Longitude of a point (with lat), instead of an area.")
LAYER_QUERY = Query(default="rainfall", description="rainfall, rainy_days, dry_spell_days, wet_spell_days, extreme_rain or onset.")
AGGREGATION_QUERY = Query(default=None, description="total, weekly or daily (default: the layer's first).")
INDEX_QUERY = Query(default=None, ge=1, description="Week or day number (1 = first).")
STAT_QUERY = Query(default=None, description="Rainfall: mean, median or chance. Spells: chance or days.")
THRESHOLD_QUERY = Query(default=None, description="Rainfall chance: the amount (mm), e.g. 10/25/50 a day.")
SPELL_DAYS_QUERY = Query(default=None, ge=1, description="Spells: the spell length in days.")
# Open data: the CDN may keep it, which also shields the server from heavy use. "Latest" answers
# change when a new run is published, so they are kept 5 minutes; a named run never changes.
LATEST_CACHE = "public, max-age=300, s-maxage=300"
RUN_CACHE = "public, max-age=86400, s-maxage=86400"


def _respond(request: Request, payload: dict[str, Any], run: str | None, body: Response | None = None) -> Response:
    """JSON (or the given body) with an ETag for the run, answering 304 when the caller has it."""
    meta = payload.get("meta") or {}
    tag = f'"{meta.get("model")}:{meta.get("run_id")}"' if meta.get("run_id") else None
    headers = {"Cache-Control": RUN_CACHE if run else LATEST_CACHE}
    if tag:
        headers["ETag"] = tag
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers=headers)
    response = body or JSONResponse(jsonable_encoder(payload))
    response.headers.update(headers)
    return response


@app.get("/runs/latest", summary="The newest run")
def runs_latest(request: Request, model: str | None = MODEL_QUERY, _caller: str = Depends(identify_caller("runs"))):
    return _respond(request, partner_service.latest_run(get_settings(), model), None)


@app.get("/runs", summary="Runs you can ask for, newest first")
def runs(
    model: str | None = MODEL_QUERY,
    start: str | None = Query(default=None, alias="from", description="Issued on or after (YYYY-MM-DD)."),
    end: str | None = Query(default=None, alias="to", description="Issued on or before (YYYY-MM-DD)."),
    _caller: str = Depends(identify_caller("runs")),
):
    payload = partner_service.list_partner_runs(get_settings(), model, start, end)
    return JSONResponse(payload, headers={"Cache-Control": LATEST_CACHE})


@app.get("/areas", summary="Region and district ids")
def areas(level: str = LEVEL_QUERY, model: str | None = MODEL_QUERY, _caller: str = Depends(identify_caller("areas"))):
    payload = partner_service.list_areas(get_settings(), model, level)
    return JSONResponse(payload, headers={"Cache-Control": RUN_CACHE})


@app.get("/summary", summary="Whole-run and weekly figures for every area")
def summary(
    request: Request,
    level: str = LEVEL_QUERY,
    model: str | None = MODEL_QUERY,
    run: str | None = RUN_QUERY,
    format: str = Query(default="json", description="json, csv or geojson (districts)."),
    _caller: str = Depends(identify_caller("summary")),
):
    settings = get_settings()
    if format not in partner_service.FORMATS:
        format = "json"
    payload = partner_service.area_summaries(settings, model, run, level)
    if format == "csv":
        meta = payload["meta"]
        body = PlainTextResponse(partner_service.summary_csv(payload), media_type="text/csv")
        body.headers["Content-Disposition"] = f'attachment; filename="{meta["run_id"]}_{level}.csv"'
        return _respond(request, payload, run, body)
    if format == "geojson":
        collection = partner_service.summary_geojson(settings, payload, level)
        return _respond(request, payload, run, JSONResponse(collection, media_type="application/geo+json"))
    return _respond(request, payload, run)


@app.get("/areas/{level}/{area}/forecast", summary="One area's full daily and weekly forecast")
def area_forecast(
    request: Request,
    level: str,
    area: str,
    model: str | None = MODEL_QUERY,
    run: str | None = RUN_QUERY,
    members: bool = MEMBERS_QUERY,
    _caller: str = Depends(identify_caller("forecast")),
):
    return _respond(request, partner_service.area_forecast(get_settings(), model, run, level, area, members), run)


@app.get("/point/forecast", summary="The full daily and weekly forecast at any point")
def point_forecast(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    model: str | None = MODEL_QUERY,
    run: str | None = RUN_QUERY,
    members: bool = MEMBERS_QUERY,
    _caller: str = Depends(identify_caller("forecast")),
):
    return _respond(request, partner_service.point_forecast(get_settings(), model, run, lat, lon, members), run)


@app.get("/rainfall", summary="Rain over any dates, for an area or a point")
def rainfall(
    request: Request,
    start: date = Query(..., description="First day (YYYY-MM-DD), inside the run."),
    end: date = Query(..., description="Last day (YYYY-MM-DD), inside the run."),
    level: str = LEVEL_QUERY,
    area: str | None = Query(default=None, description="Area id or name (see /v1/areas)."),
    lat: float | None = LAT_QUERY,
    lon: float | None = LON_QUERY,
    model: str | None = MODEL_QUERY,
    run: str | None = RUN_QUERY,
    _caller: str = Depends(identify_caller("forecast")),
):
    payload = partner_service.period_rainfall(
        get_settings(), model, run, start, end, level=level, area=area, latitude=lat, longitude=lon
    )
    return _respond(request, payload, run)


@app.get("/map", summary="A map layer: legend, national figures and PNG tile URL")
def map_layer(
    request: Request,
    layer: str = LAYER_QUERY,
    aggregation: str | None = AGGREGATION_QUERY,
    index: int | None = INDEX_QUERY,
    stat: str | None = STAT_QUERY,
    threshold: float | None = THRESHOLD_QUERY,
    spell_days: int | None = SPELL_DAYS_QUERY,
    model: str | None = MODEL_QUERY,
    run: str | None = RUN_QUERY,
    _caller: str = Depends(identify_caller("maps")),
):
    payload = partner_service.map_layer(
        get_settings(),
        model,
        run,
        str(request.base_url).removesuffix("/v1/").removesuffix("/"),
        layer=layer,
        aggregation=aggregation,
        index=index,
        stat=stat,
        threshold=threshold,
        spell_days=spell_days,
    )
    return _respond(request, payload, run)


@app.get("/grid", summary="A map layer's values on the forecast grid, or rain over chosen dates")
def grid(
    request: Request,
    layer: str = LAYER_QUERY,
    aggregation: str | None = AGGREGATION_QUERY,
    index: int | None = INDEX_QUERY,
    stat: str | None = STAT_QUERY,
    threshold: float | None = THRESHOLD_QUERY,
    spell_days: int | None = SPELL_DAYS_QUERY,
    start: date | None = Query(default=None, description="Rain summed from this day (YYYY-MM-DD); overrides layer."),
    end: date | None = Query(default=None, description="Rain summed to this day (YYYY-MM-DD); overrides layer."),
    model: str | None = MODEL_QUERY,
    run: str | None = RUN_QUERY,
    format: str = Query(default="json", description="json, csv or netcdf."),
    _caller: str = Depends(identify_caller("maps")),
):
    payload = partner_service.grid(
        get_settings(),
        model,
        run,
        start=start,
        end=end,
        layer=layer,
        aggregation=aggregation,
        index=index,
        stat=stat,
        threshold=threshold,
        spell_days=spell_days,
    )
    if format not in partner_service.GRID_FORMATS:
        format = "json"
    name = f"{payload['meta']['run_id']}_{payload['data']['layer']}_{payload['data']['start_date']}_{payload['data']['end_date']}"
    if format == "csv":
        body = PlainTextResponse(partner_service.grid_csv(payload), media_type="text/csv")
        body.headers["Content-Disposition"] = f'attachment; filename="{name}.csv"'
        return _respond(request, payload, run, body)
    if format == "netcdf":
        body = Response(partner_service.grid_netcdf(payload), media_type="application/x-netcdf")
        body.headers["Content-Disposition"] = f'attachment; filename="{name}.nc"'
        return _respond(request, payload, run, body)
    return _respond(request, payload, run)


@app.get("/alerts", summary="Area-weeks at or above chosen chances")
def alerts(
    request: Request,
    level: str = LEVEL_QUERY,
    model: str | None = MODEL_QUERY,
    run: str | None = RUN_QUERY,
    extreme_chance: float = Query(default=60, ge=0, le=100, description="Chance (%) of a 50 mm+ day in the week."),
    dry_spell_chance: float = Query(default=60, ge=0, le=100, description="Chance (%) of a dry spell touching the week."),
    wet_spell_chance: float | None = Query(default=None, ge=0, le=100, description="Optional: chance (%) of a wet spell."),
    _caller: str = Depends(identify_caller("alerts")),
):
    payload = partner_service.alerts(get_settings(), model, run, level, extreme_chance, dry_spell_chance, wet_spell_chance)
    return _respond(request, payload, run)
