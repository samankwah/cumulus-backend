"""Sub-seasonal (46-day) rainfall endpoints, for every model in the model picker."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from cumulus.schemas import (
    SubseasonalAreaValuesResponse,
    SubseasonalLayerResponse,
    SubseasonalModelsResponse,
    SubseasonalRunsResponse,
    SubseasonalSeriesResponse,
)
from cumulus.services.subseasonal_service import (
    get_area_values,
    get_layer,
    list_models,
    list_runs,
    model_settings,
    render_tile,
    sample_area,
    sample_point,
)
from cumulus.settings import get_settings

router = APIRouter(prefix="/subseasonal", tags=["subseasonal"])

# Tiles are addressed by run_id, so a given URL never changes content: let browsers and the CDN
# keep them, which makes timeline playback instant after the first pass. Vercel's CDN only stores a
# function response that carries s-maxage; max-age alone is kept by the browser only.
IMMUTABLE_CACHE = "public, max-age=31536000, s-maxage=31536000, immutable"
SHORT_CACHE = "public, max-age=300, s-maxage=300"
LISTING_CACHE = "public, max-age=60, s-maxage=60"

# Shared query parameters: which model, and (ensemble rainfall) which statistic of its members.
MODEL_QUERY = Query(default=None, description="Model id from /subseasonal/models; default: the configured model.")
STAT_QUERY = Query(default=None, description="Ensemble rainfall only: mean, median or chance.")
THRESHOLD_QUERY = Query(default=None, ge=0, description="With stat=chance: the amount (mm) members must reach.")
SPELL_DAYS_QUERY = Query(default=None, ge=1, description="Spell layers: the spell length, at least this many days in a row.")
DRY_SPELL_DAYS_QUERY = Query(default=None, ge=1, description="Series: the dry spell length to use (default: the configured one).")
WET_SPELL_DAYS_QUERY = Query(default=None, ge=1, description="Series: the wet spell length to use (default: the configured one).")


@router.get("/models", response_model=SubseasonalModelsResponse)
def subseasonal_models_endpoint(response: Response) -> SubseasonalModelsResponse:
    response.headers["Cache-Control"] = LISTING_CACHE
    return SubseasonalModelsResponse(**list_models(get_settings()))


@router.get("/runs", response_model=SubseasonalRunsResponse)
def subseasonal_runs_endpoint(response: Response, model: str | None = MODEL_QUERY) -> SubseasonalRunsResponse:
    response.headers["Cache-Control"] = LISTING_CACHE
    return SubseasonalRunsResponse(**list_runs(model_settings(get_settings(), model)))


@router.get("/layer", response_model=SubseasonalLayerResponse)
def subseasonal_layer_endpoint(
    response: Response,
    layer: str = Query(default="rainfall"),
    aggregation: str | None = Query(default=None),
    index: int | None = Query(default=None, ge=1),
    run_id: str | None = Query(default=None),
    model: str | None = MODEL_QUERY,
    stat: str | None = STAT_QUERY,
    threshold: float | None = THRESHOLD_QUERY,
    spell_days: int | None = SPELL_DAYS_QUERY,
) -> SubseasonalLayerResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = get_layer(
        model_settings(get_settings(), model),
        layer=layer,
        aggregation=aggregation,
        index=index,
        run_id=run_id,
        stat=stat,
        threshold=threshold,
        spell_days=spell_days,
    )
    return SubseasonalLayerResponse(**payload)


@router.get("/tiles/{z}/{x}/{y}.png")
def subseasonal_tile_endpoint(
    z: int,
    x: int,
    y: int,
    layer: str = Query(default="rainfall"),
    aggregation: str | None = Query(default=None),
    index: int | None = Query(default=None, ge=1),
    run_id: str | None = Query(default=None),
    model: str | None = MODEL_QUERY,
    stat: str | None = STAT_QUERY,
    threshold: float | None = THRESHOLD_QUERY,
    spell_days: int | None = SPELL_DAYS_QUERY,
) -> Response:
    png_bytes = render_tile(
        model_settings(get_settings(), model),
        z=z,
        x=x,
        y=y,
        layer=layer,
        aggregation=aggregation,
        index=index,
        run_id=run_id,
        stat=stat,
        threshold=threshold,
        spell_days=spell_days,
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": IMMUTABLE_CACHE if run_id else SHORT_CACHE},
    )


@router.get("/area-values", response_model=SubseasonalAreaValuesResponse)
def subseasonal_area_values_endpoint(
    response: Response,
    level: str = Query(default="region"),
    layer: str = Query(default="rainfall"),
    aggregation: str | None = Query(default=None),
    index: int | None = Query(default=None, ge=1),
    run_id: str | None = Query(default=None),
    model: str | None = MODEL_QUERY,
    stat: str | None = STAT_QUERY,
    threshold: float | None = THRESHOLD_QUERY,
    spell_days: int | None = SPELL_DAYS_QUERY,
) -> SubseasonalAreaValuesResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = get_area_values(
        model_settings(get_settings(), model),
        level=level,
        layer=layer,
        aggregation=aggregation,
        index=index,
        run_id=run_id,
        stat=stat,
        threshold=threshold,
        spell_days=spell_days,
    )
    return SubseasonalAreaValuesResponse(**payload)


@router.get("/sample", response_model=SubseasonalSeriesResponse)
def subseasonal_sample_endpoint(
    response: Response,
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    run_id: str | None = Query(default=None),
    model: str | None = MODEL_QUERY,
    dry_spell_days: int | None = DRY_SPELL_DAYS_QUERY,
    wet_spell_days: int | None = WET_SPELL_DAYS_QUERY,
) -> SubseasonalSeriesResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = sample_point(
        model_settings(get_settings(), model),
        latitude=latitude,
        longitude=longitude,
        run_id=run_id,
        dry_spell_days=dry_spell_days,
        wet_spell_days=wet_spell_days,
    )
    return SubseasonalSeriesResponse(**payload)


@router.get("/area", response_model=SubseasonalSeriesResponse)
def subseasonal_area_endpoint(
    response: Response,
    level: str = Query(...),
    name: str = Query(..., min_length=1),
    run_id: str | None = Query(default=None),
    model: str | None = MODEL_QUERY,
    dry_spell_days: int | None = DRY_SPELL_DAYS_QUERY,
    wet_spell_days: int | None = WET_SPELL_DAYS_QUERY,
) -> SubseasonalSeriesResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = sample_area(
        model_settings(get_settings(), model),
        level=level,
        name=name,
        run_id=run_id,
        dry_spell_days=dry_spell_days,
        wet_spell_days=wet_spell_days,
    )
    return SubseasonalSeriesResponse(**payload)
