"""Sub-seasonal (46-day) IFS-UNet rainfall endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from cumulus.schemas import (
    SubseasonalAreaValuesResponse,
    SubseasonalLayerResponse,
    SubseasonalRunsResponse,
    SubseasonalSeriesResponse,
)
from cumulus.services.subseasonal_service import (
    get_area_values,
    get_layer,
    list_runs,
    render_tile,
    sample_area,
    sample_point,
)
from cumulus.settings import get_settings

router = APIRouter(prefix="/subseasonal", tags=["subseasonal"])

# Tiles are addressed by run_id, so a given URL never changes content: let browsers and the CDN
# keep them, which makes timeline playback instant after the first pass.
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
SHORT_CACHE = "public, max-age=300"


@router.get("/runs", response_model=SubseasonalRunsResponse)
def subseasonal_runs_endpoint(response: Response) -> SubseasonalRunsResponse:
    response.headers["Cache-Control"] = "public, max-age=60"
    return SubseasonalRunsResponse(**list_runs(get_settings()))


@router.get("/layer", response_model=SubseasonalLayerResponse)
def subseasonal_layer_endpoint(
    response: Response,
    layer: str = Query(default="rainfall"),
    aggregation: str | None = Query(default=None),
    index: int | None = Query(default=None, ge=1),
    run_id: str | None = Query(default=None),
) -> SubseasonalLayerResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = get_layer(get_settings(), layer=layer, aggregation=aggregation, index=index, run_id=run_id)
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
) -> Response:
    png_bytes = render_tile(
        get_settings(), z=z, x=x, y=y, layer=layer, aggregation=aggregation, index=index, run_id=run_id
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
) -> SubseasonalAreaValuesResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = get_area_values(
        get_settings(), level=level, layer=layer, aggregation=aggregation, index=index, run_id=run_id
    )
    return SubseasonalAreaValuesResponse(**payload)


@router.get("/sample", response_model=SubseasonalSeriesResponse)
def subseasonal_sample_endpoint(
    response: Response,
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    run_id: str | None = Query(default=None),
) -> SubseasonalSeriesResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = sample_point(get_settings(), latitude=latitude, longitude=longitude, run_id=run_id)
    return SubseasonalSeriesResponse(**payload)


@router.get("/area", response_model=SubseasonalSeriesResponse)
def subseasonal_area_endpoint(
    response: Response,
    level: str = Query(...),
    name: str = Query(..., min_length=1),
    run_id: str | None = Query(default=None),
) -> SubseasonalSeriesResponse:
    response.headers["Cache-Control"] = IMMUTABLE_CACHE if run_id else SHORT_CACHE
    payload = sample_area(get_settings(), level=level, name=name, run_id=run_id)
    return SubseasonalSeriesResponse(**payload)
