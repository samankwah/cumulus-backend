"""Advisory endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from cumulus.advisory.rules import build_advisory
from cumulus.frontend_contract.serializers import serialize_advisory
from cumulus.schemas import LegacyAdvisoryRequest, PointAdvisoryResponse, PointRequest, SeasonalAdvisoryResponse
from cumulus.services.advisory_service import generate_point_advisory
from cumulus.settings import get_settings

router = APIRouter(tags=["advisory"])


@router.post("/advisory", response_model=PointAdvisoryResponse)
def advisory_endpoint(request: PointRequest) -> PointAdvisoryResponse:
    return generate_point_advisory(get_settings(), request)


@router.post("/advisory/legacy", response_model=SeasonalAdvisoryResponse)
def legacy_advisory_endpoint(request: LegacyAdvisoryRequest) -> SeasonalAdvisoryResponse:
    frame = pd.DataFrame([{"time": item.date, "rainfall_corrected_mm": item.rainfall_mm} for item in request.rainfall_series])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return serialize_advisory(build_advisory(frame, get_settings().advisory))
