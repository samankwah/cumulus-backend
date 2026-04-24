"""Farmer advisory endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from cumulus.advisory.farmer_rules import build_farmer_advisory
from cumulus.frontend_contract.serializers import serialize_farmer_advisory
from cumulus.schemas import FarmerAdvisoryRequest, FarmerAdvisoryResponse
from cumulus.settings import get_settings

router = APIRouter(tags=["farmer-advisory"])


@router.post("/farmer-advisory", response_model=FarmerAdvisoryResponse)
def farmer_advisory_endpoint(request: FarmerAdvisoryRequest) -> FarmerAdvisoryResponse:
    frame = pd.DataFrame(
        [
            {
                "time": item.date,
                "rainfall_corrected_mm": item.rainfall_mm,
                "temp_c": item.temperature_c,
            }
            for item in request.daily_forecast
        ]
    )
    payload = build_farmer_advisory(frame, get_settings().advisory)
    payload["location_id"] = request.location_id
    return serialize_farmer_advisory(payload)
