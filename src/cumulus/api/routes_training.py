"""Training endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cumulus.schemas import TrainRequest, TrainResponse
from cumulus.services.training_service import train_from_inputs
from cumulus.settings import get_settings

router = APIRouter(tags=["training"])


@router.post("/train", response_model=TrainResponse)
def train_endpoint(request: TrainRequest) -> TrainResponse:
    try:
        artifact = train_from_inputs(
            get_settings(),
            merged_dataset_path=request.merged_dataset_path,
            station_path=request.station_path,
            forecast_path=request.forecast_path,
            forecast_source=request.forecast_source,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TrainResponse(
        model_version=str(artifact["model_version"]),
        metrics=artifact["metrics"],
        bias_method=str(artifact["bias_method"]),
        bias_comparison=artifact.get("bias_comparison", {}),
        evaluation_paths=artifact.get("evaluation_paths", {}),
    )
