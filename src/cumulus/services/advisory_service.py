"""Public farmer advisory orchestration."""

from __future__ import annotations

import logging

from cumulus.advisory.farmer_rules import build_farmer_advisory
from cumulus.api.errors import ForecastSourceReadError, InferenceExecutionError
from cumulus.frontend_contract.serializers import serialize_farmer_advisory
from cumulus.schemas import PointAdvisoryResponse, PointRequest
from cumulus.services.agro_service import build_agro_characteristics
from cumulus.services.prediction_service import predict_for_point
from cumulus.settings import Settings


logger = logging.getLogger(__name__)


def generate_point_advisory(settings: Settings, request: PointRequest) -> PointAdvisoryResponse:
    prediction = predict_for_point(settings, request)
    if prediction.precomputed_advisory is not None:
        return prediction.precomputed_advisory
    forecast_frame = prediction.forecast_frame
    if "temp_c" not in forecast_frame.columns:
        raise ForecastSourceReadError(
            "Configured forecast source does not include temperature data required for advisory generation."
        )

    try:
        advisory_payload = build_farmer_advisory(
            forecast_frame[["time", "rainfall_corrected_mm", "temp_c"]].copy(),
            settings.advisory,
        )
    except Exception as exc:
        raise InferenceExecutionError("Farmer advisory generation failed for the predicted forecast.") from exc

    advisory_payload["location_id"] = prediction.location_id
    serialized = serialize_farmer_advisory(advisory_payload)
    logger.info(
        "advisory.success location_id=%s model_version=%s",
        prediction.location_id,
        prediction.model_version,
    )
    return PointAdvisoryResponse(
        location_id=prediction.location_id,
        latitude=prediction.latitude,
        longitude=prediction.longitude,
        forecast_source=prediction.forecast_source,
        data_origin=prediction.data_origin,
        source_run_id=prediction.source_run_id,
        spatial_resolution_km=prediction.spatial_resolution_km,
        model_version=prediction.model_version,
        calibration_version=prediction.calibration_version,
        generated_at=prediction.generated_at,
        agro_characteristics=prediction.agro_characteristics
        or build_agro_characteristics(prediction.forecast_frame, settings),
        planting_recommendation=serialized.planting_recommendation,
        dry_spell_alert=serialized.dry_spell_alert,
        irrigation_advice=serialized.irrigation_advice,
    )
