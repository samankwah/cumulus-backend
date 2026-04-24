"""Agrocharacteristic derivation from normalized forecast outputs."""

from __future__ import annotations

from cumulus.advisory.farmer_rules import build_farmer_advisory
from cumulus.advisory.rules import build_advisory
from cumulus.schemas import AgroCharacteristicsResponse
from cumulus.settings import Settings


def build_agro_characteristics(forecast_frame, settings: Settings) -> AgroCharacteristicsResponse:
    seasonal = build_advisory(
        forecast_frame[["time", "rainfall_corrected_mm"]].copy(),
        settings.advisory,
    )
    farmer = build_farmer_advisory(
        forecast_frame[["time", "rainfall_corrected_mm", "temp_c"]].copy(),
        settings.advisory,
    )
    return AgroCharacteristicsResponse(
        planting_window_signal=str(farmer["planting_recommendation"]["level"]),
        dry_spell_risk=bool(seasonal["dry_spell_risk"]),
        dry_spell_length_days=int(seasonal["dry_spell_length_days"]),
        irrigation_need_signal=str(farmer["irrigation_advice"]["level"]),
        irrigation_deficit_mm=float(farmer["irrigation_advice"].get("rainfall_deficit_mm") or 0.0),
        onset_date=seasonal["onset_date"],
        cessation_date=seasonal["cessation_date"],
        cum_rain_7d_mm=float(seasonal["cum_rain_7d_mm"]),
        cum_rain_14d_mm=float(seasonal["cum_rain_14d_mm"]),
        seasonal_cum_rain_mm=float(seasonal["seasonal_cum_rain_mm"]),
    )
