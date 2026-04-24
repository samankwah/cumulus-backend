"""Frontend-facing serializers."""

from __future__ import annotations

import pandas as pd

from cumulus.schemas import DailyForecastResponse, FarmerAdvisoryResponse, SeasonalAdvisoryResponse


def serialize_daily_forecast(df: pd.DataFrame) -> list[DailyForecastResponse]:
    return [
        DailyForecastResponse(
            date=row["time"].date(),
            rainfall_raw_mm=float(row["rainfall_raw_mm"]),
            rainfall_corrected_mm=float(row["rainfall_corrected_mm"]),
            temperature_c=float(row["temp_c"]) if "temp_c" in df.columns and pd.notna(row.get("temp_c")) else None,
            horizon_day=index + 1,
        )
        for index, (_, row) in enumerate(df.iterrows())
    ]


def serialize_advisory(payload: dict[str, object]) -> SeasonalAdvisoryResponse:
    return SeasonalAdvisoryResponse(**payload)


def serialize_farmer_advisory(payload: dict[str, object]) -> FarmerAdvisoryResponse:
    return FarmerAdvisoryResponse(**payload)
