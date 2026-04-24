"""Cumulative rainfall calculations."""

from __future__ import annotations

import pandas as pd


def calculate_cumulative_rainfall(frame: pd.DataFrame, windows: list[int]) -> dict[str, float]:
    rainfall = frame["rainfall_corrected_mm"].astype(float)
    payload: dict[str, float] = {"seasonal_cum_rain_mm": float(rainfall.sum())}
    for window in windows:
        payload[f"cum_rain_{window}d_mm"] = float(rainfall.head(window).sum())
    return payload
