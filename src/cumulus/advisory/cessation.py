"""Cessation date estimation."""

from __future__ import annotations

import pandas as pd


def estimate_cessation_date(
    frame: pd.DataFrame,
    cessation_window_days: int,
    cessation_threshold_mm: float,
) -> pd.Timestamp | None:
    rainfall = frame["rainfall_corrected_mm"].astype(float).reset_index(drop=True)
    times = frame["time"].reset_index(drop=True)
    for start in range(max(0, len(frame) // 2), max(0, len(frame) - cessation_window_days + 1)):
        window_total = rainfall.iloc[start : start + cessation_window_days].sum()
        if window_total <= cessation_threshold_mm:
            return pd.Timestamp(times.iloc[start])
    return None
