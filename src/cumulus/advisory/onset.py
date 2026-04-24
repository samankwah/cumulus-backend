"""Onset date estimation."""

from __future__ import annotations

import pandas as pd


def estimate_onset_date(
    frame: pd.DataFrame,
    onset_window_days: int,
    onset_threshold_mm: float,
    onset_guard_days: int,
    onset_guard_dry_days: int,
    dry_day_threshold_mm: float,
) -> pd.Timestamp | None:
    rainfall = frame["rainfall_corrected_mm"].astype(float).reset_index(drop=True)
    times = frame["time"].reset_index(drop=True)
    for start in range(0, max(0, len(frame) - onset_window_days + 1)):
        window_total = rainfall.iloc[start : start + onset_window_days].sum()
        if window_total < onset_threshold_mm:
            continue
        guard = rainfall.iloc[start : start + onset_guard_days]
        dry_run = 0
        max_dry_run = 0
        for value in guard:
            if value < dry_day_threshold_mm:
                dry_run += 1
                max_dry_run = max(max_dry_run, dry_run)
            else:
                dry_run = 0
        if max_dry_run < onset_guard_dry_days:
            return pd.Timestamp(times.iloc[start])
    return None
