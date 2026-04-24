"""Dry spell detection."""

from __future__ import annotations

import pandas as pd


def detect_dry_spell(
    frame: pd.DataFrame,
    dry_day_threshold_mm: float,
    dry_spell_days: int,
) -> dict[str, object]:
    rainfall = frame["rainfall_corrected_mm"].astype(float).tolist()
    max_run = 0
    current_run = 0
    start_index = None
    first_start = None
    for index, value in enumerate(rainfall):
        if value < dry_day_threshold_mm:
            current_run += 1
            if start_index is None:
                start_index = index
            if current_run > max_run:
                max_run = current_run
                first_start = start_index
        else:
            current_run = 0
            start_index = None
    return {
        "dry_spell_risk": max_run >= dry_spell_days,
        "dry_spell_length_days": int(max_run),
        "dry_spell_start": frame.iloc[first_start]["time"].date() if first_start is not None else None,
    }
