"""Aggregate advisory logic."""

from __future__ import annotations

import pandas as pd

from cumulus.advisory.cessation import estimate_cessation_date
from cumulus.advisory.cumulative_rainfall import calculate_cumulative_rainfall
from cumulus.advisory.dry_spell import detect_dry_spell
from cumulus.advisory.onset import estimate_onset_date
from cumulus.settings import AdvisoryConfig


def build_advisory(frame: pd.DataFrame, config: AdvisoryConfig) -> dict[str, object]:
    cumulative = calculate_cumulative_rainfall(frame, config.cumulative_windows_days)
    dry_spell = detect_dry_spell(frame, config.dry_day_threshold_mm, config.dry_spell_days)
    onset_date = estimate_onset_date(
        frame,
        config.onset_window_days,
        config.onset_threshold_mm,
        config.onset_guard_days,
        config.onset_guard_dry_days,
        config.dry_day_threshold_mm,
    )
    cessation_date = estimate_cessation_date(
        frame,
        config.cessation_window_days,
        config.cessation_threshold_mm,
    )
    reason = _build_reason(onset_date, dry_spell["dry_spell_risk"], cumulative.get("cum_rain_7d_mm", 0.0))
    return {
        "onset_date": onset_date.date() if onset_date is not None else None,
        "cessation_date": cessation_date.date() if cessation_date is not None else None,
        "dry_spell_risk": bool(dry_spell["dry_spell_risk"]),
        "dry_spell_length_days": int(dry_spell["dry_spell_length_days"]),
        "cum_rain_7d_mm": float(cumulative.get("cum_rain_7d_mm", 0.0)),
        "cum_rain_14d_mm": float(cumulative.get("cum_rain_14d_mm", 0.0)),
        "seasonal_cum_rain_mm": float(cumulative["seasonal_cum_rain_mm"]),
        "reason": reason,
    }


def _build_reason(onset_date: object, dry_spell_risk: bool, cum_rain_7d_mm: float) -> str:
    if dry_spell_risk:
        return "A prolonged dry spell is detected in the forecast window."
    if onset_date is None:
        return f"Seven-day cumulative rainfall of {cum_rain_7d_mm:.1f} mm does not yet meet onset conditions."
    return "Rainfall accumulation meets onset conditions and no severe dry spell is detected."
