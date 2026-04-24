"""Farmer-focused advisory rules for Ghana v1."""

from __future__ import annotations

import pandas as pd

from cumulus.advisory.dry_spell import detect_dry_spell
from cumulus.settings import AdvisoryConfig, FarmerAdvisoryConfig


def build_farmer_advisory(frame: pd.DataFrame, config: AdvisoryConfig) -> dict[str, object]:
    series = _normalize_frame(frame)
    farmer_config = config.farmer
    dry_spell = _build_dry_spell_alert(series, config, farmer_config)
    return {
        "planting_recommendation": _build_planting_recommendation(series, config, farmer_config),
        "dry_spell_alert": dry_spell,
        "irrigation_advice": _build_irrigation_advice(series, farmer_config),
    }


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"time", "rainfall_corrected_mm", "temp_c"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for farmer advisory: {sorted(missing)}")
    series = frame.copy()
    series["time"] = pd.to_datetime(series["time"], utc=True)
    series = series.sort_values("time").reset_index(drop=True)
    series["rainfall_corrected_mm"] = series["rainfall_corrected_mm"].astype(float)
    series["temp_c"] = series["temp_c"].astype(float)
    return series


def _build_planting_recommendation(
    frame: pd.DataFrame,
    advisory_config: AdvisoryConfig,
    farmer_config: FarmerAdvisoryConfig,
) -> dict[str, object]:
    planting_window = frame.head(farmer_config.planting_window_days)
    guard_window = frame.head(farmer_config.planting_guard_days)
    window_rainfall = float(planting_window["rainfall_corrected_mm"].sum())
    avg_temp = float(planting_window["temp_c"].mean())
    max_guard_dry_run = _longest_dry_run(
        guard_window["rainfall_corrected_mm"].tolist(),
        advisory_config.dry_day_threshold_mm,
    )
    temperature_band = _temperature_band(avg_temp, farmer_config)

    if window_rainfall < farmer_config.planting_rain_threshold_mm:
        return {
            "level": "wait_for_more_rain",
            "headline": "Wait for more rain",
            "recommendation": "Do not plant maize yet; wait for a stronger soaking rain.",
            "reason": (
                f"Only {window_rainfall:.1f} mm is forecast in the next {farmer_config.planting_window_days} days, "
                f"below the {farmer_config.planting_rain_threshold_mm:.1f} mm planting trigger."
            ),
            "window_rainfall_mm": window_rainfall,
            "dry_spell_length_days": max_guard_dry_run,
            "avg_temperature_c": avg_temp,
            "temperature_band": temperature_band,
        }

    if max_guard_dry_run >= farmer_config.planting_guard_dry_days:
        return {
            "level": "delay_due_to_dry_spell_risk",
            "headline": "Delay maize planting",
            "recommendation": "Hold planting until rain resumes and the dry spell risk drops.",
            "reason": (
                f"About {window_rainfall:.1f} mm is forecast in the next {farmer_config.planting_window_days} days, "
                f"but the following outlook includes a {max_guard_dry_run}-day dry spell that could stop germination."
            ),
            "window_rainfall_mm": window_rainfall,
            "dry_spell_length_days": max_guard_dry_run,
            "avg_temperature_c": avg_temp,
            "temperature_band": temperature_band,
        }

    if avg_temp >= farmer_config.high_stress_temperature_c:
        return {
            "level": "wait_for_more_rain",
            "headline": "Wait for cooler planting weather",
            "recommendation": "Give planting a short pause until conditions are less stressful for germination.",
            "reason": (
                f"Rainfall is close to the planting trigger, but average temperature is {avg_temp:.1f} C, "
                "which is in the high-stress range for early maize establishment."
            ),
            "window_rainfall_mm": window_rainfall,
            "dry_spell_length_days": max_guard_dry_run,
            "avg_temperature_c": avg_temp,
            "temperature_band": temperature_band,
        }

    return {
        "level": "plant_now",
        "headline": "Plant maize now",
        "recommendation": "Start planting now if your field is ready.",
        "reason": (
            f"About {window_rainfall:.1f} mm is forecast in the next {farmer_config.planting_window_days} days, "
            "with no long dry spell immediately after planting."
        ),
        "window_rainfall_mm": window_rainfall,
        "dry_spell_length_days": max_guard_dry_run,
        "avg_temperature_c": avg_temp,
        "temperature_band": temperature_band,
    }


def _build_dry_spell_alert(
    frame: pd.DataFrame,
    advisory_config: AdvisoryConfig,
    farmer_config: FarmerAdvisoryConfig,
) -> dict[str, object]:
    detection = detect_dry_spell(frame, advisory_config.dry_day_threshold_mm, farmer_config.dry_spell_warning_days)
    dry_spell_length_days = int(detection["dry_spell_length_days"])
    if dry_spell_length_days >= farmer_config.dry_spell_warning_days:
        return {
            "level": "warning",
            "headline": "Long dry spell likely",
            "recommendation": "Delay planting or protect young crops because a week-long dry spell is likely.",
            "reason": (
                f"The forecast shows up to {dry_spell_length_days} consecutive dry days, "
                "which meets the warning threshold."
            ),
            "dry_spell_length_days": dry_spell_length_days,
        }

    if dry_spell_length_days >= farmer_config.dry_spell_watch_days:
        return {
            "level": "watch",
            "headline": "Dry spell watch",
            "recommendation": "Keep seed on hold and watch for a break in the dry spell.",
            "reason": (
                f"The forecast shows up to {dry_spell_length_days} consecutive dry days, "
                f"which is above the {farmer_config.dry_spell_watch_days}-day watch level."
            ),
            "dry_spell_length_days": dry_spell_length_days,
        }

    return {
        "level": "none",
        "headline": "No dry spell alert",
        "recommendation": "Rainfall should stay frequent enough for early crop growth.",
        "reason": (
            f"The longest run of dry days in the forecast is {dry_spell_length_days} day(s), "
            f"which is below the {farmer_config.dry_spell_watch_days}-day watch level."
        ),
        "dry_spell_length_days": dry_spell_length_days,
    }


def _build_irrigation_advice(frame: pd.DataFrame, farmer_config: FarmerAdvisoryConfig) -> dict[str, object]:
    irrigation_window = frame.head(farmer_config.irrigation_window_days)
    window_rainfall = float(irrigation_window["rainfall_corrected_mm"].sum())
    avg_temp = float(irrigation_window["temp_c"].mean())
    rainfall_deficit = max(0.0, float(farmer_config.irrigation_target_rain_mm - window_rainfall))
    temperature_band = _temperature_band(avg_temp, farmer_config)

    severity = 0
    if rainfall_deficit > 0:
        severity = 1
    if rainfall_deficit >= farmer_config.irrigation_severe_deficit_mm:
        severity = 2
    if rainfall_deficit > 0 and avg_temp >= farmer_config.hot_temperature_c:
        severity = min(2, severity + 1)

    if severity == 0:
        return {
            "level": "no_irrigation_needed",
            "headline": "No irrigation needed now",
            "recommendation": "Recent forecast rain should support maize establishment without extra watering.",
            "reason": (
                f"About {window_rainfall:.1f} mm is forecast over the next {farmer_config.irrigation_window_days} days, "
                "so there is no rainfall deficit to cover."
            ),
            "window_rainfall_mm": window_rainfall,
            "rainfall_deficit_mm": rainfall_deficit,
            "avg_temperature_c": avg_temp,
            "temperature_band": temperature_band,
        }

    if severity == 1:
        return {
            "level": "monitor_soil_moisture",
            "headline": "Watch soil moisture",
            "recommendation": "Check soil moisture closely and be ready to irrigate if the topsoil dries out.",
            "reason": (
                f"Only {window_rainfall:.1f} mm is forecast over the next {farmer_config.irrigation_window_days} days, "
                f"leaving a {rainfall_deficit:.1f} mm rainfall deficit."
            ),
            "window_rainfall_mm": window_rainfall,
            "rainfall_deficit_mm": rainfall_deficit,
            "avg_temperature_c": avg_temp,
            "temperature_band": temperature_band,
        }

    heat_note = ""
    if avg_temp >= farmer_config.hot_temperature_c:
        heat_note = " Hot conditions increase moisture loss."
    return {
        "level": "irrigate_if_possible",
        "headline": "Irrigation is recommended",
        "recommendation": "If you can irrigate, apply water to keep the root zone from drying out.",
        "reason": (
            f"Only {window_rainfall:.1f} mm is forecast over the next {farmer_config.irrigation_window_days} days, "
            f"leaving a {rainfall_deficit:.1f} mm rainfall deficit.{heat_note}"
        ).strip(),
        "window_rainfall_mm": window_rainfall,
        "rainfall_deficit_mm": rainfall_deficit,
        "avg_temperature_c": avg_temp,
        "temperature_band": temperature_band,
    }


def _temperature_band(avg_temp: float, farmer_config: FarmerAdvisoryConfig) -> str:
    if avg_temp >= farmer_config.high_stress_temperature_c:
        return "high_stress"
    if avg_temp >= farmer_config.hot_temperature_c:
        return "hot"
    return "normal"


def _longest_dry_run(rainfall: list[float], dry_day_threshold_mm: float) -> int:
    max_run = 0
    current_run = 0
    for value in rainfall:
        if float(value) < dry_day_threshold_mm:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run
