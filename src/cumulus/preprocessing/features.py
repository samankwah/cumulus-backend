"""Feature engineering for training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_features(
    df: pd.DataFrame,
    history_column: str = "rainfall_mm",
) -> pd.DataFrame:
    frame = df.copy().sort_values(["location_id", "time"]).reset_index(drop=True)
    frame["month"] = frame["time"].dt.month
    frame["day_of_year"] = frame["time"].dt.dayofyear
    frame["week_of_year"] = frame["time"].dt.isocalendar().week.astype(int)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)

    if {"u10", "v10"}.issubset(frame.columns):
        frame["wind_speed"] = np.sqrt(frame["u10"] ** 2 + frame["v10"] ** 2)
    else:
        frame["wind_speed"] = 0.0

    if "temp_c" not in frame.columns:
        frame["temp_c"] = 0.0
    if "precip_mm" not in frame.columns:
        frame["precip_mm"] = 0.0

    grouped = frame.groupby("location_id", group_keys=False)
    history = grouped[history_column].shift(1) if history_column in frame.columns else grouped["precip_mm"].shift(1)

    frame["rain_lag_1"] = history
    frame["rain_lag_3"] = grouped[history_column].shift(3) if history_column in frame.columns else grouped["precip_mm"].shift(3)
    frame["rain_lag_7"] = grouped[history_column].shift(7) if history_column in frame.columns else grouped["precip_mm"].shift(7)
    frame["rain_roll_3"] = history.groupby(frame["location_id"]).rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    frame["rain_roll_7"] = history.groupby(frame["location_id"]).rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
    frame["rain_roll_14"] = history.groupby(frame["location_id"]).rolling(window=14, min_periods=1).sum().reset_index(level=0, drop=True)

    fill_columns = [
        "rain_lag_1",
        "rain_lag_3",
        "rain_lag_7",
        "rain_roll_3",
        "rain_roll_7",
        "rain_roll_14",
    ]
    frame[fill_columns] = frame[fill_columns].fillna(0.0)
    frame["temp_c"] = frame["temp_c"].fillna(0.0)
    frame["precip_mm"] = frame["precip_mm"].fillna(0.0)
    frame["wind_speed"] = frame["wind_speed"].fillna(0.0)
    return frame
