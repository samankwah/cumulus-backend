"""Cleaning helpers for climate and station tables."""

from __future__ import annotations

import pandas as pd


def normalize_time_column(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def coerce_numeric_columns(df: pd.DataFrame, exclude_columns: set[str] | None = None) -> pd.DataFrame:
    frame = df.copy()
    excluded = exclude_columns or set()
    for column in frame.columns:
        if column in excluded:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def fill_missing_predictors(
    df: pd.DataFrame,
    target_columns: set[str] | None = None,
    group_key: str = "location_id",
) -> pd.DataFrame:
    frame = df.copy()
    targets = target_columns or set()
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    for column in numeric_columns:
        if column in targets:
            continue
        missing_mask = frame[column].isna()
        if not missing_mask.any():
            continue
        frame[f"{column}_missing"] = missing_mask.astype(int)
        if group_key in frame.columns:
            frame[column] = frame.groupby(group_key)[column].transform(_fill_with_group_median)
        non_null = frame[column].dropna()
        if not non_null.empty:
            frame[column] = frame[column].fillna(non_null.median())
        frame[column] = frame[column].fillna(0.0)
    return frame


def _fill_with_group_median(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series
    return series.fillna(non_null.median())


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["time"] = normalize_time_column(frame["time"])
    frame = frame.drop_duplicates(subset=["location_id", "time"])
    frame = coerce_numeric_columns(frame, exclude_columns={"location_id", "station_id", "time", "region"})
    return frame.sort_values(["location_id", "time"]).reset_index(drop=True)
