"""Forecast and observation alignment."""

from __future__ import annotations

import pandas as pd

from cumulus.preprocessing.cleaning import coerce_numeric_columns, fill_missing_predictors, normalize_time_column


def align_forecast_and_station_daily(
    forecast_df: pd.DataFrame,
    station_df: pd.DataFrame,
    location_key: str = "location_id",
    daily_frequency: str = "D",
) -> pd.DataFrame:
    forecast_daily = aggregate_forecast_daily(forecast_df, location_key=location_key, daily_frequency=daily_frequency)
    station_daily = aggregate_station_daily(station_df, location_key=location_key, daily_frequency=daily_frequency)

    merged = forecast_daily.merge(
        station_daily,
        on=[location_key, "time"],
        how="outer",
        suffixes=("_forecast", "_obs"),
    )
    merged["has_forecast"] = merged["forecast_record_count"].fillna(0).gt(0)
    merged["has_target"] = merged["rainfall_mm"].notna()

    if "station_id_forecast" in merged.columns or "station_id_obs" in merged.columns:
        forecast_station_id = merged["station_id_forecast"] if "station_id_forecast" in merged.columns else pd.Series([pd.NA] * len(merged))
        observed_station_id = merged["station_id_obs"] if "station_id_obs" in merged.columns else pd.Series([pd.NA] * len(merged))
        merged["station_id"] = forecast_station_id.combine_first(observed_station_id)
    elif "station_id" not in merged.columns:
        merged["station_id"] = merged[location_key]

    for coordinate in ["latitude", "longitude"]:
        left_column = f"{coordinate}_forecast"
        right_column = f"{coordinate}_obs"
        if left_column in merged.columns or right_column in merged.columns:
            left_series = merged[left_column] if left_column in merged.columns else pd.Series([pd.NA] * len(merged), index=merged.index)
            right_series = merged[right_column] if right_column in merged.columns else pd.Series([pd.NA] * len(merged), index=merged.index)
            merged[coordinate] = left_series.combine_first(right_series)

    merged = merged.sort_values([location_key, "time"]).reset_index(drop=True)
    return merged


def build_ml_ready_dataset(
    aligned_df: pd.DataFrame,
    target_column: str = "rainfall_mm",
    location_key: str = "location_id",
) -> pd.DataFrame:
    dataset = aligned_df.copy()
    dataset = dataset[dataset["has_forecast"].fillna(False)].copy()
    dataset = dataset[dataset[target_column].notna()].copy()
    dataset = coerce_numeric_columns(dataset, exclude_columns={location_key, "station_id", "time"})
    dataset = fill_missing_predictors(dataset, target_columns={target_column}, group_key=location_key)
    return dataset.sort_values([location_key, "time"]).reset_index(drop=True)


def merge_forecast_and_station(
    forecast_df: pd.DataFrame,
    station_df: pd.DataFrame,
    location_key: str = "location_id",
) -> pd.DataFrame:
    aligned = align_forecast_and_station_daily(forecast_df, station_df, location_key=location_key)
    return build_ml_ready_dataset(aligned, location_key=location_key)


def aggregate_forecast_daily(
    forecast_df: pd.DataFrame,
    location_key: str = "location_id",
    daily_frequency: str = "D",
) -> pd.DataFrame:
    frame = forecast_df.copy()
    frame["time"] = normalize_time_column(frame["time"]).dt.floor(daily_frequency)
    if "station_id" not in frame.columns:
        frame["station_id"] = frame[location_key].astype(str)

    aggregation: dict[str, str] = {}
    for column in ["latitude", "longitude", "requested_latitude", "requested_longitude", "station_id"]:
        if column in frame.columns:
            aggregation[column] = "first"

    if "precip_mm" in frame.columns:
        aggregation["precip_mm"] = "sum"
    if "temp_c" in frame.columns:
        aggregation["temp_c"] = "mean"
        aggregation["temp_min_c"] = None
        aggregation["temp_max_c"] = None
    for column in ["u10", "v10"]:
        if column in frame.columns:
            aggregation[column] = "mean"

    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        if column in aggregation or column in {"temp_min_c", "temp_max_c"}:
            continue
        aggregation[column] = "mean"

    group_columns = [location_key, "time"]
    grouped = frame.groupby(group_columns, dropna=False)
    result = grouped.agg({key: value for key, value in aggregation.items() if value is not None}).reset_index()
    if "temp_c" in frame.columns:
        temperature_stats = grouped["temp_c"].agg(temp_min_c="min", temp_max_c="max").reset_index()
        result = result.merge(temperature_stats, on=group_columns, how="left")
    result["forecast_record_count"] = grouped.size().to_numpy()
    return result


def aggregate_station_daily(
    station_df: pd.DataFrame,
    location_key: str = "location_id",
    daily_frequency: str = "D",
) -> pd.DataFrame:
    frame = station_df.copy()
    frame["time"] = normalize_time_column(frame["time"]).dt.floor(daily_frequency)
    if location_key not in frame.columns:
        frame[location_key] = frame.get("station_id")
    frame["station_id"] = frame["station_id"].astype(str)

    aggregation: dict[str, str] = {"station_id": "first"}
    for column in ["latitude", "longitude"]:
        if column in frame.columns:
            aggregation[column] = "first"
    if "rainfall_mm" in frame.columns:
        aggregation["rainfall_mm"] = "sum"
    if "temp_c" in frame.columns:
        aggregation["obs_temp_mean_c"] = None

    group_columns = [location_key, "time"]
    grouped = frame.groupby(group_columns, dropna=False)
    result = grouped.agg({key: value for key, value in aggregation.items() if value is not None}).reset_index()
    if "temp_c" in frame.columns:
        temperature_stats = grouped["temp_c"].agg(
            obs_temp_min_c="min",
            obs_temp_max_c="max",
            obs_temp_mean_c="mean",
        ).reset_index()
        result = result.merge(temperature_stats, on=group_columns, how="left")
    result["station_record_count"] = grouped.size().to_numpy()
    return result
