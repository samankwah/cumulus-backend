"""Spatial extraction utilities."""

from __future__ import annotations

import pandas as pd
import xarray as xr


def extract_point_timeseries(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    variables: list[str],
    method: str = "nearest",
) -> pd.DataFrame:
    subset = ds[variables].sel(latitude=lat, longitude=lon, method=method)
    frame = subset.to_dataframe().reset_index()
    if "latitude" not in frame:
        frame["latitude"] = lat
    if "longitude" not in frame:
        frame["longitude"] = lon
    return frame


def extract_locations(
    ds: xr.Dataset,
    locations_df: pd.DataFrame,
    variables: list[str],
    method: str = "nearest",
) -> pd.DataFrame:
    if locations_df.empty:
        return pd.DataFrame(columns=["time", "location_id", "latitude", "longitude", *variables])

    subset = ds[variables].sel(
        latitude=xr.DataArray(locations_df["latitude"].to_numpy(), dims="points"),
        longitude=xr.DataArray(locations_df["longitude"].to_numpy(), dims="points"),
        method=method,
    )
    subset = subset.assign_coords(
        location_id=("points", locations_df["location_id"].astype(str).to_numpy()),
        requested_latitude=("points", locations_df["latitude"].to_numpy()),
        requested_longitude=("points", locations_df["longitude"].to_numpy()),
    )
    result = subset.to_dataframe().reset_index()
    if "points" in result.columns:
        result = result.drop(columns=["points"])
    result["time"] = pd.to_datetime(result["time"], utc=True)
    return result


def extract_station_points(
    ds: xr.Dataset,
    stations_df: pd.DataFrame,
    variables: list[str],
    method: str = "nearest",
) -> pd.DataFrame:
    locations = stations_df.rename(columns={"station_id": "location_id"})[["location_id", "latitude", "longitude"]].drop_duplicates()
    extracted = extract_locations(ds, locations, variables, method=method)
    extracted["station_id"] = extracted["location_id"]
    return extracted
