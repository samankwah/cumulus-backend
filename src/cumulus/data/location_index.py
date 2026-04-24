"""Location catalog utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cumulus.utils.io import load_yaml


def load_locations(path: str | Path) -> pd.DataFrame:
    payload = load_yaml(Path(path))
    locations = payload.get("locations", [])
    frame = pd.DataFrame(locations)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "location_id",
                "latitude",
                "longitude",
                "region",
                "district",
                "agro_ecological_zone",
                "is_serving_location",
            ]
        )

    defaults = {
        "region": None,
        "district": None,
        "agro_ecological_zone": None,
        "is_serving_location": True,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["location_id"] = frame["location_id"].astype(str)
    frame["is_serving_location"] = frame["is_serving_location"].fillna(True).astype(bool)
    return frame


def load_serving_locations(path: str | Path) -> pd.DataFrame:
    frame = load_locations(path)
    if frame.empty:
        return frame
    return frame[frame["is_serving_location"]].reset_index(drop=True)


def find_location_metadata(
    path: str | Path,
    *,
    location_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    tolerance_degrees: float = 0.15,
) -> dict[str, object] | None:
    frame = load_locations(path)
    if frame.empty:
        return None
    if location_id is not None:
        matched = frame[frame["location_id"].astype(str) == str(location_id)]
        if not matched.empty:
            return matched.iloc[0].to_dict()
    if latitude is None or longitude is None:
        return None
    matched = frame[
        frame["latitude"].sub(float(latitude)).abs().le(tolerance_degrees)
        & frame["longitude"].sub(float(longitude)).abs().le(tolerance_degrees)
    ]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()
