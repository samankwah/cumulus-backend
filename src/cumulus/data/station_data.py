"""Station observation loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cumulus.preprocessing.cleaning import coerce_numeric_columns, normalize_time_column


COLUMN_ALIASES = {
    "station": "station_id",
    "stationid": "station_id",
    "id": "station_id",
    "datetime": "time",
    "timestamp": "time",
    "date": "time",
    "year": "year",
    "month": "month",
    "lat": "latitude",
    "geogr2": "latitude",
    "lon": "longitude",
    "geogr1": "longitude",
    "rain": "rainfall_mm",
    "rainfall": "rainfall_mm",
    "precip": "rainfall_mm",
    "temp": "temp_c",
    "temperature": "temp_c",
    "tmean": "temp_c",
    "name": "name",
    "datatype": "data_type",
    "elementid": "element_id",
}

REQUIRED_COLUMNS = {"station_id", "time", "latitude", "longitude", "rainfall_mm"}


def load_station_observations(path: str | Path) -> pd.DataFrame:
    station_path = Path(path)
    suffix = station_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(station_path)
    elif suffix in {".xlsx", ".xls"}:
        df = load_excel_station_observations(station_path)
    else:
        df = pd.read_csv(station_path)
    df = normalize_station_columns(df)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Station dataset missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["station_id", "time", "latitude", "longitude"]).copy()
    df["time"] = normalize_time_column(df["time"])
    df["station_id"] = df["station_id"].astype(str)
    df = coerce_numeric_columns(df, exclude_columns={"station_id", "time"})
    if "rainfall_mm" in df.columns:
        df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0.0)
    return (
        df.sort_values(["station_id", "time"])
        .drop_duplicates(subset=["station_id", "time", "latitude", "longitude"], keep="last")
        .reset_index(drop=True)
    )


def load_excel_station_observations(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    normalized = normalize_station_columns(raw)
    day_columns = [column for column in normalized.columns if _normalize_day_column(column) is not None]
    if not {"station_id", "year", "month", "longitude", "latitude"}.issubset(normalized.columns) or not day_columns:
        return normalized

    value_name = "rainfall_mm" if "elementid" not in {_sanitize_column_name(column) for column in raw.columns} else "observed_value"
    melted = normalized.melt(
        id_vars=[column for column in normalized.columns if column not in day_columns],
        value_vars=day_columns,
        var_name="day",
        value_name=value_name,
    )
    melted["day"] = melted["day"].map(_normalize_day_column)
    melted = melted.dropna(subset=["year", "month", "day"])
    melted["day"] = melted["day"].astype(int)
    melted["year"] = pd.to_numeric(melted["year"], errors="coerce")
    melted["month"] = pd.to_numeric(melted["month"], errors="coerce")
    melted = melted.dropna(subset=["year", "month"])
    melted["year"] = melted["year"].astype(int)
    melted["month"] = melted["month"].astype(int)
    melted["base_date"] = pd.to_datetime(
        {
            "year": melted["year"],
            "month": melted["month"],
            "day": melted["day"],
        },
        errors="coerce",
    )
    melted = melted.dropna(subset=["base_date"])

    if "time" in melted.columns:
        time_strings = melted["time"].astype(str).replace({"NaT": "00:00:00", "nan": "00:00:00"})
    else:
        time_strings = pd.Series(["00:00:00"] * len(melted), index=melted.index)
    melted["time"] = pd.to_datetime(
        melted["base_date"].dt.strftime("%Y-%m-%d") + " " + time_strings,
        errors="coerce",
        utc=True,
    )
    melted = melted.dropna(subset=["time"])

    if value_name == "observed_value":
        melted["element_id"] = melted.get("element_id", "").astype(str).str.upper()
        rainfall_mask = melted["element_id"].eq("RR")
        melted["rainfall_mm"] = pd.to_numeric(melted["observed_value"], errors="coerce").where(rainfall_mask)
        if "temp_c" not in melted.columns:
            melted["temp_c"] = pd.NA
    else:
        melted["rainfall_mm"] = pd.to_numeric(melted["rainfall_mm"], errors="coerce")

    keep_columns = [
        column
        for column in [
            "station_id",
            "time",
            "longitude",
            "latitude",
            "rainfall_mm",
            "temp_c",
            "name",
            "data_type",
            "element_id",
        ]
        if column in melted.columns
    ]
    return melted[keep_columns]


def normalize_station_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    rename_map: dict[str, str] = {}
    for column in normalized.columns:
        key = _sanitize_column_name(column)
        if key in COLUMN_ALIASES and COLUMN_ALIASES[key] not in normalized.columns:
            rename_map[column] = COLUMN_ALIASES[key]
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    return normalized


def _sanitize_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "").replace("-", "").replace(".", "").replace("_", "")


def _normalize_day_column(column: object) -> int | None:
    text = str(column).strip()
    if not text.isdigit():
        return None
    day = int(text)
    if 1 <= day <= 31:
        return day
    return None
