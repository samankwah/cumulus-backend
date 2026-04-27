"""Deterministic forecast raster metadata and tile rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import struct
from typing import Any
from urllib.parse import urlencode
import zlib

import numpy as np
import pandas as pd
import xarray as xr

from cumulus.api.errors import InvalidForecastRasterVariableError, InvalidHorizonError
from cumulus.services.source_resolution import open_source_dataset, resolve_forecast_source
from cumulus.settings import Settings

TILE_SIZE = 256


@dataclass(frozen=True)
class RasterVariableSpec:
    variable: str
    label: str
    unit: str
    round_digits: int
    color_ramp: tuple[tuple[float, str], ...]


@dataclass(frozen=True)
class PreparedForecastRaster:
    layer_id: str
    variable: str
    variable_label: str
    unit: str
    round_digits: int
    horizon_day: int
    valid_time: datetime
    generated_at: datetime
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    data_origin: str | None
    lower_bound: float
    upper_bound: float
    available_horizon_days: tuple[int, ...]
    legend_ticks: tuple[float, ...]
    color_ramp: tuple[tuple[float, str], ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    values: np.ndarray


SUPPORTED_FORECAST_RASTER_VARIABLES: dict[str, RasterVariableSpec] = {
    "rainfall_daily_mm": RasterVariableSpec(
        variable="rainfall_daily_mm",
        label="Daily Rainfall",
        unit="mm",
        round_digits=1,
        color_ramp=(
            (0.0, "#2b0a59"),
            (0.16, "#43388d"),
            (0.32, "#3665c8"),
            (0.48, "#238cb5"),
            (0.64, "#2aa27b"),
            (0.8, "#94c93d"),
            (1.0, "#efd84d"),
        ),
    ),
    "temperature_c": RasterVariableSpec(
        variable="temperature_c",
        label="Air Temperature",
        unit="C",
        round_digits=1,
        color_ramp=(
            (0.0, "#2b0a59"),
            (0.16, "#43388d"),
            (0.32, "#3665c8"),
            (0.48, "#238cb5"),
            (0.64, "#2aa27b"),
            (0.8, "#94c93d"),
            (1.0, "#efd84d"),
        ),
    ),
}


def get_forecast_raster_metadata(
    settings: Settings,
    *,
    variable: str = "rainfall_daily_mm",
    horizon_day: int = 1,
    forecast_source: str | None = None,
) -> dict[str, Any]:
    resolved_source = resolve_forecast_source(settings, forecast_source)
    prepared = _prepare_forecast_raster(
        settings,
        variable=variable,
        horizon_day=horizon_day,
        forecast_source=forecast_source,
    )
    query = urlencode(
        {
            "variable": prepared.variable,
            "horizon_day": prepared.horizon_day,
            **({"forecast_source": resolved_source.source_id} if forecast_source or resolved_source.source_id else {}),
        }
    )
    return {
        "layer_id": prepared.layer_id,
        "tile_url": f"/forecast/raster/tiles/{{z}}/{{x}}/{{y}}.png?{query}",
        "variable": prepared.variable,
        "variable_label": prepared.variable_label,
        "unit": prepared.unit,
        "horizon_day": prepared.horizon_day,
        "valid_time": prepared.valid_time,
        "generated_at": prepared.generated_at,
        "forecast_source": prepared.forecast_source,
        "forecast_source_label": prepared.forecast_source_label,
        "source_run_id": prepared.source_run_id,
        "data_origin": prepared.data_origin,
        "lower_bound": prepared.lower_bound,
        "upper_bound": prepared.upper_bound,
        "available_horizon_days": list(prepared.available_horizon_days),
        "legend_ticks": list(prepared.legend_ticks),
        "color_ramp": [{"offset": offset, "color": color} for offset, color in prepared.color_ramp],
        "bounds": {
            "latitude_min": float(prepared.latitudes.min()),
            "latitude_max": float(prepared.latitudes.max()),
            "longitude_min": float(prepared.longitudes.min()),
            "longitude_max": float(prepared.longitudes.max()),
        },
        "grid": {
            "latitudes": [round(float(value), 4) for value in prepared.latitudes.tolist()],
            "longitudes": [round(float(value), 4) for value in prepared.longitudes.tolist()],
            "values": _serialize_grid(prepared.values, prepared.round_digits),
        },
    }


def render_forecast_raster_tile(
    settings: Settings,
    *,
    z: int,
    x: int,
    y: int,
    variable: str = "rainfall_daily_mm",
    horizon_day: int = 1,
    forecast_source: str | None = None,
) -> bytes:
    prepared = _prepare_forecast_raster(
        settings,
        variable=variable,
        horizon_day=horizon_day,
        forecast_source=forecast_source,
    )
    return _render_tile_png(prepared, z=z, x=x, y=y)


def sample_forecast_raster(
    settings: Settings,
    *,
    latitude: float,
    longitude: float,
    variable: str = "rainfall_daily_mm",
    horizon_day: int = 1,
    forecast_source: str | None = None,
) -> dict[str, Any]:
    prepared = _prepare_forecast_raster(
        settings,
        variable=variable,
        horizon_day=horizon_day,
        forecast_source=forecast_source,
    )
    latitude_index = _nearest_axis_index(prepared.latitudes, latitude)
    longitude_index = _nearest_axis_index(prepared.longitudes, longitude)
    nearest_latitude = float(prepared.latitudes[latitude_index]) if prepared.latitudes.size else None
    nearest_longitude = float(prepared.longitudes[longitude_index]) if prepared.longitudes.size else None
    sampled_value = float(prepared.values[latitude_index, longitude_index])
    return {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "nearest_latitude": round(nearest_latitude, 4) if nearest_latitude is not None else None,
        "nearest_longitude": round(nearest_longitude, 4) if nearest_longitude is not None else None,
        "value": round(sampled_value, prepared.round_digits) if math.isfinite(sampled_value) else None,
        "variable": prepared.variable,
        "variable_label": prepared.variable_label,
        "unit": prepared.unit,
        "horizon_day": prepared.horizon_day,
        "valid_time": prepared.valid_time,
        "forecast_source": prepared.forecast_source,
        "forecast_source_label": prepared.forecast_source_label,
        "source_run_id": prepared.source_run_id,
        "data_origin": prepared.data_origin,
    }


def _prepare_forecast_raster(
    settings: Settings,
    *,
    variable: str,
    horizon_day: int,
    forecast_source: str | None,
) -> PreparedForecastRaster:
    spec = _resolve_raster_variable(variable)
    resolved_source = resolve_forecast_source(settings, forecast_source)
    dataset = open_source_dataset(settings, resolved_source)
    field = _select_forecast_field(dataset, spec, horizon_day)
    latitudes = np.asarray(field["latitude"].to_numpy(), dtype=float)
    longitudes = np.asarray(field["longitude"].to_numpy(), dtype=float)
    values = np.asarray(field.to_numpy(), dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise InvalidHorizonError(
            f"Forecast raster variable '{spec.variable}' does not contain any finite values for horizon day {horizon_day}."
        )
    lower_bound = round(float(np.nanmin(finite_values)), spec.round_digits)
    upper_bound = round(float(np.nanmax(finite_values)), spec.round_digits)
    tick_values = tuple(
        round(float(item), spec.round_digits)
        for item in np.linspace(lower_bound, upper_bound, num=5)
    )
    valid_time = _to_utc_datetime(field.coords["time"].item())
    generated_at = datetime.now(UTC)
    layer_id = f"{resolved_source.source_id}_{spec.variable}_day_{horizon_day}_{resolved_source.source_run_id}"
    available_horizon_days = tuple(range(1, int(dataset.sizes.get("time", 0)) + 1))
    return PreparedForecastRaster(
        layer_id=layer_id,
        variable=spec.variable,
        variable_label=spec.label,
        unit=spec.unit,
        round_digits=spec.round_digits,
        horizon_day=horizon_day,
        valid_time=valid_time,
        generated_at=generated_at,
        forecast_source=resolved_source.source_id,
        forecast_source_label=_forecast_source_label(resolved_source.source_id),
        source_run_id=resolved_source.source_run_id,
        data_origin=resolved_source.data_origin,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        available_horizon_days=available_horizon_days,
        legend_ticks=tick_values,
        color_ramp=spec.color_ramp,
        latitudes=latitudes,
        longitudes=longitudes,
        values=values,
    )


def _resolve_raster_variable(variable: str) -> RasterVariableSpec:
    key = str(variable).strip().lower()
    try:
        return SUPPORTED_FORECAST_RASTER_VARIABLES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_FORECAST_RASTER_VARIABLES))
        raise InvalidForecastRasterVariableError(
            f"Unsupported forecast raster variable '{variable}'. Supported values: {supported}."
        ) from exc


def _select_forecast_field(dataset: xr.Dataset, spec: RasterVariableSpec, horizon_day: int) -> xr.DataArray:
    if "time" not in dataset.coords:
        raise InvalidHorizonError("Forecast raster requests require a dataset with a time coordinate.")
    total_horizons = int(dataset.sizes.get("time", 0))
    if total_horizons <= 0:
        raise InvalidHorizonError("Forecast raster requests require at least one forecast timestep.")
    if horizon_day < 1 or horizon_day > total_horizons:
        raise InvalidHorizonError(
            f"Horizon day {horizon_day} is outside the available range 1 to {total_horizons}."
        )

    if spec.variable == "rainfall_daily_mm":
        field = _resolve_rainfall_data_array(dataset)
    elif spec.variable == "temperature_c":
        field = _resolve_temperature_data_array(dataset)
    else:
        raise InvalidForecastRasterVariableError(f"Unsupported forecast raster variable '{spec.variable}'.")

    selected = field.isel(time=horizon_day - 1).transpose("latitude", "longitude")
    return selected


def _resolve_rainfall_data_array(dataset: xr.Dataset) -> xr.DataArray:
    if "precip_mm" in dataset.data_vars:
        return dataset["precip_mm"]
    if "tp" in dataset.data_vars:
        rainfall = dataset["tp"]
        units = str(rainfall.attrs.get("units", "")).lower()
        if units in {"m", "meter", "meters"}:
            rainfall = rainfall * 1000.0
            rainfall.attrs["units"] = "mm"
        return rainfall
    raise InvalidForecastRasterVariableError("The forecast dataset does not expose a supported rainfall field.")


def _resolve_temperature_data_array(dataset: xr.Dataset) -> xr.DataArray:
    if "temp_c" in dataset.data_vars:
        return dataset["temp_c"]
    if "t2m" in dataset.data_vars:
        temperature = dataset["t2m"]
        units = str(temperature.attrs.get("units", "")).lower()
        if units in {"k", "kelvin"}:
            temperature = temperature - 273.15
            temperature.attrs["units"] = "C"
        return temperature
    raise InvalidForecastRasterVariableError("The forecast dataset does not expose a supported temperature field.")


def _to_utc_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _serialize_grid(values: np.ndarray, round_digits: int) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in values.tolist():
        serialized_row: list[float | None] = []
        for value in row:
            if value is None or not math.isfinite(float(value)):
                serialized_row.append(None)
                continue
            serialized_row.append(round(float(value), round_digits))
        rows.append(serialized_row)
    return rows


def _render_tile_png(prepared: PreparedForecastRaster, *, z: int, x: int, y: int) -> bytes:
    longitudes = _tile_pixel_longitudes(z, x)
    latitudes = _tile_pixel_latitudes(z, y)
    lon_indices, lon_inside = _nearest_indices(prepared.longitudes, longitudes)
    lat_indices, lat_inside = _nearest_indices(prepared.latitudes, latitudes)
    sampled = prepared.values[np.ix_(lat_indices, lon_indices)]
    valid_mask = np.isfinite(sampled) & lat_inside[:, None] & lon_inside[None, :]

    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    if np.any(valid_mask):
        if math.isclose(prepared.lower_bound, prepared.upper_bound):
            normalized = np.full(sampled.shape, 0.5, dtype=float)
        else:
            normalized = (sampled - prepared.lower_bound) / (prepared.upper_bound - prepared.lower_bound)
        normalized = np.clip(normalized, 0.0, 1.0)
        red, green, blue = _interpolate_color_channels(normalized, prepared.color_ramp)
        rgba[..., 0] = red
        rgba[..., 1] = green
        rgba[..., 2] = blue
        rgba[..., 3] = np.where(valid_mask, 214, 0).astype(np.uint8)

    return _encode_png(rgba)


def _nearest_indices(axis_values: np.ndarray, sample_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(axis_values, sample_values)
    indices = np.clip(indices, 0, len(axis_values) - 1)
    previous = np.clip(indices - 1, 0, len(axis_values) - 1)
    choose_previous = np.abs(sample_values - axis_values[previous]) <= np.abs(sample_values - axis_values[indices])
    nearest = np.where(choose_previous, previous, indices)
    inside = (sample_values >= axis_values.min()) & (sample_values <= axis_values.max())
    return nearest.astype(int), inside


def _nearest_axis_index(axis_values: np.ndarray, sample_value: float) -> int:
    if axis_values.size == 0:
        raise InvalidHorizonError("Forecast raster grid is empty.")
    indices, _ = _nearest_indices(axis_values, np.asarray([sample_value], dtype=float))
    return int(indices[0])


def _tile_pixel_longitudes(z: int, x: int) -> np.ndarray:
    pixel_positions = (np.arange(TILE_SIZE, dtype=float) + 0.5) / TILE_SIZE
    return ((x + pixel_positions) / (2**z)) * 360.0 - 180.0


def _tile_pixel_latitudes(z: int, y: int) -> np.ndarray:
    pixel_positions = (np.arange(TILE_SIZE, dtype=float) + 0.5) / TILE_SIZE
    mercator = math.pi * (1 - 2 * ((y + pixel_positions) / (2**z)))
    return np.degrees(np.arctan(np.sinh(mercator)))


def _interpolate_color_channels(
    normalized: np.ndarray,
    color_ramp: tuple[tuple[float, str], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stops = np.asarray([stop for stop, _ in color_ramp], dtype=float)
    reds = np.asarray([int(color[1:3], 16) for _, color in color_ramp], dtype=float)
    greens = np.asarray([int(color[3:5], 16) for _, color in color_ramp], dtype=float)
    blues = np.asarray([int(color[5:7], 16) for _, color in color_ramp], dtype=float)
    return (
        np.interp(normalized, stops, reds).astype(np.uint8),
        np.interp(normalized, stops, greens).astype(np.uint8),
        np.interp(normalized, stops, blues).astype(np.uint8),
    )


def _encode_png(rgba: np.ndarray) -> bytes:
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("PNG encoder expects an RGBA array.")
    raw = b"".join(b"\x00" + rgba[row_index].tobytes() for row_index in range(height))
    compressed = zlib.compress(raw, level=6)
    header = struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        ]
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return b"".join(
        [
            struct.pack("!I", len(payload)),
            chunk_type,
            payload,
            struct.pack("!I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF),
        ]
    )


def _forecast_source_label(source_id: str) -> str:
    return "Configured Forecast Feed" if source_id == "configured" else source_id.upper()
