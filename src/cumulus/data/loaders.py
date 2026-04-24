"""Load and normalize gridded forecast datasets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock

import xarray as xr


COORDINATE_ALIASES = {
    "lat": "latitude",
    "latitude": "latitude",
    "y": "latitude",
    "lon": "longitude",
    "longitude": "longitude",
    "x": "longitude",
    "valid_time": "time",
    "date": "time",
}

_DATASET_LOAD_LOCK = Lock()

def open_dataset(
    path: str | Path,
    variable_aliases: dict[str, str] | None = None,
    variables: list[str] | None = None,
    chunks: dict[str, int] | None = None,
    engine: str | None = None,
) -> xr.Dataset:
    variable_alias_items = tuple(sorted((variable_aliases or {}).items()))
    requested_variables = tuple(variables or ())
    chunk_items = tuple(sorted((chunks or {}).items()))
    with _DATASET_LOAD_LOCK:
        dataset = _open_dataset_cached(
            str(Path(path).resolve()),
            variable_alias_items,
            requested_variables,
            chunk_items,
            engine,
        )
    return dataset.copy(deep=False)


@lru_cache(maxsize=8)
def _open_dataset_cached(
    path: str,
    variable_alias_items: tuple[tuple[str, str], ...],
    requested_variables: tuple[str, ...],
    chunk_items: tuple[tuple[str, int], ...],
    engine: str | None,
) -> xr.Dataset:
    variable_aliases = dict(variable_alias_items)
    variables = list(requested_variables)
    chunks = dict(chunk_items)
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    open_kwargs: dict[str, object] = {}
    if chunks:
        open_kwargs["chunks"] = chunks
    if engine is not None:
        ds = xr.open_dataset(dataset_path, engine=engine, **open_kwargs)
    elif suffix in {".grib", ".grb", ".grb2"}:
        ds = xr.open_dataset(dataset_path, engine="cfgrib", **open_kwargs)
    else:
        ds = xr.open_dataset(dataset_path, **open_kwargs)
    normalized = normalize_dataset(ds, variable_aliases)
    if variables:
        resolved_variables = resolve_variables(normalized, variables, variable_aliases)
        normalized = normalized[resolved_variables]
    normalized.load()
    ds.close()
    return normalized


def normalize_dataset(ds: xr.Dataset, variable_aliases: dict[str, str]) -> xr.Dataset:
    rename_map = {name: alias for name, alias in COORDINATE_ALIASES.items() if name in ds.coords or name in ds.dims}
    if rename_map:
        ds = ds.rename(rename_map)

    variable_rename_map = {
        source_name: target_name
        for source_name, target_name in variable_aliases.items()
        if source_name in ds.data_vars and source_name != target_name
    }
    if variable_rename_map:
        ds = ds.rename(variable_rename_map)

    if "time" not in ds.coords and "time" in ds:
        ds = ds.set_coords("time")

    if "longitude" in ds.coords:
        ds = _normalize_longitude(ds)
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")

    if "precip_mm" in ds.data_vars:
        ds["precip_mm"] = _convert_precip_to_mm(ds["precip_mm"])

    if "temp_c" in ds.data_vars:
        ds["temp_c"] = _convert_kelvin_to_celsius(ds["temp_c"])

    return ds


def _convert_precip_to_mm(data_array: xr.DataArray) -> xr.DataArray:
    units = str(data_array.attrs.get("units", "")).lower()
    values = data_array
    if units in {"m", "meter", "meters"}:
        values = values * 1000.0
        values.attrs["units"] = "mm"
    return values


def _convert_kelvin_to_celsius(data_array: xr.DataArray) -> xr.DataArray:
    units = str(data_array.attrs.get("units", "")).lower()
    if units in {"k", "kelvin"}:
        converted = data_array - 273.15
        converted.attrs["units"] = "C"
        return converted
    return data_array


def resolve_variables(ds: xr.Dataset, variables: list[str], variable_aliases: dict[str, str]) -> list[str]:
    resolved: list[str] = []
    for variable in variables:
        candidate = variable_aliases.get(variable, variable)
        if candidate in ds.data_vars:
            resolved.append(candidate)
    if not resolved:
        raise ValueError(f"None of the requested variables are present in dataset: {variables}")
    return resolved


def subset_ghana_bbox(
    ds: xr.Dataset,
    bounds: dict[str, float],
    variables: list[str] | None = None,
) -> xr.Dataset:
    dataset = ds[variables] if variables else ds
    return dataset.sel(
        latitude=slice(bounds["latitude_min"], bounds["latitude_max"]),
        longitude=slice(bounds["longitude_min"], bounds["longitude_max"]),
    )


def _normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    longitude = ds["longitude"]
    if float(longitude.max()) > 180:
        wrapped = (((longitude + 180) % 360) - 180)
        ds = ds.assign_coords(longitude=wrapped)
    return ds.sortby("longitude")
