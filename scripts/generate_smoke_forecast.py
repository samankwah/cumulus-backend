from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def _cube(base: float, scale: float, time_count: int, lat_count: int, lon_count: int) -> list[list[list[float]]]:
    values: list[list[list[float]]] = []
    for t_idx in range(time_count):
        plane: list[list[float]] = []
        for lat_idx in range(lat_count):
            row: list[float] = []
            for lon_idx in range(lon_count):
                row.append(base + scale * ((t_idx + lat_idx + lon_idx) % 7))
            plane.append(row)
        values.append(plane)
    return values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "data" / "sample_forecast_smoke.nc"
    forecast_times = pd.date_range("2026-04-24", periods=14, freq="D")
    latitudes = [5.6037, 6.6927, 7.9465, 9.4075, 10.7870]
    longitudes = [-0.1870, -1.6234, -1.0232, -0.8533, -0.8500]

    dataset = xr.Dataset(
        data_vars={
            "tp": (
                ("time", "latitude", "longitude"),
                _cube(0.002, 0.0015, len(forecast_times), len(latitudes), len(longitudes)),
            ),
            "t2m": (
                ("time", "latitude", "longitude"),
                _cube(299.15, 0.6, len(forecast_times), len(latitudes), len(longitudes)),
            ),
            "u10": (
                ("time", "latitude", "longitude"),
                _cube(2.0, 0.15, len(forecast_times), len(latitudes), len(longitudes)),
            ),
            "v10": (
                ("time", "latitude", "longitude"),
                _cube(1.0, 0.12, len(forecast_times), len(latitudes), len(longitudes)),
            ),
        },
        coords={"time": forecast_times, "latitude": latitudes, "longitude": longitudes},
    )
    dataset["tp"].attrs["units"] = "m"
    dataset["t2m"].attrs["units"] = "K"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path, engine="scipy")
    print(output_path)


if __name__ == "__main__":
    main()
