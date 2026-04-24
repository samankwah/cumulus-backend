"""Run batch forecast generation from CLI."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from cumulus.schemas import LocationRequest
from cumulus.services.forecast_service import generate_forecast
from cumulus.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-path", required=True)
    parser.add_argument("--location-id", required=True)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--horizon-days", default=14, type=int)
    args = parser.parse_args()

    results, metadata = generate_forecast(
        get_settings(),
        locations=[LocationRequest(location_id=args.location_id, latitude=args.lat, longitude=args.lon)],
        forecast_path=args.forecast_path,
        variables=["tp", "u10", "v10", "t2m"],
        horizon_days=args.horizon_days,
    )
    print({"model_version": metadata["model_version"], "results": [item.model_dump() for item in results]})


if __name__ == "__main__":
    main()
