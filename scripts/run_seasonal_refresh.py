"""Generate one or all seasonal advisory map products from the active forecast source."""

from __future__ import annotations

import argparse
import json

from cumulus.services.seasonal_map_service import generate_all_seasonal_map_products, generate_seasonal_map_product
from cumulus.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seasonal advisory map products.")
    parser.add_argument(
        "--season-profile",
        default=None,
        help="One of northern_single, southern_major, southern_minor.",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help="One of onset, cessation, early_dry_spell, late_dry_spell, rainfall_amount, rainy_days.",
    )
    parser.add_argument("--forecast-source", default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.theme and args.season_profile:
        payload = generate_seasonal_map_product(
            settings,
            args.theme,
            args.season_profile,
            forecast_source=args.forecast_source,
        )
    else:
        payload = generate_all_seasonal_map_products(settings, forecast_source=args.forecast_source)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
