"""Generate one or all seasonal advisory map products from the active forecast source."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from cumulus.services.seasonal_map_service import generate_seasonal_map_product, refresh_seasonal_map_products
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
    parser.add_argument("--mode", default=None, help="Optional mode filter: seasonal or calendar.")
    parser.add_argument("--subseason", default=None, help="Optional calendar subseason filter such as MAM or SON.")
    parser.add_argument("--forecast-source", default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.theme and args.season_profile and args.mode:
        payload = generate_seasonal_map_product(
            settings,
            args.theme,
            args.season_profile,
            mode=args.mode,
            subseason=args.subseason,
            forecast_source=args.forecast_source,
        )
    else:
        payload = refresh_seasonal_map_products(
            settings,
            theme=args.theme,
            season_profile=args.season_profile,
            mode=args.mode,
            subseason=args.subseason,
            forecast_source=args.forecast_source,
        )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
