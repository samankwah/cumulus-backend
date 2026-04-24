"""Time utilities."""

from __future__ import annotations

import pandas as pd


def to_utc_timestamp_series(values: pd.Series) -> pd.Series:
    series = pd.to_datetime(values, utc=True)
    return series.dt.tz_convert("UTC")
