from __future__ import annotations

import pandas as pd

from cumulus.advisory.rules import build_advisory
from cumulus.settings import AdvisoryConfig


def test_advisory_outputs_onset_and_cumulative_rainfall():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-05-01", periods=14, freq="D", tz="UTC"),
            "rainfall_corrected_mm": [8.0, 7.0, 6.0, 4.0, 3.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    advisory = build_advisory(frame, AdvisoryConfig())
    assert advisory["onset_date"] is not None
    assert advisory["cum_rain_7d_mm"] == 34.0
    assert advisory["seasonal_cum_rain_mm"] == 35.0


def test_advisory_detects_dry_spell():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-05-01", periods=10, freq="D", tz="UTC"),
            "rainfall_corrected_mm": [0.0] * 8 + [5.0, 6.0],
        }
    )
    advisory = build_advisory(frame, AdvisoryConfig())
    assert advisory["dry_spell_risk"] is True
    assert advisory["dry_spell_length_days"] == 8
