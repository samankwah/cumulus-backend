"""Fixed, stepped colour scales for sub-seasonal layers.

Scales never rescale to the data: a given colour means the same amount on every day and
every run, which is what makes stepping through lead days meaningful (ECMWF/Meteoblue style).
Rain ramps run light sky-blue to deep violet (sequential, readable for common colour-vision
deficiencies); dry-spell ramps run sand to rust; wet-spell ramps run mint to deep teal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np


@dataclass(frozen=True)
class LegendBin:
    lower: float
    upper: float | None  # None = open-ended top bin
    color: str
    label: str


@dataclass(frozen=True)
class Legend:
    key: str
    unit: str
    bins: tuple[LegendBin, ...]
    note: str | None = None
    below_min_transparent: bool = False
    opacity: int = 222  # PNG alpha for coloured pixels
    categorical: bool = False  # bins are named classes, not numeric ranges


def _bins(edges: list[float], colors: list[str], unit: str, *, first_label: str | None = None) -> tuple[LegendBin, ...]:
    if len(colors) != len(edges):
        raise ValueError("Each lower edge needs exactly one colour.")
    bins: list[LegendBin] = []
    for index, lower in enumerate(edges):
        upper = edges[index + 1] if index + 1 < len(edges) else None
        if upper is None:
            label = f"{_fmt(lower)}+"
        elif index == 0 and first_label:
            label = first_label
        else:
            label = f"{_fmt(lower)}–{_fmt(upper)}"
        bins.append(LegendBin(lower=lower, upper=upper, color=colors[index], label=label))
    return tuple(bins)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


RAIN_DAILY = Legend(
    key="rain_daily",
    unit="mm",
    bins=_bins(
        [1, 2, 5, 10, 20, 30, 50, 75, 100],
        ["#cfeefa", "#9bd9f2", "#5cbbe6", "#2f96d4", "#2a6fc0", "#3e4ca8", "#5a2f97", "#7a1f86", "#9c1470"],
        "mm",
    ),
    note="Below 1 mm not shown",
    below_min_transparent=True,
)

RAIN_WEEKLY = Legend(
    key="rain_weekly",
    unit="mm",
    bins=_bins(
        [0, 5, 10, 25, 50, 75, 100, 150, 200],
        ["#f3e9cf", "#cfeefa", "#9bd9f2", "#5cbbe6", "#2f96d4", "#2a6fc0", "#3e4ca8", "#5a2f97", "#7a1f86"],
        "mm",
        first_label="<5",
    ),
)

RAIN_TOTAL = Legend(
    key="rain_total",
    unit="mm",
    bins=_bins(
        [0, 25, 50, 100, 150, 200, 300, 400],
        ["#f3e9cf", "#cfeefa", "#9bd9f2", "#5cbbe6", "#2f96d4", "#2a6fc0", "#3e4ca8", "#5a2f97"],
        "mm",
        first_label="<25",
    ),
)

RAINY_DAYS = Legend(
    key="rainy_days",
    unit="days",
    bins=_bins(
        [0, 5, 10, 15, 20, 25, 30, 35],
        ["#f3e9cf", "#cfeefa", "#9bd9f2", "#5cbbe6", "#2f96d4", "#2a6fc0", "#3e4ca8", "#5a2f97"],
        "days",
        first_label="<5",
    ),
)

DRY_SPELL_DAYS = Legend(
    key="dry_spell_days",
    unit="days",
    bins=_bins(
        [0, 1, 5, 10, 15, 20, 25, 30],
        ["#f4f1e8", "#fbe7a6", "#f8cf6a", "#f0a945", "#de7d2c", "#c0531f", "#95361a", "#6b2214"],
        "days",
        first_label="0",
    ),
)

WET_SPELL_DAYS = Legend(
    key="wet_spell_days",
    unit="days",
    bins=_bins(
        [0, 1, 5, 10, 15, 20, 25, 30],
        ["#f4f1e8", "#d4f0e3", "#a3dfc8", "#6cc7ac", "#3aa894", "#1f8580", "#14636a", "#0d4352"],
        "days",
        first_label="0",
    ),
)


# Per-day indicator maps: each cell is 100 (the day is a rain day / inside a spell) or 0, so an
# area mean reads as the share of the area. Only the flagged class is shaded.
RAINY_DAY_DAILY = Legend(
    key="rainy_day_daily",
    unit="%",
    bins=(LegendBin(lower=50, upper=None, color="#2f96d4", label="Rain day"),),
    note="Dry days not shaded",
    below_min_transparent=True,
    categorical=True,
)

DRY_SPELL_DAILY = Legend(
    key="dry_spell_daily",
    unit="%",
    bins=(LegendBin(lower=50, upper=None, color="#de7d2c", label="In a dry spell"),),
    note="Days outside a dry spell not shaded",
    below_min_transparent=True,
    categorical=True,
)

WET_SPELL_DAILY = Legend(
    key="wet_spell_daily",
    unit="%",
    bins=(LegendBin(lower=50, upper=None, color="#3aa894", label="In a wet spell"),),
    note="Days outside a wet spell not shaded",
    below_min_transparent=True,
    categorical=True,
)

# Per-week counts (0-7 days; the last window may be shorter). Same ramps as the outlook totals.
_WEEK_DAYS = [1, 2, 3, 4, 5, 6, 7]

RAINY_DAYS_WEEKLY = Legend(
    key="rainy_days_weekly",
    unit="days",
    bins=_bins(_WEEK_DAYS, ["#cfeefa", "#9bd9f2", "#5cbbe6", "#2f96d4", "#2a6fc0", "#3e4ca8", "#5a2f97"], "days"),
    note="Weeks without a rain day not shaded",
    below_min_transparent=True,
)

DRY_SPELL_DAYS_WEEKLY = Legend(
    key="dry_spell_days_weekly",
    unit="days",
    bins=_bins(_WEEK_DAYS, ["#fbe7a6", "#f8cf6a", "#f0a945", "#de7d2c", "#c0531f", "#95361a", "#6b2214"], "days"),
    note="Weeks without dry-spell days not shaded",
    below_min_transparent=True,
)

WET_SPELL_DAYS_WEEKLY = Legend(
    key="wet_spell_days_weekly",
    unit="days",
    bins=_bins(_WEEK_DAYS, ["#d4f0e3", "#a3dfc8", "#6cc7ac", "#3aa894", "#1f8580", "#14636a", "#0d4352"], "days"),
    note="Weeks without wet-spell days not shaded",
    below_min_transparent=True,
)


# Onset date: early onsets deep green, late ones pale yellow; grey where the window has no onset.
_ONSET_COLORS = ["#1b5e3b", "#2e7d4f", "#4f9a55", "#7db55a", "#a9c95f", "#d3d96a", "#f1e483"]
_NO_ONSET_COLOR = "#cfcac0"


def onset_legend(init_date: date, day_count: int, week_length: int = 7) -> Legend:
    """Weekly lead-day classes labelled with their calendar dates; depends on the run, so not fixed.

    Values are lead days (1 = the first forecast day); 0 means no onset inside the forecast.
    """
    bins = [LegendBin(lower=0, upper=1, color=_NO_ONSET_COLOR, label="No onset")]
    starts = list(range(1, day_count + 1, week_length))
    last = init_date
    for position, start in enumerate(starts):
        end = min(start + week_length - 1, day_count)
        first = init_date + timedelta(days=start - 1)
        last = init_date + timedelta(days=end - 1)
        bins.append(
            LegendBin(
                lower=start,
                upper=end + 1 if position + 1 < len(starts) else None,
                color=_ONSET_COLORS[min(position, len(_ONSET_COLORS) - 1)],
                label=f"{first.day} {first:%b}–{last.day} {last:%b}",
            )
        )
    return Legend(key="onset", unit="date", bins=tuple(bins), note=f"Grey: no onset by {last.day} {last:%b}")


# Onset countdown for one day: started, then how many days until it starts; grey if not in the forecast.
# Values are (onset lead day - selected day): <= 0 started, 1.. days to go, NO_ONSET for none.
NO_ONSET = 999.0

ONSET_COUNTDOWN = Legend(
    key="onset_countdown",
    unit="days_to_onset",
    bins=(
        LegendBin(lower=-NO_ONSET, upper=1, color=_ONSET_COLORS[0], label="Started"),
        LegendBin(lower=1, upper=4, color="#4f9a55", label="1–3 days"),
        LegendBin(lower=4, upper=8, color="#7db55a", label="4–7 days"),
        LegendBin(lower=8, upper=15, color="#a9c95f", label="8–14 days"),
        LegendBin(lower=15, upper=22, color="#d3d96a", label="15–21 days"),
        LegendBin(lower=22, upper=NO_ONSET, color="#f1e483", label="22+ days"),
        LegendBin(lower=NO_ONSET, upper=None, color=_NO_ONSET_COLOR, label="Not in forecast"),
    ),
    note="Days until the rains set in",
)


def classify(values: np.ndarray, legend: Legend) -> np.ndarray:
    """Bin index per value; -1 for NaN or (when transparent) below the first edge."""
    lowers = np.asarray([item.lower for item in legend.bins], dtype=float)
    finite = np.isfinite(values)
    safe = np.where(finite, values, -math.inf)
    index = np.searchsorted(lowers, safe, side="right") - 1
    if not legend.below_min_transparent:
        index = np.where(finite, np.maximum(index, 0), -1)
    return np.where(finite, index, -1).astype(np.int16)


def bin_colors_rgb(legend: Legend) -> np.ndarray:
    return np.asarray([[int(item.color[i : i + 2], 16) for i in (1, 3, 5)] for item in legend.bins], dtype=np.uint8)


def legend_payload(legend: Legend) -> dict[str, object]:
    return {
        "key": legend.key,
        "unit": legend.unit,
        "note": legend.note,
        "categorical": legend.categorical,
        "bins": [
            {"min": item.lower, "max": item.upper, "color": item.color, "label": item.label}
            for item in legend.bins
        ],
    }
