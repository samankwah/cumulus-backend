"""Fixed, stepped colour scales for sub-seasonal layers.

Scales never rescale to the data: a given colour means the same amount on every day and
every run, which is what makes stepping through lead days meaningful (ECMWF/Meteoblue style).
Rain ramps run light sky-blue to deep violet (sequential, readable for common colour-vision
deficiencies); dry-spell ramps run sand to rust; wet-spell ramps run mint to deep teal.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        "bins": [
            {"min": item.lower, "max": item.upper, "color": item.color, "label": item.label}
            for item in legend.bins
        ],
    }
