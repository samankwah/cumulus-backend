"""Rainfall arithmetic for sub-seasonal runs: weekly windows, run lengths and wet/dry spells.

All gridded functions take arrays shaped ``(day, ...)`` so the same code serves a full
``(day, lat, lon)`` grid and a single ``(day,)`` point or area series.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np

SpellKind = Literal["dry", "wet"]


@dataclass(frozen=True)
class Window:
    index: int
    start_day: int  # 1-based lead day, inclusive
    end_day: int  # 1-based lead day, inclusive
    partial: bool

    @property
    def days(self) -> int:
        return self.end_day - self.start_day + 1


@dataclass(frozen=True)
class Spell:
    kind: SpellKind
    start_day: int
    end_day: int
    open_start: bool  # touches day 1, so it may have begun before the forecast window
    open_end: bool  # touches the last day, so it may continue after the forecast window

    @property
    def days(self) -> int:
        return self.end_day - self.start_day + 1


def weekly_windows(day_count: int, week_length: int = 7) -> list[Window]:
    """Consecutive 7-day windows counted from day 1; a trailing remainder becomes a partial window."""
    windows: list[Window] = []
    start = 1
    index = 1
    while start <= day_count:
        end = min(start + week_length - 1, day_count)
        windows.append(Window(index=index, start_day=start, end_day=end, partial=(end - start + 1) < week_length))
        start = end + 1
        index += 1
    return windows


def window_sums(values: np.ndarray, windows: list[Window]) -> np.ndarray:
    """Sum ``values`` (day, ...) over each window; NaN where every day in the window is NaN."""
    sums = []
    for window in windows:
        chunk = values[window.start_day - 1 : window.end_day]
        total = np.nansum(chunk, axis=0)
        all_missing = np.all(np.isnan(chunk), axis=0)
        sums.append(np.where(all_missing, np.nan, total))
    return np.stack(sums, axis=0)


def run_lengths(condition: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True values each day belongs to (0 where False).

    Works along axis 0 for any trailing shape, in O(days) vectorised passes.
    """
    condition = np.asarray(condition, dtype=bool)
    days = condition.shape[0]
    forward = np.zeros(condition.shape, dtype=np.int32)
    backward = np.zeros(condition.shape, dtype=np.int32)
    running = np.zeros(condition.shape[1:], dtype=np.int32)
    for day in range(days):
        running = np.where(condition[day], running + 1, 0)
        forward[day] = running
    running = np.zeros(condition.shape[1:], dtype=np.int32)
    for day in range(days - 1, -1, -1):
        running = np.where(condition[day], running + 1, 0)
        backward[day] = running
    return np.where(condition, forward + backward - 1, 0)


def spell_day_mask(condition: np.ndarray, min_days: int) -> np.ndarray:
    """True on days that belong to a run of ``condition`` lasting at least ``min_days``."""
    return run_lengths(condition) >= max(int(min_days), 1)


def wet_mask(values: np.ndarray, threshold_mm: float) -> np.ndarray:
    return np.nan_to_num(values, nan=-np.inf) >= threshold_mm


def dry_mask(values: np.ndarray, threshold_mm: float) -> np.ndarray:
    return np.isfinite(values) & (values < threshold_mm)


def outlook_metrics(values: np.ndarray, *, wet_threshold_mm: float, dry_spell_min_days: int, wet_spell_min_days: int) -> dict[str, np.ndarray]:
    """Whole-window outlook fields for ``values`` shaped (day, ...)."""
    wet = wet_mask(values, wet_threshold_mm)
    dry = dry_mask(values, wet_threshold_mm)
    all_missing = np.all(np.isnan(values), axis=0)
    total = np.where(all_missing, np.nan, np.nansum(values, axis=0))

    def _count(mask: np.ndarray) -> np.ndarray:
        return np.where(all_missing, np.nan, mask.sum(axis=0).astype(float))

    return {
        "total": total,
        "rainy_days": _count(wet),
        "dry_spell_days": _count(spell_day_mask(dry, dry_spell_min_days)),
        "wet_spell_days": _count(spell_day_mask(wet, wet_spell_min_days)),
    }


def find_spells(series: np.ndarray, *, wet_threshold_mm: float, dry_spell_min_days: int, wet_spell_min_days: int) -> list[Spell]:
    """List qualifying dry and wet spells in a 1-D daily series, in chronological order."""
    series = np.asarray(series, dtype=float)
    last_day = int(series.size)
    spells: list[Spell] = []
    for kind, mask, min_days in (
        ("dry", dry_mask(series, wet_threshold_mm), dry_spell_min_days),
        ("wet", wet_mask(series, wet_threshold_mm), wet_spell_min_days),
    ):
        day = 0
        while day < last_day:
            if not mask[day]:
                day += 1
                continue
            start = day
            while day < last_day and mask[day]:
                day += 1
            if day - start >= min_days:
                spells.append(
                    Spell(
                        kind=kind,  # type: ignore[arg-type]
                        start_day=start + 1,
                        end_day=day,
                        open_start=start == 0,
                        open_end=day == last_day,
                    )
                )
    return sorted(spells, key=lambda spell: spell.start_day)


def rain_day(init_date: date, lead_day: int) -> date:
    """Calendar day whose 00-24 UTC rain the given lead-day accumulation holds.

    Lead day 1 of a 00 UTC run accumulates from init to init + 24 h, i.e. the init date itself.
    Ghana is on UTC, so this is also the local calendar day.
    """
    return init_date + timedelta(days=lead_day - 1)
