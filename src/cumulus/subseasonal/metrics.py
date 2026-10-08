"""Rainfall arithmetic for sub-seasonal runs: weekly windows, run lengths, wet/dry spells and onset.

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


@dataclass(frozen=True)
class Onset:
    """Per-cell onset search over the forecast window, each array shaped like ``values[0]``.

    ``day`` is the 1-based lead day the onset rains start on (0 = no onset in the window, NaN =
    no data). ``guard_days`` is how many of the guard days the window still covered, so a value
    below the full guard means the dry-spell check was cut short by the end of the forecast.
    """

    day: np.ndarray
    rain_mm: np.ndarray
    longest_dry_after: np.ndarray
    guard_days: np.ndarray


def onset(
    values: np.ndarray,
    *,
    threshold_mm: float,
    window_days: int,
    guard_days: int,
    guard_max_dry_days: int,
    dry_threshold_mm: float,
) -> Onset:
    """First rain day from which ``threshold_mm`` falls within ``window_days`` days, with no dry
    spell longer than ``guard_max_dry_days`` in the ``guard_days`` days from that day.

    Mirrors the seasonal onset rule, searched only inside the forecast, with no fallback: a cell
    where no day qualifies has no onset. Rain is never negative, so "within at most N days" is the
    same test as an N-day sum.
    """
    values = np.asarray(values, dtype=float)
    days = values.shape[0]
    window_days = max(int(window_days), 1)
    guard_days = max(int(guard_days), 1)
    rain = np.nan_to_num(values, nan=0.0)
    dry = dry_mask(values, dry_threshold_mm)
    all_missing = np.all(np.isnan(values), axis=0)

    shape = values.shape[1:]
    found = np.zeros(shape, dtype=bool)
    day = np.zeros(shape, dtype=float)
    rain_mm = np.full(shape, np.nan)
    longest_after = np.full(shape, np.nan)
    covered = np.full(shape, np.nan)
    for start in range(days - window_days + 1):
        total = rain[start : start + window_days].sum(axis=0)
        end = min(start + guard_days, days)
        longest = run_lengths(dry[start:end]).max(axis=0)
        # The window opens on a rain day, so the onset date is the first rainy day of the burst.
        hit = ~found & ~dry[start] & (total >= threshold_mm) & (longest <= guard_max_dry_days)
        if not hit.any():
            continue
        day[hit] = start + 1
        rain_mm[hit] = total[hit]
        longest_after[hit] = longest[hit]
        covered[hit] = end - start
        found |= hit
        if found.all():
            break
    day = np.where(all_missing, np.nan, day)
    return Onset(day=day, rain_mm=rain_mm, longest_dry_after=longest_after, guard_days=covered)


def rain_day(init_date: date, lead_day: int) -> date:
    """Calendar day whose 00-24 UTC rain the given lead-day accumulation holds.

    Lead day 1 of a 00 UTC run accumulates from init to init + 24 h, i.e. the init date itself.
    Ghana is on UTC, so this is also the local calendar day.
    """
    return init_date + timedelta(days=lead_day - 1)
