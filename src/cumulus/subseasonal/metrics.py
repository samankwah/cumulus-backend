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


def spell_day_mask(condition: np.ndarray, min_days: int, prefix: np.ndarray | None = None) -> np.ndarray:
    """True on days that belong to a run of ``condition`` lasting at least ``min_days``.

    ``prefix`` is the condition on the days just before the first one (observed days before a
    forecast starts, shaped (days, ...) and broadcast to ``condition``'s cells): a run already
    under way then counts in full, so three dry days before the start and two after make a spell.
    """
    return spell_masks(condition, min_days, prefix)[0]


def spell_masks(
    condition: np.ndarray, min_days: int, prefix: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(in_spell, starts): days inside a qualifying run, and the days such a run begins on.

    A run touching the first day only "starts" there when ``prefix`` shows the day before was not
    part of it; with no prefix nothing is known about the day before, so it is not a start.
    """
    condition = np.asarray(condition, dtype=bool)
    min_days = max(int(min_days), 1)
    lead = 0 if prefix is None else int(np.asarray(prefix).shape[0])
    if lead:
        before = np.broadcast_to(np.asarray(prefix, dtype=bool), (lead, *condition.shape[1:]))
        full = run_lengths(np.concatenate([before, condition], axis=0)) >= min_days
        in_spell = full[lead:]
        previous = full[lead - 1]
    else:
        in_spell = run_lengths(condition) >= min_days
        previous = np.ones(condition.shape[1:], dtype=bool)
    shifted = np.concatenate([previous[None], in_spell[:-1]], axis=0) if in_spell.shape[0] else in_spell
    return in_spell, in_spell & ~shifted


def starts_within(starts: np.ndarray, days: int) -> np.ndarray:
    """True on day d when a run starts on any of days d .. d + days - 1 (along axis 0)."""
    starts = np.asarray(starts, dtype=bool)
    counts = np.cumsum(starts, axis=0, dtype=np.int32)
    padded = np.concatenate([np.zeros((1, *starts.shape[1:]), dtype=np.int32), counts], axis=0)
    ends = np.minimum(np.arange(starts.shape[0]) + max(int(days), 1), starts.shape[0])
    return (padded[ends] - padded[np.arange(starts.shape[0])]) > 0


def wet_mask(values: np.ndarray, threshold_mm: float) -> np.ndarray:
    return np.nan_to_num(values, nan=-np.inf) >= threshold_mm


def dry_mask(values: np.ndarray, threshold_mm: float) -> np.ndarray:
    return np.isfinite(values) & (values < threshold_mm)


def outlook_metrics(
    values: np.ndarray,
    *,
    wet_threshold_mm: float,
    dry_spell_min_days: int,
    wet_spell_min_days: int,
    prefix: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Whole-window outlook fields for ``values`` shaped (day, ...); ``prefix`` is rain on the days
    just before (observed), so spells already under way at the start count in full."""
    wet = wet_mask(values, wet_threshold_mm)
    dry = dry_mask(values, wet_threshold_mm)
    wet_before = None if prefix is None else wet_mask(prefix, wet_threshold_mm)
    dry_before = None if prefix is None else dry_mask(prefix, wet_threshold_mm)
    all_missing = np.all(np.isnan(values), axis=0)
    total = np.where(all_missing, np.nan, np.nansum(values, axis=0))

    def _count(mask: np.ndarray) -> np.ndarray:
        return np.where(all_missing, np.nan, mask.sum(axis=0).astype(float))

    return {
        "total": total,
        "rainy_days": _count(wet),
        "dry_spell_days": _count(spell_day_mask(dry, dry_spell_min_days, dry_before)),
        "wet_spell_days": _count(spell_day_mask(wet, wet_spell_min_days, wet_before)),
    }


def find_spells(
    series: np.ndarray,
    *,
    wet_threshold_mm: float,
    dry_spell_min_days: int,
    wet_spell_min_days: int,
    prefix: np.ndarray | None = None,
) -> list[Spell]:
    """List qualifying dry and wet spells in a 1-D daily series, in chronological order.

    ``prefix`` is observed rain on the days just before the series: a spell already under way
    counts in full (and is open at the start), and one that begins on day 1 is known to begin
    there. Without it, a spell touching day 1 may have begun earlier, so it is open.
    """
    series = np.asarray(series, dtype=float)
    lead = 0 if prefix is None else int(np.asarray(prefix).size)
    full = series if not lead else np.concatenate([np.asarray(prefix, dtype=float).reshape(-1), series])
    last_day = int(full.size)
    spells: list[Spell] = []
    for kind, mask, min_days in (
        ("dry", dry_mask(full, wet_threshold_mm), dry_spell_min_days),
        ("wet", wet_mask(full, wet_threshold_mm), wet_spell_min_days),
    ):
        day = 0
        while day < last_day:
            if not mask[day]:
                day += 1
                continue
            start = day
            while day < last_day and mask[day]:
                day += 1
            if day - start >= min_days and day > lead:
                spells.append(
                    Spell(
                        kind=kind,  # type: ignore[arg-type]
                        start_day=max(start - lead, 0) + 1,
                        end_day=day - lead,
                        open_start=start < lead or (start == 0 and not lead),
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
    cell_shape = values.shape[1:]
    # Cells flattened to one axis, so a single series (shape (day,)) works like a grid.
    values = values.reshape(days, -1)
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
        # The window opens on a rain day, so the onset date is the first rainy day of the burst.
        candidate = ~found & ~dry[start] & (total >= threshold_mm)
        if not candidate.any():
            continue
        # The dry-spell guard only for the cells that passed the rain test: few do on any one day,
        # and checking every cell for every start day made this quadratic in the run length.
        where = np.nonzero(candidate)
        longest_found = run_lengths(dry[start:end][(slice(None), *where)]).max(axis=0)
        passed = longest_found <= guard_max_dry_days
        if not passed.any():
            continue
        hit = tuple(axis[passed] for axis in where)
        day[hit] = start + 1
        rain_mm[hit] = total[hit]
        longest_after[hit] = longest_found[passed]
        covered[hit] = end - start
        found[hit] = True
        if found.all():
            break
    day = np.where(all_missing, np.nan, day)
    return Onset(
        day=day.reshape(cell_shape),
        rain_mm=rain_mm.reshape(cell_shape),
        longest_dry_after=longest_after.reshape(cell_shape),
        guard_days=covered.reshape(cell_shape),
    )


def rain_day(init_date: date, lead_day: int) -> date:
    """Calendar day whose 00-24 UTC rain the given lead-day accumulation holds.

    Lead day 1 of a 00 UTC run accumulates from init to init + 24 h, i.e. the init date itself.
    Ghana is on UTC, so this is also the local calendar day.
    """
    return init_date + timedelta(days=lead_day - 1)
