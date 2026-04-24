"""Geospatial helpers."""

from __future__ import annotations

import numpy as np


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(values - target).argmin())
