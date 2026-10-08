"""Rasterised Ghana district polygons for the sub-seasonal grid.

The 0.1 degree forecast grid is too coarse to mask tiles cleanly or to average small districts,
so the district polygons are burned once into a fine raster (``FINE_FACTOR`` sub-pixels per
forecast cell, ~1.1 km at 0.01 degrees) whose extent is aligned to the forecast cell edges:

* the fine raster masks map tiles at pixel level (sharp coastline / borders);
* summing sub-pixels per forecast cell gives exact area weights for district and region means.

The result is cached as a small compressed ``.npz`` next to the run artifacts and rebuilt
whenever the geojson or the grid changes (keyed by a hash of both).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

FINE_FACTOR = 10
MASK_FILE_NAME = "ghana_district_mask.npz"


@dataclass(frozen=True)
class AreaIndex:
    """District raster plus per-area cell weights on the forecast grid."""

    key: str
    latitudes: np.ndarray  # forecast-cell centres, ascending
    longitudes: np.ndarray  # forecast-cell centres, ascending
    fine_latitudes: np.ndarray  # fine-pixel centres, ascending
    fine_longitudes: np.ndarray
    fine_district: np.ndarray  # int16 (fine_lat, fine_lon), -1 outside Ghana
    district_names: tuple[str, ...]
    district_regions: tuple[str, ...]
    region_names: tuple[str, ...]

    @property
    def fine_inside(self) -> np.ndarray:
        return self.fine_district >= 0

    def cell_land_fraction(self) -> np.ndarray:
        return _block_mean(self.fine_inside.astype(np.float32))

    def district_weights(self, name: str) -> np.ndarray | None:
        try:
            index = self.district_names.index(name)
        except ValueError:
            return None
        return _block_mean((self.fine_district == index).astype(np.float32))

    def region_weights(self, name: str) -> np.ndarray | None:
        indices = [i for i, region in enumerate(self.district_regions) if region == name]
        if not indices:
            return None
        return _block_mean(np.isin(self.fine_district, indices).astype(np.float32))

    def all_weights(self, level: str) -> dict[str, np.ndarray]:
        names = self.district_names if level == "district" else self.region_names
        getter = self.district_weights if level == "district" else self.region_weights
        weights: dict[str, np.ndarray] = {}
        for name in names:
            value = getter(name)
            if value is not None and float(value.sum()) > 0:
                weights[name] = value
        return weights


@dataclass(frozen=True)
class AreaWeights:
    """Land fraction per forecast cell and one weight row per region/district (cells flattened)."""

    land: np.ndarray  # (lat, lon) float32
    matrices: dict[str, tuple[tuple[str, ...], np.ndarray]]  # level -> (names, (areas, cells) float32)


_WEIGHTS_CACHE: dict[str, AreaWeights] = {}


def area_weights(index: AreaIndex) -> AreaWeights:
    """Weights for every area in one pass over the fine raster, shared by all runs on the same grid.

    Equivalent to ``all_weights`` (fraction of each cell's sub-pixels inside the area), but a single
    ``bincount`` instead of one full-raster comparison per district.
    """
    cached = _WEIGHTS_CACHE.get(index.key)
    if cached is not None:
        return cached
    fine = index.fine_district
    rows, cols = fine.shape[0] // FINE_FACTOR, fine.shape[1] // FINE_FACTOR
    cells = rows * cols
    cell_id = (np.arange(fine.shape[0])[:, None] // FINE_FACTOR) * cols + (np.arange(fine.shape[1])[None, :] // FINE_FACTOR)
    inside = fine >= 0
    districts = len(index.district_names)
    counts = np.bincount(
        fine[inside].astype(np.int64) * cells + cell_id[inside],
        minlength=districts * cells,
    ).reshape(districts, cells).astype(np.float32) / float(FINE_FACTOR * FINE_FACTOR)
    region_rows = {name: np.zeros(cells, dtype=np.float32) for name in index.region_names}
    for position, region in enumerate(index.district_regions):
        region_rows[region] += counts[position]

    def _matrix(named: list[tuple[str, np.ndarray]]) -> tuple[tuple[str, ...], np.ndarray]:
        kept = [(name, row) for name, row in named if float(row.sum()) > 0]
        names = tuple(name for name, _ in kept)
        matrix = np.stack([row for _, row in kept]).astype(np.float32) if kept else np.zeros((0, cells), np.float32)
        return names, matrix

    weights = AreaWeights(
        land=_block_mean(inside.astype(np.float32)).astype(np.float32),
        matrices={
            "district": _matrix(list(zip(index.district_names, counts))),
            "region": _matrix([(name, region_rows[name]) for name in index.region_names]),
        },
    )
    if len(_WEIGHTS_CACHE) >= 4:
        _WEIGHTS_CACHE.pop(next(iter(_WEIGHTS_CACHE)))
    _WEIGHTS_CACHE[index.key] = weights
    return weights


def _block_mean(fine: np.ndarray) -> np.ndarray:
    rows, cols = fine.shape
    return fine.reshape(rows // FINE_FACTOR, FINE_FACTOR, cols // FINE_FACTOR, FINE_FACTOR).mean(axis=(1, 3))


def fine_axis(centres: np.ndarray) -> np.ndarray:
    """Fine-pixel centres covering the cells of an evenly spaced, ascending centre axis."""
    step = float(centres[1] - centres[0]) if centres.size > 1 else 0.1
    fine_step = step / FINE_FACTOR
    start = float(centres[0]) - step / 2 + fine_step / 2
    return start + fine_step * np.arange(centres.size * FINE_FACTOR, dtype=float)


def load_area_index(geojson_path: Path, latitudes: np.ndarray, longitudes: np.ndarray, cache_dir: Path | None) -> AreaIndex:
    latitudes = np.round(np.asarray(latitudes, dtype=float), 4)
    longitudes = np.round(np.asarray(longitudes, dtype=float), 4)
    return _load_area_index_cached(
        str(geojson_path),
        tuple(latitudes.tolist()),
        tuple(longitudes.tolist()),
        str(cache_dir) if cache_dir else "",
    )


@lru_cache(maxsize=4)
def _load_area_index_cached(geojson_path: str, latitudes: tuple[float, ...], longitudes: tuple[float, ...], cache_dir: str) -> AreaIndex:
    geojson_bytes = Path(geojson_path).read_bytes()
    key = hashlib.sha1(geojson_bytes + repr((latitudes, longitudes, FINE_FACTOR)).encode()).hexdigest()[:16]
    lat_axis = np.asarray(latitudes)
    lon_axis = np.asarray(longitudes)
    cache_file = Path(cache_dir) / MASK_FILE_NAME if cache_dir else None

    if cache_file and cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                if str(cached["key"]) == key:
                    return _area_index(key, lat_axis, lon_axis, cached["fine_district"], cached["district_names"], cached["district_regions"])
        except Exception:
            pass  # stale or corrupt cache: rebuild below

    features = json.loads(geojson_bytes.decode("utf-8")).get("features", [])
    names = [str((feature.get("properties") or {}).get("display_name") or "").strip() for feature in features]
    regions = [str((feature.get("properties") or {}).get("region") or "").strip() for feature in features]
    fine_district = rasterize_features(features, fine_axis(lat_axis), fine_axis(lon_axis))

    if cache_file:
        _write_cache(cache_file, key=key, fine_district=fine_district, names=names, regions=regions)
    return _area_index(key, lat_axis, lon_axis, fine_district, np.asarray(names), np.asarray(regions))


def _area_index(key: str, latitudes: np.ndarray, longitudes: np.ndarray, fine_district: np.ndarray, names: Any, regions: Any) -> AreaIndex:
    district_names = tuple(str(item) for item in np.asarray(names).tolist())
    district_regions = tuple(str(item) for item in np.asarray(regions).tolist())
    return AreaIndex(
        key=key,
        latitudes=latitudes,
        longitudes=longitudes,
        fine_latitudes=fine_axis(latitudes),
        fine_longitudes=fine_axis(longitudes),
        fine_district=np.asarray(fine_district, dtype=np.int16),
        district_names=district_names,
        district_regions=district_regions,
        region_names=tuple(sorted(set(district_regions))),
    )


def _write_cache(path: Path, *, key: str, fine_district: np.ndarray, names: list[str], regions: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        os.close(handle)
        np.savez_compressed(
            temporary,
            key=np.asarray(key),
            fine_district=fine_district.astype(np.int16),
            district_names=np.asarray(names),
            district_regions=np.asarray(regions),
        )
        os.replace(temporary, path)
    except OSError:
        # Read-only deployments (Vercel) simply recompute in memory.
        pass


def rasterize_features(features: list[dict[str, Any]], fine_latitudes: np.ndarray, fine_longitudes: np.ndarray) -> np.ndarray:
    """Burn each feature's index into a raster of pixel centres (even-odd rule, holes respected)."""
    raster = np.full((fine_latitudes.size, fine_longitudes.size), -1, dtype=np.int16)
    lat_step = float(fine_latitudes[1] - fine_latitudes[0])
    lon_step = float(fine_longitudes[1] - fine_longitudes[0])
    for index, feature in enumerate(features):
        geometry = feature.get("geometry") or {}
        polygons = _polygons(geometry)
        if not polygons:
            continue
        all_points = np.concatenate([np.asarray(ring, dtype=float)[:, :2] for polygon in polygons for ring in polygon])
        min_lon, min_lat = all_points.min(axis=0)
        max_lon, max_lat = all_points.max(axis=0)
        row_start = max(int(np.floor((min_lat - fine_latitudes[0]) / lat_step)), 0)
        row_stop = min(int(np.ceil((max_lat - fine_latitudes[0]) / lat_step)) + 1, fine_latitudes.size)
        col_start = max(int(np.floor((min_lon - fine_longitudes[0]) / lon_step)), 0)
        col_stop = min(int(np.ceil((max_lon - fine_longitudes[0]) / lon_step)) + 1, fine_longitudes.size)
        if row_start >= row_stop or col_start >= col_stop:
            continue
        lon_grid, lat_grid = np.meshgrid(fine_longitudes[col_start:col_stop], fine_latitudes[row_start:row_stop])
        inside = np.zeros(lon_grid.shape, dtype=bool)
        for polygon in polygons:
            polygon_inside = np.zeros(lon_grid.shape, dtype=bool)
            for ring in polygon:  # outer ring and holes toggle with the even-odd rule
                polygon_inside ^= _points_in_ring(lon_grid, lat_grid, np.asarray(ring, dtype=float)[:, :2])
            inside |= polygon_inside
        block = raster[row_start:row_stop, col_start:col_stop]
        block[inside & (block < 0)] = index
    return raster


def _polygons(geometry: dict[str, Any]) -> list[list[list[list[float]]]]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "Polygon":
        return [coordinates]
    if kind == "MultiPolygon":
        return list(coordinates)
    return []


def _points_in_ring(x: np.ndarray, y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    inside = np.zeros(x.shape, dtype=bool)
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        if ay == by:
            continue
        crosses = (ay > y) != (by > y)
        if not crosses.any():
            continue
        x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (x < x_cross)
    return inside
