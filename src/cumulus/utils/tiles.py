"""Shared XYZ map-tile helpers: Web Mercator pixel coordinates and a dependency-free PNG encoder."""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np

TILE_SIZE = 256


def tile_pixel_longitudes(z: int, x: int, size: int = TILE_SIZE) -> np.ndarray:
    pixel_positions = (np.arange(size, dtype=float) + 0.5) / size
    return ((x + pixel_positions) / (2**z)) * 360.0 - 180.0


def tile_pixel_latitudes(z: int, y: int, size: int = TILE_SIZE) -> np.ndarray:
    pixel_positions = (np.arange(size, dtype=float) + 0.5) / size
    mercator = math.pi * (1 - 2 * ((y + pixel_positions) / (2**z)))
    return np.degrees(np.arctan(np.sinh(mercator)))


def encode_png(rgba: np.ndarray) -> bytes:
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("PNG encoder expects an RGBA array.")
    raw = b"".join(b"\x00" + rgba[row_index].tobytes() for row_index in range(height))
    compressed = zlib.compress(raw, level=6)
    header = struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        ]
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return b"".join(
        [
            struct.pack("!I", len(payload)),
            chunk_type,
            payload,
            struct.pack("!I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF),
        ]
    )
