"""Harmony / Acapella flatfield profiles: evaluate them and apply them lazily.

Each channel's ``<FlatfieldProfile>`` is a 2-D polynomial (``Type: Polynomial``)
in normalised coordinates::

    u = (x - Origin[0]) * Scale[0],   v = (y - Origin[1]) * Scale[1]
    P(x, y) = sum_k sum_j Coefficients[k][j] * u**(k - j) * v**j

The term order (x power decreasing within each degree) was checked against the
data: the median over all fields of each channel correlates with P at 0.95-0.98,
against 0.5-0.8 for the transposed order.

Correction: ``corrected = raw * mean(P) / P``, i.e. P is normalised to mean 1 so
intensities keep their original scale.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
import xarray as xr
import zarr
from xarray.backends import BackendArray
from xarray.core import indexing

PROFILE_KEY = "Background"
_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4, thread_name_prefix="flatfield")


def profile_image(profile: dict[str, Any], shape: tuple[int, int], key: str = PROFILE_KEY) -> np.ndarray | None:
    """Evaluate a parsed ``FlatfieldProfile`` on a ``(height, width)`` grid, normalised to mean 1.

    Returns ``None`` if the profile has no polynomial (e.g. ``Character: Flat``).
    """
    poly = (profile.get(key) or {}).get("Profile")
    if not poly or poly.get("Type", "Polynomial") != "Polynomial":
        return None
    height, width = shape
    dims_x, dims_y = poly.get("Dims", [width, height])
    (ox, oy), (sx, sy) = poly["Origin"], poly["Scale"]
    # Pixel centres of this image mapped into the profile's pixel grid (handles binning).
    x = (np.arange(width) + 0.5) * dims_x / width - 0.5
    y = (np.arange(height) + 0.5) * dims_y / height - 0.5
    u = (x - ox) * sx
    v = (y - oy) * sy
    out = np.zeros(shape, dtype=np.float64)
    for k, row in enumerate(poly["Coefficients"]):
        for j, c in enumerate(row):
            out += c * np.outer(v**j, u ** (k - j))
    return (out / out.mean()).astype(np.float32)


def _channel_profiles(plate: zarr.Group) -> list[tuple[str, dict[str, Any] | None]]:
    """(channel name, parsed FlatfieldProfile or None) in array ``c`` order."""
    entries = plate["phenix"].attrs["index_xml"]["channels"]
    out = []
    for ch in plate.attrs["phenix"]["channels"]:
        ff = entries.get(str(ch["id"]), {}).get("entries", {}).get("FlatfieldProfile")
        out.append((ch["name"], ff if isinstance(ff, dict) else None))
    return out


def _summary(ff: dict[str, Any] | None) -> dict[str, Any]:
    background = (ff or {}).get(PROFILE_KEY) or {}
    has_poly = bool((background.get("Profile") or {}).get("Coefficients"))
    return {
        "applied": has_poly,
        **{k: background.get(k) for k in ("Character", "Mean", "NoiseConst", "NonFlatness", "Quality")},
        "version": (ff or {}).get("Version"),
    }


def _evaluate(ff: dict[str, Any] | None, shape: tuple[int, int]) -> np.ndarray:
    img = profile_image(ff, shape) if ff else None
    return np.ones(shape, np.float32) if img is None else img


@lru_cache(maxsize=4)
def _gain(profiles_json: str, shape: tuple[int, int]) -> np.ndarray:
    profiles = np.stack([_evaluate(ff, shape) for ff in json.loads(profiles_json)])
    return (1.0 / np.where(profiles > 0, profiles, 1.0)).astype(np.float32)


def flatfield_gain(plate: zarr.Group, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """``mean(P) / P`` per channel as ``(C, H, W)`` float32 (cached), plus a per-channel summary."""
    channels = _channel_profiles(plate)
    gain = _gain(json.dumps([ff for _, ff in channels], sort_keys=True), tuple(shape))
    return gain, {name: _summary(ff) for name, ff in channels}


def flatfield_profiles(plate: zarr.Group) -> xr.DataArray:
    """The normalised flatfield of every channel as a ``(c, y, x)`` DataArray, for inspection."""
    shape = tuple(plate.attrs["phenix"]["tiff_layout"]["shape"])
    channels = _channel_profiles(plate)
    info = {name: _summary(ff) for name, ff in channels}
    return xr.DataArray(
        np.stack([_evaluate(ff, shape) for _, ff in channels]),
        dims=("c", "y", "x"),
        coords={"c": [name for name, _ in channels]},
        name="flatfield",
        attrs={"normalisation": "mean 1", "profile": PROFILE_KEY, "channels": info},
    )


class FlatfieldArray(BackendArray):
    """Lazy ``(t, c, z, y, x)`` view of a zarr array with ``raw * gain[c, y, x]`` applied on read."""

    def __init__(self, array: zarr.Array, gain: np.ndarray):
        self.array = array
        self.gain = gain  # (C, H, W) = mean(P) / P
        self.shape = array.shape
        self.dtype = np.dtype(np.float32)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key: tuple) -> np.ndarray:
        # Keep every axis while reading so the gain broadcasts, then drop integer-indexed ones.
        full = tuple(slice(k, k + 1) if isinstance(k, (int, np.integer)) else k for k in key)
        raw = self.array[full]
        gain = self.gain[full[1], full[3], full[4]]
        out = np.empty(raw.shape, dtype=np.float32)

        def plane(idx: tuple[int, int, int]) -> None:
            t, c, z = idx
            # one fused uint -> float32 multiply per plane; numpy releases the GIL
            np.multiply(raw[t, c, z], gain[c], out=out[t, c, z])

        list(_POOL.map(plane, product(*(range(n) for n in raw.shape[:3]))))
        drop = tuple(i for i, k in enumerate(key) if isinstance(k, (int, np.integer)))
        return out.squeeze(axis=drop) if drop else out


def flatfield_variable(array: zarr.Array, gain: np.ndarray, dims: tuple[str, ...]) -> xr.Variable:
    return xr.Variable(dims, indexing.LazilyIndexedArray(FlatfieldArray(array, gain)))
