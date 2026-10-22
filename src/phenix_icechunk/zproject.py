"""Z projection of every field, committed to the repo as its own icechunk transaction.

Projections cannot be virtual (they are new pixels), so they are written as real
chunks under ``/projections/z<method>``, a complete OME-Zarr HCS plate with the same
wells and fields as the root plate and ``z`` of size 1::

    /projections/zmax/<row>/<col>/<field>/0   (t, c, 1, y, x)

The projection is ``reduce_z(flatfield(raw))`` when flatfield correction is on, so
it is computed *after* the correction. ``max``/``min`` are stored as ``uint16``
(rounded, clipped), the other methods as ``float32``. It is committed separately from the virtual
plate: the commit holds only the projection plus the root ``phenix.projections``
entry, and a failed or repeated projection never touches the plate itself.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .convert import open_repo
from .flatfield import OUTPUT_DTYPE, to_dtype
from .sharding import shards_for
from .xarray_io import open_field

log = logging.getLogger(__name__)

METHODS: dict[str, Callable[..., np.ndarray]] = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "median": np.median,
    "sum": np.sum,
    "std": np.std,
}
# methods stored as OUTPUT_DTYPE (others are float32)
_INTEGER_OUT = {"max", "min"}


def projection_path(method: str) -> str:
    return f"projections/z{method}"


def _project(data: np.ndarray, method: str, pool: ThreadPoolExecutor) -> np.ndarray:
    """(t, c, z, y, x) -> (t, c, 1, y, x), one (t, c) stack per thread."""
    func = METHODS[method]
    integer_out = method in _INTEGER_OUT
    out = np.empty((*data.shape[:2], 1, *data.shape[3:]), dtype=OUTPUT_DTYPE if integer_out else np.float32)

    def one(tc: tuple[int, int]) -> None:
        t, c = tc
        stack = data[t, c]
        kwargs = {"dtype": np.float32} if method in ("mean", "sum", "std") else {}
        projected = func(stack, axis=0, **kwargs)
        out[t, c, 0] = to_dtype(projected, out.dtype) if integer_out else projected

    list(pool.map(one, product(range(data.shape[0]), range(data.shape[1]))))
    return out


def _field_attrs(src: dict[str, Any], method: str, info: dict[str, Any], dtype: np.dtype) -> dict[str, Any]:
    attrs = copy.deepcopy(src)
    if dtype.kind in "ui":  # the source window is for the raw dtype
        dmin, dmax = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
        for ch in attrs["ome"].get("omero", {}).get("channels", []):
            ch["window"] = {"min": dmin, "max": dmax, "start": 0, "end": dmax}
    ms = attrs["ome"]["multiscales"][0]
    ms["name"] = f"{ms['name']} z{method}"
    ms["metadata"] = {"description": f"z {method} projection", **info}
    zs = attrs.get("phenix", {}).get("plane_z_um") or []
    z_mid = float(np.mean(zs)) if zs else 0.0
    for t in ms["datasets"][0]["coordinateTransformations"]:
        if t["type"] == "translation":
            t["translation"][2] = z_mid
    attrs.setdefault("phenix", {})
    attrs["phenix"]["plane_z_um"] = [z_mid]
    attrs["phenix"]["zproject"] = info | {"source_plane_z_um": zs}
    return attrs


def zproject(
    repo_path: str | Path,
    method: str = "max",
    *,
    flatfield: bool | None = None,
    branch: str = "main",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Project every field along z and commit the result as one icechunk transaction.

    ``flatfield=None`` follows the repo default. With flatfield on, planes are
    corrected first and then projected. Re-running a method needs ``overwrite=True``,
    which replaces it in a new commit (history keeps the old one).
    """
    if method not in METHODS:
        raise ValueError(f"unknown projection {method!r}; choose from {sorted(METHODS)}")
    t0 = time.perf_counter()
    repo = open_repo(repo_path)
    source_session = repo.readonly_session(branch=branch)
    source = zarr.open_group(source_session.store, mode="r", zarr_format=3)
    if flatfield is None:
        flatfield = bool(source.attrs["phenix"].get("flatfield_default", False))

    session = repo.writable_session(branch)
    root = zarr.open_group(session.store, mode="r+", zarr_format=3)
    dest_path = projection_path(method)
    if dest_path in root:
        if not overwrite:
            raise FileExistsError(f"{dest_path} already exists; pass overwrite=True to recompute it")
        del root[dest_path]

    plate_meta = source.attrs["phenix"]
    n_planes = None
    info = {
        "method": method,
        "axis": "z",
        "flatfield_applied": flatfield,
        "order": "flatfield, then projection" if flatfield else "projection of raw planes",
        "source_snapshot": source_session.snapshot_id,
    }
    plate_ome = copy.deepcopy(source.attrs["ome"])
    plate_ome["plate"]["name"] = f"{plate_ome['plate']['name']} z{method}"
    dest = root.create_group(
        dest_path,
        attributes={"ome": plate_ome, "phenix": {"zproject": info, "plate_id": plate_meta.get("plate_id")}},
    )

    fields = [
        (well["path"], image["path"])
        for well in source.attrs["ome"]["plate"]["wells"]
        for image in source[well["path"]].attrs["ome"]["well"]["images"]
    ]
    for well in source.attrs["ome"]["plate"]["wells"]:
        dest.create_group(well["path"], attributes=dict(source[well["path"]].attrs))

    def read(well_path: str, field: str) -> np.ndarray:
        return open_field(source, well_path, field, flatfield=flatfield).values

    out_dtype = None
    with ThreadPoolExecutor(max_workers=1) as reader, ThreadPoolExecutor() as pool:
        pending = reader.submit(read, *fields[0])
        for k, (well_path, field) in enumerate(fields):
            data = pending.result()
            if k + 1 < len(fields):
                pending = reader.submit(read, *fields[k + 1])
            n_planes = data.shape[2]
            projected = _project(data, method, pool)
            out_dtype = projected.dtype
            group = dest[well_path].create_group(
                field,
                attributes=_field_attrs(
                    dict(source[f"{well_path}/{field}"].attrs), method, info | {"n_planes": n_planes}, projected.dtype
                ),
            )
            chunks = (1, 1, 1, *projected.shape[3:])
            arr = group.create_array(
                "0",
                shape=projected.shape,
                chunks=chunks,
                shards=shards_for(projected.shape, chunks, projected.dtype.itemsize),
                dtype=projected.dtype,
                fill_value=0,
                dimension_names=("t", "c", "z", "y", "x"),
            )
            arr[:] = projected
            if (k + 1) % 10 == 0 or k + 1 == len(fields):
                log.info("projected %d/%d fields", k + 1, len(fields))

    info |= {"n_planes": n_planes, "dtype": str(out_dtype), "path": dest_path, "fields": len(fields)}
    dest.attrs["phenix"] = {"zproject": info, "plate_id": plate_meta.get("plate_id")}
    root_phenix = dict(root.attrs["phenix"])
    root_phenix["projections"] = {**root_phenix.get("projections", {}), f"z{method}": info}
    root.attrs["phenix"] = root_phenix

    message = f"Z {method} projection of {len(fields)} fields" + (" after flatfield correction" if flatfield else "")
    snapshot = session.commit(message, metadata=info)
    summary = info | {"snapshot": snapshot, "message": message, "seconds": round(time.perf_counter() - t0, 1)}
    log.info("committed %s", summary)
    return summary
