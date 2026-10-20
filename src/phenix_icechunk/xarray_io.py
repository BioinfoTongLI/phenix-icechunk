"""Read the plate as labelled, lazy xarray objects with the stored metadata attached.

``open_field`` returns one field of view as an ``xr.DataArray`` with dims
``(t, c, z, y, x)``. Nothing is read until you index it and call ``.values``
or ``.load()``, and then only the needed planes are read.

Coordinates:

* ``t``: seconds since the start of the measurement (``timepoint_id`` alongside)
* ``c``: channel names, with ``channel_id``, wavelengths, exposure, colour, ...
* ``z``: plane z in µm (``plane_id`` alongside)
* ``y``, ``x``: µm in plate/stage space (``x = stageX + col*px``, ``y = -stageY + row*py``),
  the same transform as the OME-Zarr ``multiscales`` metadata
* per-plane ``(t, c, z)`` coords from ``phenix/images``: image id, acquisition time,
  absolute z, intensity stats and environment readings from the TIFF tags

``attrs`` holds the plate, well and field metadata (``attrs["ome"]`` is the
field's OME-Zarr metadata). ``open_datatree`` exposes the whole plate as an
``xr.DataTree``.

``flatfield=True`` divides each plane by its channel's Harmony flatfield profile
(see :mod:`phenix_icechunk.flatfield`), still lazily, and returns float32.

``zproject="max"`` (or another method) reads the stored z projection written by
:func:`phenix_icechunk.zproject.zproject` instead of the raw planes (``z`` has size 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

from .convert import open_plate
from .flatfield import PROFILE_KEY, flatfield_gain, flatfield_variable
from .metadata import read_table

# coordinate name -> column of phenix/images; columns missing from a repo are skipped
DEFAULT_PLANE_COORDS: dict[str, str] = {
    "image_id": "id",
    "acquisition_time": "AbsTime",
    "time_offset": "MeasurementTimeOffset",
    "abs_position_z": "AbsPositionZ",
    "intensity_mean": "tiff.tag65520.mean",
    "intensity_stddev": "tiff.tag65520.stddev",
    "intensity_min": "tiff.tag65520.min",
    "intensity_max": "tiff.tag65520.max",
    "temperature": "tiff.TiffImageInfo.DiagnosticData.CurrentTemperature",
    "co2": "tiff.TiffImageInfo.DiagnosticData.CurrentCO2",
    "humidity": "tiff.TiffImageInfo.DiagnosticData.CurrentHumidity",
}
_CHANNEL_COORDS = {
    "channel_id": "id",
    "excitation_wavelength": "excitation_nm",
    "emission_wavelength": "emission_nm",
    "exposure_time": "exposure_s",
    "acquisition_type": "acquisition_type",
    "channel_type": "channel_type",
    "binning": "binning_x",
    "camera": "camera_type",
    "objective_magnification": "objective_magnification",
    "objective_na": "objective_na",
    "color": "color",
}
_CHANNEL_UNITS = {"excitation_wavelength": "nm", "emission_wavelength": "nm", "exposure_time": "s"}


def _plain(values: np.ndarray) -> np.ndarray:
    """zarr returns numpy's variable-width StringDType; xarray wants fixed-width unicode."""
    return np.array(values.tolist(), dtype=str) if values.dtype.kind == "T" else values


class PlaneTable:
    """The columns of ``phenix/images`` needed for per-plane coordinates, read once per plate."""

    def __init__(self, plate: zarr.Group, plane_coords: Mapping[str, str]):
        table = plate["phenix/images"]
        available = set(table.attrs["table"]["columns"])
        self.units: dict[str, str] = table.attrs["table"]["units"]
        self.coords = {name: col for name, col in plane_coords.items() if col in available}
        needed = {"zarr_path", "t_index", "c_index", "z_index", "TimepointID", *self.coords.values()}
        self.columns = {c: _plain(table[c][:]) for c in needed if c in available}

    def rows(self, zarr_path: str) -> np.ndarray:
        return np.flatnonzero(self.columns["zarr_path"] == zarr_path)


def _to_datetime64(values: np.ndarray) -> np.ndarray:
    def parse(text: str):
        if not text:
            return np.datetime64("NaT", "ns")
        ts = datetime.fromisoformat(text).astimezone(UTC).replace(tzinfo=None)
        return np.datetime64(ts, "ns")

    return np.array([parse(str(v)) for v in values], dtype="datetime64[ns]")


def _plane_coords(table: PlaneTable, zarr_path: str, shape: tuple[int, int, int]) -> dict[str, Any]:
    rows = table.rows(zarr_path)
    t, c, z = (table.columns[k][rows] for k in ("t_index", "c_index", "z_index"))
    coords: dict[str, Any] = {}
    for name, col in table.coords.items():
        values = table.columns[col][rows]
        unit = table.units.get(col)
        if col == "AbsTime":
            values, fill = _to_datetime64(values), np.datetime64("NaT", "ns")
        elif values.dtype.kind in "iuf":
            values, fill = values.astype(np.float64), np.nan
        else:
            values, fill = values.astype(str), ""
        if unit == "m":  # positions are stored in metres; the image axes are in µm
            values, unit = values * 1e6, "micrometer"
        grid = np.full(shape, fill, dtype=values.dtype)
        grid[t, c, z] = values
        coords[name] = xr.Variable(("t", "c", "z"), grid, {"units": unit} if unit else {})
    # t coordinate: seconds since the measurement start, per timepoint
    tcoord = np.zeros(shape[0])
    tid = np.arange(1, shape[0] + 1)
    if "time_offset" in coords:
        offsets = coords["time_offset"].values
        tcoord = np.array([np.nanmin(offsets[i]) for i in range(shape[0])])
    if "TimepointID" in table.columns:
        tid = np.zeros(shape[0], dtype=np.int64)
        tid[t] = table.columns["TimepointID"][rows]
    coords["t"] = xr.Variable("t", tcoord, {"units": "s", "long_name": "time since measurement start"})
    coords["timepoint_id"] = xr.Variable("t", tid)
    return coords


def open_field(
    source: str | Path | zarr.Group,
    well: str,
    field: int | str,
    *,
    plane_coords: Mapping[str, str] | None = None,
    chunks: Any = None,
    flatfield: bool | None = None,
    zproject: str | None = None,
    _table: PlaneTable | None = None,
    _flatfield: tuple | None = None,
) -> xr.DataArray:
    """One field of view as a lazy, labelled ``xr.DataArray``.

    ``source`` is the repo path or an already opened plate group from
    :func:`open_plate` (pass the group when opening many fields).
    ``well`` is ``"A/4"`` or ``"A4"``, ``field`` the field group index (0-based).
    ``chunks`` is passed to xarray, e.g. ``{}`` to get a dask array (one chunk per plane).
    ``flatfield=True`` applies the Harmony flatfield profile on read (float32 result);
    ``None`` follows the repo default set by ``convert(..., flatfield=True)``.
    ``zproject`` selects a stored z projection (e.g. ``"max"``). Its pixels already have
    the flatfield it was computed with; ``flatfield`` must then be ``None`` or match.
    """
    plate = source if isinstance(source, zarr.Group) else open_plate(source)
    if "/" not in well:
        well = f"{well.rstrip('0123456789')}/{well[len(well.rstrip('0123456789')):]}"
    projection: dict[str, Any] | None = None
    base = ""
    if zproject is not None:
        available = plate.attrs.get("phenix", {}).get("projections", {})
        projection = available.get(f"z{zproject}")
        if projection is None:
            raise KeyError(f"no z{zproject} projection in this repo; available: {sorted(available)}")
        if flatfield is not None and flatfield != projection["flatfield_applied"]:
            raise ValueError(
                f"z{zproject} was computed with flatfield={projection['flatfield_applied']}; "
                "pass flatfield=None or recompute the projection"
            )
        base = projection["path"] + "/"
        flatfield = False  # already applied (or not) when the projection was computed
    elif flatfield is None:
        flatfield = bool(plate.attrs.get("phenix", {}).get("flatfield_default", False))
    path = f"{base}{well}/{field}"
    ds = xr.open_dataset(
        plate.store,
        engine="zarr",
        group=path,
        chunks=None if flatfield else chunks,
        decode_cf=False,  # never turn fill_value into NaN / float
        cache=False,  # re-read on access instead of keeping every loaded plane
        consolidated=False,
        zarr_format=3,
    )
    da = ds["0"]
    nt, nc, nz, ny, nx = da.shape
    field_ome = ds.attrs["ome"]
    field_meta = ds.attrs.get("phenix", {})
    plate_meta = plate.attrs.get("phenix", {})
    well_meta = plate[base + well].attrs.get("phenix", {})

    ms = field_ome["multiscales"][0]
    transforms = {t["type"]: t for t in ms["datasets"][0]["coordinateTransformations"]}
    scale = transforms["scale"]["scale"]
    offset = transforms.get("translation", {"translation": [0.0] * 5})["translation"]
    units = {a["name"]: a.get("unit") for a in ms["axes"]}

    coords: dict[str, Any] = {
        "y": xr.Variable("y", offset[3] + scale[3] * np.arange(ny), {"units": units["y"]}),
        "x": xr.Variable("x", offset[4] + scale[4] * np.arange(nx), {"units": units["x"]}),
    }
    plane_z = field_meta.get("plane_z_um") or list(offset[2] + scale[2] * np.arange(nz))
    coords["z"] = xr.Variable("z", np.asarray(plane_z, dtype=np.float64), {"units": units["z"]})
    coords["plane_id"] = xr.Variable("z", np.arange(1, nz + 1))

    channels = plate_meta.get("channels") or [
        {"name": ch.get("label", str(i))} for i, ch in enumerate(field_ome.get("omero", {}).get("channels", []))
    ]
    coords["c"] = xr.Variable("c", np.array([ch["name"] for ch in channels], dtype=str))
    for coord, key in _CHANNEL_COORDS.items():
        values = [ch.get(key) for ch in channels]
        if any(v is not None for v in values):
            attrs = {"units": _CHANNEL_UNITS[coord]} if coord in _CHANNEL_UNITS else {}
            coords[coord] = xr.Variable("c", np.array(values), attrs)

    table = _table or PlaneTable(plate, DEFAULT_PLANE_COORDS if plane_coords is None else plane_coords)
    if projection is None:
        coords |= _plane_coords(table, f"{path}/0", (nt, nc, nz))
    else:
        # per-plane coords don't survive a projection; keep the time axis of the source planes
        nz_source = plate[f"{well}/{field}/0"].shape[2]
        source_coords = _plane_coords(table, f"{well}/{field}/0", (nt, nc, nz_source))
        coords |= {k: source_coords[k] for k in ("t", "timepoint_id")}
        coords["plane_id"] = xr.Variable("z", [0], {"long_name": f"z {zproject} projection"})

    attrs = {
        "zarr_path": f"{path}/0",
        "well": well.replace("/", ""),
        **{k: v for k, v in well_meta.items() if k != "assay_layout"},
        "assay_layout": well_meta.get("assay_layout", {}),
        **field_meta,
        "pixel_size_um": [scale[3], scale[4]],
        **{
            k: plate_meta[k]
            for k in (
                "plate_id", "plate_name", "plate_type", "measurement_id", "measurement_start",
                "user", "instrument_type", "index_xml",
            )
            if k in plate_meta
        },
        "ome": field_ome,
    }
    if projection is not None:
        attrs["zproject"] = projection
        if projection["flatfield_applied"]:
            gain_info = flatfield_gain(plate, (ny, nx))[1]
            attrs["flatfield"] = {
                "applied": True,
                "formula": "raw * mean(P) / P, before the projection",
                "profile": f"FlatfieldProfile.{PROFILE_KEY}",
                "channels": gain_info,
            }
    variable = da.variable
    if flatfield:
        gain, info = _flatfield or flatfield_gain(plate, (ny, nx))
        variable = flatfield_variable(plate[f"{path}/0"], gain, da.dims)
        if chunks is not None:
            variable = variable.chunk({"t": 1, "c": 1, "z": 1} if chunks == {} else chunks)
        attrs["flatfield"] = {
            "applied": True,
            "formula": "raw * mean(P) / P",
            "profile": f"FlatfieldProfile.{PROFILE_KEY}",
            "channels": info,
        }
    return xr.DataArray(variable, coords=coords, name="image", attrs=attrs)


def open_datatree(
    source: str | Path | zarr.Group,
    *,
    plane_coords: Mapping[str, str] | None = None,
    chunks: Any = None,
    flatfield: bool | None = None,
    zproject: str | None = None,
    include_table: bool = True,
) -> xr.DataTree:
    """The whole plate as an ``xr.DataTree``.

    The tree mirrors the zarr layout: ``/<row>/<col>/<field>`` nodes hold an ``image``
    DataArray from :func:`open_field`. The root and well nodes carry the plate and well
    attrs, and ``/phenix/images`` holds the per-image table as a Dataset (dim ``image``).
    Pixels stay lazy. ``zproject="max"`` builds the tree from that stored projection.
    """
    plate = source if isinstance(source, zarr.Group) else open_plate(source)
    if flatfield is None and zproject is None:
        flatfield = bool(plate.attrs.get("phenix", {}).get("flatfield_default", False))
    table = PlaneTable(plate, DEFAULT_PLANE_COORDS if plane_coords is None else plane_coords)
    flat = (
        flatfield_gain(plate, tuple(plate.attrs["phenix"]["tiff_layout"]["shape"]))
        if flatfield and zproject is None
        else None
    )
    nodes: dict[str, xr.Dataset] = {"/": xr.Dataset(attrs=dict(plate.attrs))}
    for well in plate.attrs["ome"]["plate"]["wells"]:
        well_path = well["path"]
        well_group = plate[well_path]
        nodes[well_path] = xr.Dataset(attrs=dict(well_group.attrs))
        for image in well_group.attrs["ome"]["well"]["images"]:
            da = open_field(
                plate, well_path, image["path"], chunks=chunks, flatfield=flatfield,
                zproject=zproject, _table=table, _flatfield=flat,
            )
            nodes[f"{well_path}/{image['path']}"] = da.to_dataset()
    if include_table:
        table_group = plate["phenix/images"]
        data = read_table(table_group)
        images = xr.Dataset(
            {c: ("image", _plain(np.asarray(data[c]))) for c in table_group.attrs["table"]["columns"]}
        )
        images.attrs = dict(table_group.attrs)
        nodes["phenix"] = xr.Dataset(attrs=dict(plate["phenix"].attrs))
        nodes["phenix/images"] = images
    return xr.DataTree.from_dict(nodes)
