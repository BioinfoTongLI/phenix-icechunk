"""Export the plate to Bio-Formats-compatible OME-TIFF, one file per well.

Each well becomes ``<plate>_<well>.ome.tiff``:

* one OME Image (series) per field of view, i.e. one tile per field, planes in XYZCT order;
* pixels stored as 512x512 TIFF tiles (zlib by default), BigTIFF;
* the first IFD's ImageDescription holds an OME-XML with the Plate/Well/WellSample
  layout, stage positions, channels, per-plane times/positions, instrument and
  annotations (the same document as ``/OME/METADATA.ome.xml``, restricted to the well),
  with ``TiffData`` pointing every image at its IFDs.

Everything is read back from the icechunk repo (Index.xml copy, tables, attrs), so
this works on any repo written by :func:`phenix_icechunk.convert`.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
import zarr

from .convert import _row_name, open_plate
from .flatfield import OUTPUT_DTYPE, to_dtype
from .index_xml import PhenixIndex, parse_index
from .metadata import read_file
from .ome_xml import PIXEL_TYPES, OmeField, build_ome_xml
from .xarray_io import open_field

log = logging.getLogger(__name__)

_COMPRESSION = {"zlib": ("zlib", {"level": 1}), "lzw": ("lzw", None), "none": (None, None)}
# TiffImageInfo values used for OME ImagingEnvironment
_TIFF_PREFIX = "tiff."
_ENV_COLUMNS = ("CurrentTemperature", "CurrentCO2", "CurrentHumidity")


@dataclass
class _PlateContext:
    index: PhenixIndex
    channel_ids: list[int]
    plane_ids: list[int]
    timepoint_ids: list[int]
    dtype: np.dtype
    colors: dict[int, str]
    tiff_values: dict[str, dict[str, Any]]
    assay_layout: dict[str, Any] | None
    wells: dict[str, list[tuple[str, OmeField]]]  # well path -> [(field group, OmeField)]


def _load_context(plate: zarr.Group) -> _PlateContext:
    meta = plate.attrs["phenix"]
    index_path = Path(meta["index_xml"])
    try:
        rel = index_path.relative_to(meta["measurement_dir"]).as_posix()
        index = parse_index(index_path, data=read_file(plate["phenix/files"][rel]))
    except (KeyError, ValueError):
        index = parse_index(index_path)

    table = plate["phenix/images"]
    columns = table.attrs["table"]["columns"]
    env_cols = [c for c in columns if c.endswith(_ENV_COLUMNS) and c.startswith(_TIFF_PREFIX)]
    urls = table["URL"][:].tolist()
    env = {c: table[c][:].tolist() for c in env_cols}
    tiff_values = {
        url: {c[len(_TIFF_PREFIX):]: env[c][i] for c in env_cols} for i, url in enumerate(urls)
    }

    images = [im for im in index.images if im.ok]
    wells: dict[str, list[tuple[str, OmeField]]] = {}
    for well in plate.attrs["ome"]["plate"]["wells"]:
        well_group = plate[well["path"]]
        row, col = well_group.attrs["phenix"]["row"], well_group.attrs["phenix"]["col"]
        entries = []
        for image in well_group.attrs["ome"]["well"]["images"]:
            field_meta = well_group[image["path"]].attrs["phenix"]
            fid = field_meta["field_id"]
            entries.append(
                (
                    image["path"],
                    OmeField(
                        well_name=f"{_row_name(row)}{col}",
                        row=row,
                        col=col,
                        field_id=fid,
                        images=[im for im in images if (im.row, im.col, im.field) == (row, col, fid)],
                        physical_size_z_um=field_meta["physical_size_z_um"],
                    ),
                )
            )
        wells[well["path"]] = entries

    channels = meta["channels"]
    assay_layouts = plate["phenix"].attrs.get("assay_layouts") or {}
    return _PlateContext(
        index=index,
        channel_ids=[ch["id"] for ch in channels],
        plane_ids=sorted({im.plane for im in images}),
        timepoint_ids=sorted({im.timepoint for im in images}),
        dtype=np.dtype(meta["tiff_layout"]["dtype"]),
        colors={ch["id"]: ch["color"] for ch in channels},
        tiff_values=tiff_values,
        assay_layout=next(iter(assay_layouts.values()), None),
        wells=wells,
    )


def export_ome_tiff(
    source: str | Path | zarr.Group,
    out_dir: str | Path,
    *,
    flatfield: bool | None = None,
    zproject: str | None = None,
    compression: Literal["zlib", "lzw", "none"] = "zlib",
    tile: int = 512,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write one multi-series OME-TIFF per well into ``out_dir``.

    ``flatfield=None`` follows the repo default (``convert(..., flatfield=True)``).
    Raw stacks keep the source dtype; flatfield-corrected pixels are rounded and clipped
    to ``uint16``.
    ``zproject="max"`` exports that stored projection instead of the z stacks
    (``<plate>_<well>_zmax.ome.tiff``, SizeZ=1, in the projection's own dtype).
    """
    plate = source if isinstance(source, zarr.Group) else open_plate(source)
    ctx = _load_context(plate)
    plane_ids = ctx.plane_ids
    out_dtype = ctx.dtype
    suffix = description = ""
    if zproject is not None:
        projection = plate.attrs["phenix"].get("projections", {}).get(f"z{zproject}")
        if projection is None:
            raise KeyError(f"no z{zproject} projection in this repo; run zproject() first")
        if flatfield is not None and flatfield != projection["flatfield_applied"]:
            raise ValueError(f"z{zproject} was computed with flatfield={projection['flatfield_applied']}")
        flatfield = projection["flatfield_applied"]
        plane_ids = ctx.plane_ids[:1]  # plane metadata (time, position) of the first plane
        out_dtype = np.dtype(projection["dtype"])
        suffix = f"_z{zproject}"
        description = f"Z {zproject} projection of {projection['n_planes']} planes" + (
            ", flatfield-corrected before projection" if flatfield else ""
        )
    else:
        if flatfield is None:
            flatfield = bool(plate.attrs["phenix"].get("flatfield_default", False))
        if flatfield:
            out_dtype = OUTPUT_DTYPE
    codec, codec_args = _COMPRESSION[compression]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plate_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ctx.index.plate_name or ctx.index.plate_id or "plate")
    version = plate.attrs["phenix"].get("converter", {}).get("version", "")

    written = []
    reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ome-tiff-read")
    try:
        for well_path, entries in ctx.wells.items():
            t0 = time.perf_counter()
            well_name = entries[0][1].well_name
            name = f"{plate_name}_{well_name}{suffix}.ome.tiff"
            path = out_dir / name
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
            file_uuid = f"urn:uuid:{uuid.uuid4()}"
            xml = build_ome_xml(
                ctx.index,
                [
                    replace(f, images=[im for im in f.images if im.plane in plane_ids])
                    for _, f in entries
                ],
                ctx.channel_ids,
                plane_ids,
                ctx.timepoint_ids,
                PIXEL_TYPES.get(out_dtype.str.lstrip("<>|="), out_dtype.name),
                ctx.colors,
                ctx.tiff_values,
                ctx.assay_layout,
                creator=f"phenix-icechunk {version}",
                tiff_file=(name, file_uuid),
                image_description=description or None,
            )
            # OME-TIFF allows a UTF-8 ImageDescription. Passing bytes skips tifffile's ASCII
            # check. Character references ("&#181;m") are not an option: Bio-Formats mangles them.
            # Kept apart from `description`, which every well's <Image><Description> reuses.
            xml_bytes = xml.encode("utf-8")

            def read(group: str) -> np.ndarray:
                if zproject is not None:
                    da = open_field(plate, well_path, group, zproject=zproject)
                else:
                    da = open_field(plate, well_path, group, flatfield=flatfield)
                return to_dtype(da.values, out_dtype)

            partial = path.with_name(name + ".part")
            # Read field k+1 while field k is compressed and written.
            pending = reader.submit(read, entries[0][0])
            try:
                with tifffile.TiffWriter(partial, bigtiff=True) as tw:
                    for k in range(len(entries)):
                        data = pending.result()
                        if k + 1 < len(entries):
                            pending = reader.submit(read, entries[k + 1][0])
                        tw.write(
                            data,
                            photometric="minisblack",
                            tile=(tile, tile),
                            compression=codec,
                            compressionargs=codec_args,
                            maxworkers=os.cpu_count(),
                            description=xml_bytes if k == 0 else None,
                            metadata=None,
                            software=f"phenix-icechunk {version}",
                        )
            except BaseException:
                pending.cancel()
                partial.unlink(missing_ok=True)
                raise
            os.replace(partial, path)
            info = {
                "path": str(path),
                "well": well_name,
                "series": len(entries),
                "planes_per_series": int(np.prod(data.shape[:-2])),
                "bytes": path.stat().st_size,
                "flatfield": flatfield,
                "zproject": zproject,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            log.info("wrote %s", info)
            written.append(info)
    finally:
        reader.shutdown(wait=True)
    return written
