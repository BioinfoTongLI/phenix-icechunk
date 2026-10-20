"""Build a virtual OME-Zarr (0.5) HCS plate in icechunk from a Phenix Index.xml.

No pixel data is copied.  Every zarr chunk is an icechunk *virtual reference*
to a byte range inside one of the original TIFFs:

* ``chunking="auto"`` (default): ``"tile"`` for tiled exports (newer Harmony exports
  write 512x512 tiles), ``"plane"`` for strip exports.
* ``chunking="plane"``: one chunk per plane referencing the *whole* TIFF file,
  decoded in memory by imagecodecs' ``imagecodecs_tiff`` codec. One large read per
  plane is what NFS-backed exports are fastest at for full-field reads.
* ``chunking="strip"`` (alias ``"tile"``): one chunk per TIFF strip *or* tile, whichever
  the files use, decoded by ``TiffStripCodec``. ~18-25x more (smaller) requests, but
  a sub-region read only touches its blocks.
* Uncompressed TIFFs with back-to-back strips always use one chunk per plane
  pointing at the pixel bytes, with the standard ``bytes`` codec.

All metadata is stored in the repo as well (see README "What is stored")::

    /                         plate   (ome.plate, phenix summary)
    /<row>/<col>              well    (ome.well, phenix well info + assay layout)
    /<row>/<col>/<field>      image   (ome.multiscales + omero, phenix field info)
    /<row>/<col>/<field>/0    array   (t, c, z, y, x), virtual
    /phenix                   group   (everything parsed from Index.xml / assay layout / TIFFs)
    /phenix/images/<column>   table   (one row per <Image>, Index.xml + all TIFF tags)
    /phenix/files/<relpath>   bytes   (every non-TIFF file of the export, verbatim)
    /OME/METADATA.ome.xml     bytes   (generated OME-XML)
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import icechunk as ic
import numpy as np
import tifffile
import zarr
from imagecodecs.zarr import Tiff as TiffCodec
from zarr.codecs import BytesCodec

from .codec import TIFF_COMPRESSION, TiffStripCodec
from .index_xml import PhenixImage, PhenixIndex, parse_assay_layout, parse_index
from .metadata import expand_tags, store_source_file, tiff_tags, write_file, write_table
from .ome_xml import PIXEL_TYPES, OmeField, build_ome_xml

log = logging.getLogger(__name__)

M_TO_UM = 1e6
TIFF_SUFFIXES = {".tif", ".tiff"}
MAX_SOURCE_FILE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class TiffLayout:
    """Everything that must be identical across all TIFFs to share one zarr array spec."""

    shape: tuple[int, int]
    dtype: str
    compression: str
    predictor: int
    endian: str
    tiled: bool
    # (rows, cols) of one TIFF strip or tile: the chunk shape in strip/tile mode
    block_shape: tuple[int, int]
    contiguous: bool


@dataclass(frozen=True)
class TiffStrips:
    layout: TiffLayout
    offsets: tuple[int, ...]
    counts: tuple[int, ...]
    file_size: int
    # Rounded up to whole seconds: icechunk refuses the chunk if the file was
    # modified after this, and its comparison is at second precision.
    last_modified: datetime
    mtime: datetime
    tags: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


def scan_tiff(path: Path) -> TiffStrips:
    """Read only the TIFF header/IFD: strip locations plus every tag."""
    st = os.stat(path)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages.first
        if page.samplesperpixel != 1:
            raise ValueError(f"{path}: expected 1 sample per pixel, got {page.samplesperpixel}")
        compression = TIFF_COMPRESSION.get(int(page.compression))
        if compression is None:
            raise ValueError(f"{path}: unsupported TIFF compression {page.compression!r}")
        layout = TiffLayout(
            shape=(page.imagelength, page.imagewidth),
            dtype=page.dtype.str.lstrip("<>|="),
            compression=compression,
            predictor=int(page.predictor),
            endian="little" if tif.byteorder == "<" else "big",
            tiled=page.is_tiled,
            block_shape=(
                (page.tilelength, page.tilewidth)
                if page.is_tiled
                else (min(page.rowsperstrip, page.imagelength), page.imagewidth)
            ),
            # back-to-back strips == plain row-major pixels; tiles never are
            contiguous=(
                not page.is_tiled
                and compression == "none"
                and int(page.predictor) == 1
                and bool(page.is_contiguous)
            ),
        )
        tags = tiff_tags(page)
        tags["n_pages"] = len(tif.pages)
        return TiffStrips(
            layout=layout,
            offsets=tuple(page.dataoffsets),
            counts=tuple(page.databytecounts),
            file_size=st.st_size,
            last_modified=datetime.fromtimestamp(math.ceil(st.st_mtime), UTC),
            mtime=datetime.fromtimestamp(st.st_mtime, UTC),
            tags=tags,
        )


def scan_tiffs(image_dir: Path, urls: list[str], workers: int) -> dict[str, TiffStrips | None]:
    """Scan headers in parallel; on network filesystems this is latency-bound, not CPU-bound."""

    def scan(url: str) -> tuple[str, TiffStrips | None]:
        path = image_dir / url
        if not path.exists():
            log.warning("missing TIFF %s, its chunks will read as fill_value", path)
            return url, None
        return url, scan_tiff(path)

    out: dict[str, TiffStrips | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (url, strips) in enumerate(pool.map(scan, urls), 1):
            out[url] = strips
            if i % 500 == 0 or i == len(urls):
                log.info("scanned %d/%d TIFF headers", i, len(urls))
    return out


def container_prefix_for(directory: Path) -> str:
    """Deepest ancestor of ``directory`` whose ``file://`` URL needs no percent-encoding.

    icechunk (2.3) cannot match chunk locations against a container prefix that
    contains percent-encoded characters, e.g. the space in ``...-Measurement 2``.
    """
    for d in (directory, *directory.parents):
        if d == d.parent:
            break
        if quote(str(d)) == str(d):
            return f"file://{d}/"
    raise ValueError(f"no URL-safe ancestor directory for {directory}; pass container_prefix")


def _row_name(row: int) -> str:
    name = ""
    while row > 0:
        row, rem = divmod(row - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _emission_color(nm: float | None) -> str:
    if nm is None:
        return "FFFFFF"
    for limit, color in ((480, "0000FF"), (550, "00FF00"), (600, "FFFF00"), (680, "FF0000")):
        if nm < limit:
            return color
    return "FF00FF"


def _epoch_seconds(timestamp: str) -> int | None:
    try:
        return int(datetime.fromisoformat(timestamp).timestamp())
    except ValueError:
        return None


def _plane_z_um(images: list[PhenixImage], channel_id: int, plane_ids: list[int]) -> list[float]:
    z_by_plane = {im.plane: im.pos_z_m for im in images if im.channel == channel_id}
    return [z_by_plane[p] * M_TO_UM for p in plane_ids if p in z_by_plane]


def _time_step_s(images: list[PhenixImage], timepoint_ids: list[int]) -> float | None:
    if len(timepoint_ids) < 2:
        return None
    offsets = {}
    for im in images:
        offsets.setdefault(im.timepoint, im.time_offset_s)
    steps = np.diff([offsets[t] for t in timepoint_ids if t in offsets])
    return float(statistics.median(steps)) if len(steps) else None


def _field_attrs(
    index: PhenixIndex,
    images: list[PhenixImage],
    name: str,
    channel_ids: list[int],
    plane_ids: list[int],
    timepoint_ids: list[int],
    dtype: str,
    colors: dict[int, str],
    phenix: dict[str, Any],
) -> dict:
    ch = index.channels[channel_ids[0]]
    zs = _plane_z_um(images, channel_ids[0], plane_ids)
    dz = statistics.median(np.diff(zs)) if len(zs) > 1 else 1.0
    dt = _time_step_s(images, timepoint_ids)
    # Stage Y increases upwards while image rows increase downwards (checked on
    # overlapping neighbour fields: x = +stageX, y = -stageY).
    pos_x_um = images[0].pos_x_m * M_TO_UM
    pos_y_um = images[0].pos_y_m * M_TO_UM
    dmax = int(np.iinfo(dtype).max) if np.dtype(dtype).kind in "ui" else 1
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": name,
                    "axes": [
                        {"name": "t", "type": "time", **({"unit": "second"} if dt else {})},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [
                                        dt or 1.0,
                                        1.0,
                                        float(dz),
                                        ch.resolution_y_m * M_TO_UM,
                                        ch.resolution_x_m * M_TO_UM,
                                    ],
                                },
                                {
                                    "type": "translation",
                                    "translation": [0.0, 0.0, zs[0] if zs else 0.0, -pos_y_um, pos_x_um],
                                },
                            ],
                        }
                    ],
                }
            ],
            "omero": {
                "name": name,
                "channels": [
                    {
                        "label": index.channels[c].name,
                        "color": colors[c],
                        "active": True,
                        "window": {"min": 0, "max": dmax, "start": 0, "end": dmax},
                    }
                    for c in channel_ids
                ],
            },
        },
        "phenix": phenix
        | {
            "stage_position_um": [pos_x_um, pos_y_um],
            "plane_z_um": zs,
            "physical_size_z_um": float(dz),
            "time_step_s": dt,
        },
    }


def _source_files(measurement_dir: Path) -> list[tuple[Path, str]]:
    """Every non-TIFF file of the export (Index.xml, assay layouts, markers, ...)."""
    out = []
    for dirpath, _, filenames in os.walk(measurement_dir):
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in TIFF_SUFFIXES:
                continue
            if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                log.warning("not storing %s: larger than %d bytes", path, MAX_SOURCE_FILE_BYTES)
                continue
            out.append((path, path.relative_to(measurement_dir).as_posix()))
    return out


def convert(
    index_path: str | Path,
    repo_path: str | Path,
    *,
    workers: int = 32,
    container_prefix: str | None = None,
    chunking: Literal["auto", "plane", "strip", "tile"] = "auto",
    flatfield: bool = False,
    zproject: str | None = None,
    ome_tiff_dir: str | Path | None = None,
    ome_tiff_data: Literal["auto", "stack", "zproject", "both"] = "auto",
    ome_tiff_compression: Literal["zlib", "lzw", "none"] = "zlib",
    overwrite: bool = False,
) -> dict:
    """Create an icechunk repo at ``repo_path`` holding a virtual OME-Zarr plate.

    ``flatfield=True`` makes flatfield correction the repo's read default (pixels in
    the repo stay raw/virtual) and applies it to the OME-TIFF export.
    ``zproject="max"`` (or min/mean/median/sum/std) projects every field along z after
    flatfield correction and commits the result as a second icechunk transaction under
    ``/projections/z<method>``. By default nothing is projected.
    ``ome_tiff_dir`` additionally writes one Bio-Formats-compatible OME-TIFF per well;
    ``ome_tiff_data`` picks the stacks, the projection, or both ("auto": the projection
    if ``zproject`` is set, else the stacks).
    """
    t0 = time.perf_counter()
    if ome_tiff_data == "auto":
        ome_tiff_data = "zproject" if zproject else "stack"
    if ome_tiff_data in ("zproject", "both") and not zproject:
        raise ValueError(f"ome_tiff_data={ome_tiff_data!r} needs zproject")
    index = parse_index(index_path)
    image_dir = index.path.resolve().parent
    measurement_dir = image_dir.parent if image_dir.name.lower() == "images" else image_dir
    images = [im for im in index.images if im.ok]
    log.info("parsed %s: %d images (%d not Ok)", index.path, len(images), len(index.images) - len(images))

    strips = scan_tiffs(image_dir, sorted({im.url for im in images}), workers)
    layouts = {s.layout for s in strips.values() if s is not None}
    if len(layouts) != 1:
        raise ValueError(f"TIFFs do not share a single layout: {layouts}")
    layout = layouts.pop()

    tiff_values: dict[str, dict[str, Any]] = {}
    tiff_units: dict[str, str] = {}
    for url, s in strips.items():
        if s is not None:
            tiff_values[url], units = expand_tags(s.tags)
            tiff_units.update(units)
    present = list(tiff_values.values())
    tiff_common = {
        k: v for k, v in present[0].items() if all(p.get(k) == v for p in present[1:])
    } if present else {}

    assay_layouts = {}
    for path in sorted((measurement_dir / "Assaylayout").glob("*.xml")):
        try:
            assay_layouts[path.name] = parse_assay_layout(path)
        except Exception:  # noqa: BLE001 - keep converting; the raw file is stored anyway
            log.warning("could not parse assay layout %s", path, exc_info=True)
    assay_layout = next(iter(assay_layouts.values()), None)
    assay_by_well: dict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
    for layer in (assay_layout or {}).get("layers", []):
        for w in layer.get("wells", []):
            assay_by_well[(int(w["Row"]), int(w["Col"]))][str(layer.get("Name"))] = w.get("Value")
    export_marker = measurement_dir / ".ExportDone_IndexAndImages"
    export_done = export_marker.read_text().strip() if export_marker.exists() else None

    channel_ids = sorted({im.channel for im in images})
    plane_ids = sorted({im.plane for im in images})
    timepoint_ids = sorted({im.timepoint for im in images})
    sizes = {(index.channels[c].size_y, index.channels[c].size_x) for c in channel_ids}
    if sizes != {layout.shape}:
        raise ValueError(f"channel image sizes {sizes} differ from TIFF shape {layout.shape}")
    c_idx = {c: i for i, c in enumerate(channel_ids)}
    z_idx = {p: i for i, p in enumerate(plane_ids)}
    t_idx = {t: i for i, t in enumerate(timepoint_ids)}
    colors = {c: _emission_color(index.channels[c].emission_nm) for c in channel_ids}

    by_field: dict[tuple[int, int, int], list[PhenixImage]] = defaultdict(list)
    for im in images:
        by_field[(im.row, im.col, im.field)].append(im)

    ny, nx = layout.shape
    if chunking == "auto":
        chunking = "tile" if layout.tiled else "plane"
        log.info("chunking=auto -> %s (%s TIFFs)", chunking, "tiled" if layout.tiled else "strip")
    if layout.contiguous:
        mode, chunk_yx, serializer = "pixels", (ny, nx), BytesCodec(endian=layout.endian)
    elif chunking == "plane":
        mode, chunk_yx, serializer = "file", (ny, nx), TiffCodec()
    elif chunking in ("strip", "tile"):
        mode, chunk_yx = ("tile" if layout.tiled else "strip"), layout.block_shape
        serializer = TiffStripCodec(
            compression=layout.compression, predictor=layout.predictor, endian=layout.endian
        )
    else:
        raise ValueError(f"chunking must be 'auto', 'plane', 'strip' or 'tile', got {chunking!r}")
    blocks_across = -(-nx // chunk_yx[1])
    shape = (len(timepoint_ids), len(channel_ids), len(plane_ids), ny, nx)

    repo_path = Path(repo_path)
    if repo_path.exists():
        if not overwrite:
            raise FileExistsError(f"{repo_path} exists; pass overwrite=True to replace it")
        shutil.rmtree(repo_path)
    prefix = container_prefix or container_prefix_for(image_dir)
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(prefix, ic.local_filesystem_store(prefix.removeprefix("file://")))
    )
    repo = ic.Repository.create(
        ic.local_filesystem_storage(str(repo_path)),
        config,
        authorize_virtual_chunk_access=ic.containers_credentials(
            {prefix: ic.Credentials.LocalFileSystemAccess()}
        ),
    )
    session = repo.writable_session("main")
    store = session.store

    wells = sorted({(r, c) for r, c, _ in by_field})
    n_rows = max(index.n_rows, max(r for r, _ in wells))
    n_cols = max(index.n_cols, max(c for _, c in wells))
    fields_per_well: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r, c, f in sorted(by_field):
        fields_per_well[(r, c)].append(f)
    times = sorted(im.abs_time for im in images if im.abs_time)
    converter = {
        "name": "phenix-icechunk",
        "version": version("phenix-icechunk"),
        "created": datetime.now(UTC).isoformat(),
        "icechunk": ic.__version__,
        "zarr": zarr.__version__,
        "tifffile": tifffile.__version__,
    }

    root = zarr.create_group(
        store,
        zarr_format=3,
        attributes={
            "ome": {
                "version": "0.5",
                "plate": {
                    "name": index.plate_name or index.plate_id,
                    "rows": [{"name": _row_name(r)} for r in range(1, n_rows + 1)],
                    "columns": [{"name": str(c)} for c in range(1, n_cols + 1)],
                    "wells": [
                        {"path": f"{_row_name(r)}/{c}", "rowIndex": r - 1, "columnIndex": c - 1}
                        for r, c in wells
                    ],
                    "field_count": max(len(v) for v in fields_per_well.values()),
                    "acquisitions": [
                        {
                            "id": 0,
                            "name": index.measurement_id,
                            "description": f"{index.instrument_type} measurement {index.measurement_start}",
                            "maximumfieldcount": max(len(v) for v in fields_per_well.values()),
                            **(
                                {"starttime": _epoch_seconds(times[0]), "endtime": _epoch_seconds(times[-1])}
                                if times
                                else {}
                            ),
                        }
                    ],
                },
            },
            "phenix": {
                "index_xml": str(index.path.resolve()),
                "measurement_dir": str(measurement_dir),
                "index_version": index.version,
                "user": index.user,
                "instrument_type": index.instrument_type,
                "plate_id": index.plate_id,
                "plate_name": index.plate_name,
                "plate_type": index.plate_type,
                "measurement_id": index.measurement_id,
                "measurement_start": index.measurement_start,
                "acquisition_start": times[0] if times else None,
                "acquisition_end": times[-1] if times else None,
                "export_done": export_done,
                "channels": [
                    {k: v for k, v in asdict(index.channels[c]).items() if k not in ("entries", "units")}
                    | {"color": colors[c]}
                    for c in channel_ids
                ],
                "tiff_layout": asdict(layout),
                "chunk_refs": mode,
                "container_prefix": prefix,
                # read-time default for open_field / open_datatree / export_ome_tiff
                "flatfield_default": flatfield,
                "converter": converter,
                "see_also": {
                    "full_metadata": "phenix",
                    "images_table": "phenix/images",
                    "source_files": "phenix/files",
                    "ome_xml": "OME/METADATA.ome.xml",
                },
            },
        },
    )

    n_refs = n_missing = 0
    ome_fields: list[OmeField] = []
    array_slot: dict[str, tuple[str, int, int, int]] = {}  # image id -> (array path, t, c, z)
    for (r, c), field_ids in fields_per_well.items():
        well_path = f"{_row_name(r)}/{c}"
        well = root.create_group(
            well_path,
            attributes={
                "ome": {
                    "version": "0.5",
                    "well": {"images": [{"path": str(i), "acquisition": 0} for i in range(len(field_ids))]},
                },
                "phenix": {
                    "well_id": f"{r:02d}{c:02d}",
                    "row": r,
                    "col": c,
                    "assay_layout": assay_by_well.get((r, c), {}),
                },
            },
        )
        for i, f in enumerate(field_ids):
            field_images = by_field[(r, c, f)]
            field_times = sorted(im.abs_time for im in field_images if im.abs_time)
            attrs = _field_attrs(
                index, field_images, f"{well_path} field {f}", channel_ids, plane_ids, timepoint_ids,
                layout.dtype, colors,
                {
                    "well_id": f"{r:02d}{c:02d}",
                    "field_id": f,
                    "ome_xml_image_id": f"Image:{len(ome_fields)}",
                    "acquisition_start": field_times[0] if field_times else None,
                    "acquisition_end": field_times[-1] if field_times else None,
                },
            )
            image = well.create_group(str(i), attributes=attrs)
            ome_fields.append(
                OmeField(
                    well_name=f"{_row_name(r)}{c}",
                    row=r,
                    col=c,
                    field_id=f,
                    images=field_images,
                    physical_size_z_um=attrs["phenix"]["physical_size_z_um"],
                )
            )
            image.create_array(
                "0",
                shape=shape,
                chunks=(1, 1, 1, *chunk_yx),
                dtype=layout.dtype,
                serializer=serializer,
                compressors=None,
                fill_value=0,
                dimension_names=("t", "c", "z", "y", "x"),
            )
            specs = []
            for im in field_images:
                t, ch, z = t_idx[im.timepoint], c_idx[im.channel], z_idx[im.plane]
                array_slot[im.fields["id"]] = (f"{well_path}/{i}/0", t, ch, z)
                s = strips.get(im.url)
                if s is None:
                    n_missing += 1
                    continue
                location = "file://" + quote(str(image_dir / im.url))
                if mode == "pixels":
                    specs.append(
                        ic.VirtualChunkSpec(
                            [t, ch, z, 0, 0], location, s.offsets[0], sum(s.counts),
                            last_updated_at_checksum=s.last_modified,
                        )
                    )
                elif mode == "file":
                    specs.append(
                        ic.VirtualChunkSpec(
                            [t, ch, z, 0, 0], location, 0, s.file_size,
                            last_updated_at_checksum=s.last_modified,
                        )
                    )
                else:
                    # TIFF blocks are row-major; edge tiles are padded to full size and
                    # zarr clips the padding, the short last strip is padded by the codec
                    specs.extend(
                        ic.VirtualChunkSpec(
                            [t, ch, z, *divmod(k, blocks_across)], location, off, n,
                            last_updated_at_checksum=s.last_modified,
                        )
                        for k, (off, n) in enumerate(zip(s.offsets, s.counts, strict=True))
                    )
            failed = store.set_virtual_refs(f"{well_path}/{i}/0", specs)
            if failed:
                raise RuntimeError(f"{well_path}/{i}: {len(failed)} refs rejected by container {prefix}")
            n_refs += len(specs)

    # --- /phenix: everything parsed, nothing summarised away ---
    meta = root.create_group(
        "phenix",
        attributes={
            "description": "All metadata of the Harmony export this plate was built from.",
            "index_xml": {
                "path": str(index.path.resolve()),
                "version": index.version,
                "top_level": index.other,
                "plate": index.plate,
                "plate_units": index.plate_units,
                "image_units": index.image_units,
                "channels": {
                    str(c): {"entries": ch.entries, "units": ch.units} for c, ch in index.channels.items()
                },
            },
            "assay_layouts": assay_layouts,
            "export_done": export_done,
            "tiff_layout": asdict(layout),
            "tiff_common_tags": tiff_common,
            "tiff_tag_units": tiff_units,
            "converter": converter,
        },
    )
    rows = []
    for im in index.images:
        slot = array_slot.get(im.fields["id"])
        s = strips.get(im.url)
        row: dict[str, Any] = dict(im.fields)
        row |= {
            "zarr_path": slot[0] if slot else None,
            "t_index": slot[1] if slot else -1,
            "c_index": slot[2] if slot else -1,
            "z_index": slot[3] if slot else -1,
            "file.size": s.file_size if s else None,
            "file.mtime": s.mtime.isoformat() if s else None,
        }
        row |= {f"tiff.{k}": v for k, v in tiff_values.get(im.url, {}).items()}
        rows.append(row)
    write_table(
        meta.create_group("images"),
        rows,
        index.image_units | {f"tiff.{k}": u for k, u in tiff_units.items()},
        "One row per <Image> of Index.xml: all its fields, where it lives in the zarr "
        "arrays (zarr_path, t/c/z_index) and every tag of its TIFF (tiff.*).",
    )
    files = meta.create_group("files", attributes={"root": str(measurement_dir)})
    sources = _source_files(measurement_dir)
    for path, relpath in sources:
        store_source_file(files, path, relpath)

    # --- /OME/METADATA.ome.xml: the same information in OME-XML form ---
    ome_xml = build_ome_xml(
        index,
        ome_fields,
        channel_ids,
        plane_ids,
        timepoint_ids,
        PIXEL_TYPES.get(layout.dtype, layout.dtype),
        colors,
        tiff_values,
        assay_layout,
        creator=f"phenix-icechunk {converter['version']}",
    )
    write_file(
        root.create_group("OME", attributes={"description": "Generated OME-XML for the plate"}),
        "METADATA.ome.xml",
        ome_xml.encode("utf-8"),
        {
            "name": "METADATA.ome.xml",
            "media_type": "application/xml",
            "generated_by": f"phenix-icechunk {converter['version']}",
        },
    )

    snapshot = session.commit(
        f"Virtual OME-Zarr plate from {index.path.resolve()}",
        metadata={"index_xml": str(index.path.resolve()), "measurement_id": index.measurement_id},
    )
    summary = {
        "repo": str(repo_path),
        "snapshot": snapshot,
        "wells": len(wells),
        "fields": len(by_field),
        "array_shape": shape,
        "chunk_shape": (1, 1, 1, *chunk_yx),
        "chunk_refs": mode,
        "virtual_refs": n_refs,
        "missing_tiffs": n_missing,
        "images_table": {"rows": len(rows), "columns": len({k for r in rows for k in r})},
        "source_files": [rel for _, rel in sources],
        "container_prefix": prefix,
        "layout": layout,
        "flatfield_default": flatfield,
    }
    if flatfield:
        has_profile = [
            isinstance(ch.entries.get("FlatfieldProfile"), dict) for ch in (index.channels[c] for c in channel_ids)
        ]
        if not any(has_profile):
            log.warning("--flatfield requested but Index.xml has no FlatfieldProfile; nothing will be corrected")
    # These modules import xarray_io, which imports this one, so import them here.
    if zproject:
        from .zproject import zproject as project

        # second transaction: the virtual plate above is already committed
        summary["zproject"] = project(repo_path, zproject, flatfield=flatfield, overwrite=overwrite)
    if ome_tiff_dir is not None:
        from .ome_tiff import export_ome_tiff

        plate = open_plate(repo_path)
        summary["ome_tiff"] = [
            info
            for kind in (("stack", "zproject") if ome_tiff_data == "both" else (ome_tiff_data,))
            for info in export_ome_tiff(
                plate,
                ome_tiff_dir,
                flatfield=flatfield if kind == "stack" else None,
                zproject=zproject if kind == "zproject" else None,
                compression=ome_tiff_compression,
                overwrite=overwrite,
            )
        ]
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    log.info("done: %s", summary)
    return summary


def open_repo(repo_path: str | Path) -> ic.Repository:
    """Open the icechunk repo, authorising its own file:// virtual chunk containers."""
    storage = ic.local_filesystem_storage(str(repo_path))
    config = ic.Repository.fetch_config(storage)
    prefixes = [
        cont.url_prefix
        for cont in (config.virtual_chunk_containers or {}).values()
        if cont.url_prefix.startswith("file://")
    ]
    return ic.Repository.open(
        storage,
        authorize_virtual_chunk_access=ic.containers_credentials(
            {p: ic.Credentials.LocalFileSystemAccess() for p in prefixes}
        ),
    )


def open_plate(
    repo_path: str | Path, *, branch: str = "main", snapshot_id: str | None = None
) -> zarr.Group:
    """Open the plate read-only."""
    repo = open_repo(repo_path)
    if snapshot_id is not None:
        session = repo.readonly_session(snapshot_id=snapshot_id)
    else:
        session = repo.readonly_session(branch=branch)
    return zarr.open_group(session.store, mode="r", zarr_format=3)
