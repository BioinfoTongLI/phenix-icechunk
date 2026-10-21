# phenix-icechunk

Expose an Opera Phenix / Harmony `Index.xml` export as a **virtual OME-Zarr 0.5 HCS plate**
inside an [icechunk](https://icechunk.io) repo. No pixels are copied: every zarr chunk is a
virtual reference into the original TIFFs, so the repo is a few hundred KB.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

## Convert

```bash
.venv/bin/phenix-icechunk "/path/to/Measurement 2/Images/Index.xml" data/plate.icechunk

# + flatfield correction by default, + Bio-Formats OME-TIFFs (one per well) next to the repo
.venv/bin/phenix-icechunk "/path/to/Measurement 2/Images/Index.xml" data/plate.icechunk \
    --flatfield --ome-tiff data/ome_tiff

# + max z projection (after flatfield), committed as its own transaction, exported as OME-TIFF
.venv/bin/phenix-icechunk "/path/to/Measurement 2/Images/Index.xml" data/plate.icechunk \
    --flatfield --zproject --ome-tiff data/ome_tiff        # --zproject mean|median|min|sum|std
```

| Option | Meaning |
|---|---|
| `--flatfield` | Make Harmony flatfield correction the repo's read default (`open_field`/`open_datatree` apply it unless `flatfield=False`) and apply it to `--ome-tiff`. Repo pixels stay raw and virtual |
| `--zproject [METHOD]` | Project every field along z, **after** flatfield correction if on, and commit it as a separate icechunk transaction under `/projections/z<METHOD>`. Bare flag = `max`; also `min`, `mean`, `median`, `sum`, `std`. Off by default |
| `--ome-tiff DIR` | Also write one OME-TIFF per well to `DIR` (see below) |
| `--ome-tiff-data auto\|stack\|zproject\|both` | What `--ome-tiff` writes. `auto` (default) = the projection if `--zproject` is given, else the z stacks |
| `--ome-tiff-compression zlib\|lzw\|none` | Tile compression of the OME-TIFFs (default `zlib`, level 1) |
| `--chunking auto\|plane\|strip\|tile` | Virtual chunk layout (see "How it works"). Default `auto` = `tile` for tiled exports, `plane` for strip exports. `strip` and `tile` are the same mode |
| `--workers N` | Threads for the TIFF header scan (default 32) |
| `--container-prefix URL` | `file://` prefix for the virtual chunk container (default: deepest URL-safe parent) |
| `--overwrite` | Replace an existing repo / OME-TIFFs |

## Read as xarray

```python
import zarr
from phenix_icechunk import open_plate, open_field, open_datatree

zarr.config.set({"async.concurrency": 16})        # default 10; 16 was best on /nfs
plate = open_plate("data/plate.icechunk")         # open once, reuse for many fields

da = open_field(plate, "A4", 0)                   # lazy xr.DataArray, nothing read yet
dapi = da.sel(c="DAPI").isel(t=0).values          # reads only the 12 DAPI planes
roi = da.sel(x=slice(1900, 2000), y=slice(-7100, -7000))  # select in µm
da.attrs["plate_id"], da.attrs["assay_layout"], da.coords["acquisition_time"]

da = open_field(plate, "A4", 0, chunks={})        # same, dask-backed (needs dask)

tree = open_datatree(plate)                       # whole plate as xr.DataTree, all lazy
tree["A/4/17"]["image"]                           # field 17 of well A4
tree["phenix/images"]                             # per-image table as a Dataset
```

`open_field` returns dims `(t, c, z, y, x)` with:

* **coords**:
  * `c`: channel names, plus `channel_id`, excitation/emission wavelength, exposure time,
    acquisition type, binning, camera, objective, colour.
  * `z`: plane z in µm, plus `plane_id`.
  * `y`, `x`: µm in stage space. This is the same transform as the OME-Zarr metadata, so neighbouring fields line up.
  * `t`: seconds, plus `timepoint_id`.
  * Per-plane `(t, c, z)` coords: `image_id`, `acquisition_time` (datetime64, UTC), `abs_position_z`,
    `intensity_mean/stddev/min/max`, `temperature`, `co2`, `humidity`. Pick other table
    columns with `plane_coords={"name": "column", ...}`.
* **attrs**: well (ID, row/col, assay layout), field (ID, stage position, acquisition
  start/end, OME-XML image ID), plate (IDs, type, measurement, user, instrument, source
  Index.xml), and the field's full OME-Zarr metadata under `attrs["ome"]`.

Reads through xarray cost the same as raw zarr: about 0.35 s for a full 4×12-plane field.
Nothing is decoded to float, because fill values are not applied, so pixels stay `uint16`.

### Flatfield correction

```python
from phenix_icechunk import flatfield_profiles

da = open_field(plate, "A4", 0, flatfield=True)   # still lazy, float32
# flatfield=None (default) follows the repo: True if it was converted with --flatfield
da.attrs["flatfield"]                             # formula + per-channel profile summary
tree = open_datatree(plate, flatfield=True)       # same for every field
flatfield_profiles(plate)                         # the (c, y, x) profiles themselves
```

Each channel's Harmony `FlatfieldProfile` (`Background`, an Acapella 2-D polynomial) is
evaluated as `P = Σ Coefficients[k][j] · u^(k−j) · v^j`, with
`u = (x − Origin₀)·Scale₀` and `v = (y − Origin₁)·Scale₁`. It is normalised to mean 1, and
pixels become `raw / P`, so intensities keep their scale.

* The correction is applied on read, per plane. Nothing in the repo changes, and it costs
  about 0–0.1 s per full field on top of the 0.5 s read.
* The term order was checked against the data. The median of all 56 fields per channel
  correlates with `P` at 0.95–0.98, against 0.52–0.81 for the transposed order. After
  correction, background non-flatness drops from 3.6–6.3 % to 0.9–2.1 %, with the mean unchanged.
* Channels without a polynomial (e.g. `Character: Flat`) are left as they are
  (`attrs["flatfield"]["channels"][name]["applied"] == False`).
* If the image size differs from the profile's `Dims` (binning), the profile is rescaled.

### Z projection

```python
from phenix_icechunk import zproject

zproject("data/plate.icechunk", "mean")            # add another method later: one more commit
mip = open_field(plate, "A4", 0, zproject="max")   # (t, c, 1, y, x), attrs["zproject"]
tree = open_datatree(plate, zproject="max")
```

* Projections are real pixels (they can't be virtual). They are stored as a full OME-Zarr HCS
  plate under `/projections/z<method>/<row>/<col>/<field>/0` (z size 1, z = mid-stack),
  and listed in root `attrs["phenix"]["projections"]`.
* Each projection is **its own icechunk commit** on top of the virtual plate, e.g.
  `Z max projection of 56 fields after flatfield correction`. The plate commit is never
  touched, so a failed or replaced projection leaves it intact. Re-running a method
  needs `overwrite=True`, and the old version stays in history.
* Order: `project(flatfield(raw))`. The stored projection already has the correction, so
  `open_field(..., zproject=...)` refuses a conflicting `flatfield=` argument.
* dtype: `max`/`min` are stored as `uint16` (rounded and clipped, with or without flatfield);
  `mean/median/sum/std` are `float32`. A warning is logged if any pixel exceeds the `uint16`
  range. For this plate the max projection took 40 s to compute (3.3 GB in the repo as float32).

### OME-TIFF export (Bio-Formats / Fiji / QuPath / OMERO)

`--ome-tiff DIR` on convert, or later from any repo:

```python
from phenix_icechunk import export_ome_tiff
export_ome_tiff("data/plate.icechunk", "data/ome_tiff", flatfield=True, compression="zlib")
export_ome_tiff("data/plate.icechunk", "data/ome_tiff", zproject="max")  # -> <plate>_<well>_zmax.ome.tiff
```

The output is one `<plate>_<well>.ome.tiff` per well:

* Every field of view is its own series (tile): `XYZCT` planes, 512×512 internal tiles, BigTIFF.
* The OME-XML in the first IFD carries the Plate/Well/WellSample layout with stage positions,
  channels (names, wavelengths, modes, colours), every Plane (ΔT, exposure, XYZ), StageLabel,
  ImagingEnvironment, instrument, and assay-layout/channel MapAnnotations. It is the
  same document as `/OME/METADATA.ome.xml`, restricted to the well, plus `TiffData` per image.
* Pixels are `uint16`. With flatfield, corrected pixels are rounded and clipped to `uint16`;
  `zproject=` exports use the projection's stored dtype (`uint16` for `max`/`min`).

Checked with Bio-Formats 8.1.1 `showinf`: the file opens as OME-TIFF with 56 series of
Z=12 × C=4 × 2160², and the plate/well/well-sample positions are parsed. Planes extracted with
`bfconvert` are bit-identical to the corrected repo planes. The well above (25 GB raw) took
about 100 s to write and is 10.5 GB with zlib.

### Low-level (zarr)

```python
img = plate["A/4/0/0"]                           # zarr.Array, (t, c, z, y, x)
stack = img[0]                                   # numpy
```

`open_plate` grants icechunk access to the repo's own `file://` container. Opening the
repo with plain `icechunk.Repository.open` also works, but you have to pass
`authorize_virtual_chunk_access` yourself.

```python
from phenix_icechunk import read_file, read_table

df = read_table(plate["phenix/images"])                 # one row per <Image>, 76 columns
xml = read_file(plate["phenix/files/Images/Index.xml"]) # original bytes
ome = read_file(plate["OME/METADATA.ome.xml"]).decode() # generated OME-XML
plate["phenix"].attrs["index_xml"]["channels"]["1"]     # every Map entry for channel 1
```

## What is stored

Every piece of metadata in the export is stored in the repo, both verbatim and parsed:

| Where | What |
|---|---|
| `/` attrs `ome.plate` | OME-Zarr 0.5 plate: rows/columns/wells, acquisition (measurement ID, start/end time, max field count) |
| `/` attrs `phenix` | Summary: plate/measurement IDs, user, instrument, start/end/export times, channels, TIFF layout, converter versions |
| `/<row>/<col>` attrs | `ome.well` + `phenix`: Harmony well ID and the well's **assay-layout** values (Compound, Concentration, …) |
| `/<row>/<col>/<field>` attrs | `ome.multiscales` (scale, z step, stage translation, time step if T>1), `omero` channel names and colours, `phenix`: field ID, stage position, per-plane z, acquisition start/end, matching OME-XML image ID |
| `/phenix` attrs | Index.xml in full: top-level fields, plate block and units, **every `<Map>` entry per channel** (FlatfieldProfile parsed from Acapella YAML, SkewcropParameters decoded from base64-gzip, OrientationMatrix, CropArea, binning, camera, exposure, excitation power, …); parsed assay layouts; TIFF tags shared by all files |
| `/phenix/images/<column>` | Table with one row per `<Image>` (including non-`Ok` ones): every Index.xml field (IDs, positions, AbsPositionZ, AbsTime, time offset, …), where the plane lives (`zarr_path`, `t/c/z_index`), file size and mtime, and **every TIFF tag**. The private Harmony tags are flattened: `tiff.TiffImageInfo.*` (RecordID, ExposureID, absolute XYZ, temperature, CO₂, humidity, regulation states), `tiff.tag65520.*` (per-plane mean/stddev/min/max), FlexXML. Raw XML/JSON is kept alongside, units go in column attrs |
| `/phenix/files/<relpath>` | Every non-TIFF file of the export, byte-for-byte, with sha256, mtime and source path in attrs (`Images/Index.xml`, `Assaylayout/*.xml`, `.ExportDone_IndexAndImages`) |
| `/OME/METADATA.ome.xml` | Generated OME-XML (validates against the 2016-06 XSD): Plate/Well/WellSample/PlateAcquisition, Instrument (microscope, objective, camera), one Image per field with Channels (modes, wavelengths, binning), every Plane (DeltaT, exposure, XYZ), StageLabel, ImagingEnvironment from TIFF diagnostics, and MapAnnotations for channel settings and assay layout |

The plate, well and field metadata validate with `ome-zarr-models` (v0.5).
Only the pixel strip offsets are not stored as metadata, because they *are* the virtual chunk refs.

## How it works

* `Index.xml` gives the well/field/plane/channel/timepoint of every TIFF. Each field becomes
  one `(t, c, z, y, x)` array with OME-Zarr `multiscales` (scale from `ImageResolution`, z step
  from `PositionZ`, translation from stage position: x = +stageX, y = −stageY).
* TIFF headers are scanned in parallel (about 3 ms per file on /nfs) to get strip offsets and file sizes.
* Harmony writes two TIFF layouts, and both are supported:
  * LZW **strips**, e.g. 18 strips of 122 rows (OT_NIM).
  * LZW **512×512 tiles** in newer exports, e.g. 25 tiles per 2160² plane (large_GBM).
* `--chunking auto` (default) picks **tile** for tiled exports and **plane** for strip exports.
* **plane** chunking: one chunk per plane, referencing the whole TIFF file and
  decoded by imagecodecs' `imagecodecs_tiff` codec (libtiff). That is one large read per plane,
  which is what NFS serves fastest, and it works the same for strips and tiles.
* **strip / tile** chunking: one chunk per TIFF strip or tile, whichever the files use, decoded
  by the bundled `phenix_tiff_strip` codec. The short last strip is zero-padded by the codec,
  and padded edge tiles are clipped by zarr. Partial reads touch fewer bytes, but the request
  count goes up 18–25×.
* Uncompressed TIFFs with contiguous strips get a plain `bytes`-codec chunk per plane.
* Each reference stores the TIFF's mtime, so icechunk raises an error rather than return wrong
  pixels if a source file is rewritten.

The codecs are registered through the `zarr.codecs` entry point, so any zarr-python 3 reader in
this environment can decode the chunks. Readers elsewhere need `imagecodecs` (plane mode) or this
package (strip mode).

## Benchmark (`bench/bench_read.py`)

The measurement has 56 fields × 4 channels × 12 z planes of 2160² uint16, LZW-compressed. The table
shows one field (448 MB decoded); each method read a different field from a cold cache.

| method                              | plane repo | strip repo |
|-------------------------------------|-----------:|-----------:|
| tifffile, sequential                |     2.2 s  |     2.6 s  |
| tifffile, 16 threads                |    0.66 s  |    0.53 s  |
| **icechunk zarr `img[0]`**          | **0.39 s** |    1.05 s  |
| tifffile, 16 threads, 512² ROI      |    0.32 s  |    0.34 s  |
| icechunk zarr, 512² ROI             |    0.41 s  |    0.53 s  |

Strip chunking does not pay off on this NFS mount, because per-request latency dominates.
It may help on object storage or for viewers that fetch small tiles.

## Notes / limitations

* icechunk 2.3 cannot match a virtual-chunk container whose prefix needs percent-encoding
  (for example the space in `Measurement 2`), so the container is set at the deepest URL-safe parent
  directory.
* Only one resolution level is exposed; pyramids would have to be materialised.
* Flatfield correction is read-time only (`flatfield=True`) and uses the `Background`
  profile. No darkfield/offset is subtracted, because Harmony exports here carry no separate
  darkfield, and the `Background` profile already describes the raw background including the camera offset.
* icechunk only accepts zarr keys, so files are stored as `uint8` arrays, not plain files.
  Tools expecting bioformats2raw's plain `OME/METADATA.ome.xml` file need `read_file`.
* All channels must share image size, dtype and TIFF layout. Otherwise conversion stops
  with an error.
