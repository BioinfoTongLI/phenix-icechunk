"""Check pixel equality and compare read speed: icechunk virtual plate vs. tifffile on the raw TIFFs.

Every method reads a *different* field, so no method benefits from the
page cache warmed by another.  Run twice to see warm-cache numbers.

    python bench/bench_read.py data/<repo>.icechunk [--fields 0 1 2 3 4 5 6]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import zarr

from phenix_icechunk import open_plate, parse_index


def tiff_paths(index, row, col, field):
    """(c, z) -> path for one field, ordered like the zarr array."""
    images = [im for im in index.images if (im.row, im.col, im.field) == (row, col, field)]
    cs = sorted({im.channel for im in images})
    zs = sorted({im.plane for im in images})
    grid = {(im.channel, im.plane): index.path.parent / im.url for im in images}
    return [[grid[(c, z)] for z in zs] for c in cs]


def read_tiffs(paths, workers, roi=None):
    flat = [p for row in paths for p in row]

    def load(p):
        a = tifffile.imread(p, maxworkers=1)
        return a if roi is None else a[roi]

    if workers == 1:
        planes = [load(p) for p in flat]
    else:
        with ThreadPoolExecutor(workers) as pool:
            planes = list(pool.map(load, flat))
    return np.stack(planes).reshape(len(paths), len(paths[0]), *planes[0].shape)


def timed(label, fn, nbytes):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label:<44} {dt:7.2f} s  {nbytes / dt / 1e6:8.1f} MB/s (decoded)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--fields", type=int, nargs="+", default=list(range(7)), help="field group indices to use")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=16, help="zarr async.concurrency")
    args = p.parse_args()
    zarr.config.set({"async.concurrency": args.concurrency})

    plate = open_plate(args.repo)
    index = parse_index(plate.attrs["phenix"]["index_xml"])
    well_meta = plate.attrs["ome"]["plate"]["wells"][0]
    well_path = well_meta["path"]
    row, col = well_meta["rowIndex"] + 1, well_meta["columnIndex"] + 1
    field_ids = sorted({im.field for im in index.images if (im.row, im.col) == (row, col)})
    f = iter(args.fields)

    def field(i):
        return plate[f"{well_path}/{i}/0"], tiff_paths(index, row, col, field_ids[i])

    arr, paths = field(next(f))
    print(f"array {arr.shape} {arr.dtype}, chunks {arr.chunks}, codecs {[c.to_dict()['name'] for c in arr.metadata.codecs]}, async.concurrency {args.concurrency}")
    full = arr.nbytes // arr.shape[0]
    roi = (slice(1000, 1512), slice(1000, 1512))
    roi_bytes = full // (arr.shape[-1] * arr.shape[-2]) * 512 * 512

    # --- correctness on one field (also warms nothing else) ---
    z = arr[0]
    ref = read_tiffs(paths, args.workers)
    assert np.array_equal(z, ref), "pixel mismatch between icechunk and tifffile"
    assert np.array_equal(arr[0, :, :, 1000:1512, 1000:1512], ref[:, :, 1000:1512, 1000:1512])
    assert np.array_equal(arr[0, :, :, -50:, :], ref[:, :, -50:, :]), "last (padded) strip mismatch"
    print("pixel check: OK (full field, ROI, last strip)\n")

    a, pth = field(next(f))
    timed("tifffile, sequential, full field", lambda: read_tiffs(pth, 1), full)
    a, pth = field(next(f))
    timed(f"tifffile, {args.workers} threads, full field", lambda: read_tiffs(pth, args.workers), full)
    a, pth = field(next(f))
    timed("icechunk zarr arr[0], full field", lambda: a[0], full)
    a, pth = field(next(f))
    timed(f"tifffile, {args.workers} threads, 512x512 ROI", lambda: read_tiffs(pth, args.workers, roi), roi_bytes)
    a, pth = field(next(f))
    timed("icechunk zarr, 512x512 ROI", lambda: a[0, :, :, roi[0], roi[1]], roi_bytes)
    a, pth = field(next(f))
    timed("icechunk zarr, one plane", lambda: a[0, 0, 0], full // (a.shape[1] * a.shape[2]))


if __name__ == "__main__":
    main()
