import argparse
import json
import logging

from .convert import convert
from .zproject import METHODS


def main() -> None:
    p = argparse.ArgumentParser(
        description="Expose a Phenix/Harmony Index.xml export as a virtual OME-Zarr plate in icechunk."
    )
    p.add_argument("index_xml", help="path to Index.xml / Index.idx.xml (TIFFs next to it)")
    p.add_argument("repo", help="output icechunk repository directory")
    p.add_argument("--workers", type=int, default=32, help="threads for TIFF header scan (default 32)")
    p.add_argument(
        "--container-prefix",
        help="file:// URL prefix for the virtual chunk container (default: deepest URL-safe parent dir)",
    )
    p.add_argument(
        "--chunking",
        choices=("plane", "strip"),
        default="plane",
        help="plane: one whole-TIFF read per plane (fastest full-plane reads); "
        "strip: one chunk per TIFF strip (cheaper small-ROI reads)",
    )
    p.add_argument(
        "--flatfield",
        action="store_true",
        help="apply the Harmony flatfield profile: becomes the repo's read default "
        "(open_field/open_datatree) and is applied to --ome-tiff output; repo pixels stay raw",
    )
    p.add_argument(
        "--zproject",
        nargs="?",
        const="max",
        choices=sorted(METHODS),
        metavar="METHOD",
        help="project every field along z (after flatfield correction, if on) and commit it as a "
        "separate icechunk transaction under /projections/z<METHOD>; METHOD defaults to max "
        f"({', '.join(sorted(METHODS))}). Off unless given",
    )
    p.add_argument(
        "--ome-tiff",
        metavar="DIR",
        help="also write one Bio-Formats-compatible OME-TIFF per well (every field a tile/series) to DIR",
    )
    p.add_argument(
        "--ome-tiff-data",
        choices=("auto", "stack", "zproject", "both"),
        default="auto",
        help="what --ome-tiff writes: z stacks, the projection, or both "
        "(default auto: the projection if --zproject is given, else the stacks)",
    )
    p.add_argument(
        "--ome-tiff-compression",
        choices=("zlib", "lzw", "none"),
        default="zlib",
        help="compression of the OME-TIFF tiles (default zlib)",
    )
    p.add_argument("--overwrite", action="store_true", help="replace an existing repo / OME-TIFFs")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = convert(
        args.index_xml,
        args.repo,
        workers=args.workers,
        container_prefix=args.container_prefix,
        chunking=args.chunking,
        flatfield=args.flatfield,
        zproject=args.zproject,
        ome_tiff_dir=args.ome_tiff,
        ome_tiff_data=args.ome_tiff_data,
        ome_tiff_compression=args.ome_tiff_compression,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
