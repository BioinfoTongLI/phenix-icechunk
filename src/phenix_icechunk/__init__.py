from imagecodecs.zarr import Tiff
from zarr.registry import register_codec

from .codec import CODEC_NAME, TiffStripCodec
from .convert import convert, open_plate
from .flatfield import flatfield_profiles
from .index_xml import parse_assay_layout, parse_index
from .metadata import read_file, read_table
from .ome_tiff import export_ome_tiff
from .zproject import zproject
from .xarray_io import open_datatree, open_field

# Also registered via the "zarr.codecs" entry point; this covers non-installed use.
register_codec(CODEC_NAME, TiffStripCodec)
register_codec("imagecodecs_tiff", Tiff)

__all__ = [
    "TiffStripCodec",
    "convert",
    "export_ome_tiff",
    "flatfield_profiles",
    "open_datatree",
    "open_field",
    "open_plate",
    "parse_assay_layout",
    "parse_index",
    "read_file",
    "read_table",
    "zproject",
]
