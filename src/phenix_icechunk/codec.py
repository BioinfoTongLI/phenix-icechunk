"""Zarr v3 codec that decodes a single TIFF strip into a zarr chunk.

Harmony writes each plane as a TIFF with ``RowsPerStrip`` rows per strip, each
strip compressed independently.  Mapping one strip to one zarr chunk lets
icechunk point straight at the strip bytes inside the original file, so a read
is one ranged GET plus one decompress -- the same work tifffile does.

The last strip of a plane is usually shorter than ``RowsPerStrip`` (e.g.
2160 = 17 * 122 + 86).  A regular zarr chunk grid still expects a full chunk,
so short strips are zero-padded; zarr then clips the padding away because it
lies outside the array bounds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer

CODEC_NAME = "phenix_tiff_strip"

_DECODE = {
    "none": None,
    "lzw": imagecodecs.lzw_decode,
    "deflate": imagecodecs.zlib_decode,
    "zstd": imagecodecs.zstd_decode,
}
_ENCODE = {
    "none": None,
    "lzw": imagecodecs.lzw_encode,
    "deflate": imagecodecs.zlib_encode,
    "zstd": imagecodecs.zstd_encode,
}

# tifffile COMPRESSION enum value -> codec compression name
TIFF_COMPRESSION = {1: "none", 5: "lzw", 8: "deflate", 32946: "deflate", 50000: "zstd"}


@dataclass(frozen=True)
class TiffStripCodec(ArrayBytesCodec):
    """Array <-> bytes codec for one TIFF strip (chunk = ``(..., rows_per_strip, width)``)."""

    is_fixed_size = False

    compression: str = "lzw"
    predictor: int = 1  # 1 = none, 2 = horizontal differencing
    endian: str = "little"

    def __post_init__(self) -> None:
        if self.compression not in _DECODE:
            raise ValueError(f"unsupported compression {self.compression!r}")
        if self.predictor not in (1, 2):
            raise ValueError(f"unsupported predictor {self.predictor!r}")
        if self.endian not in ("little", "big"):
            raise ValueError(f"endian must be 'little' or 'big', got {self.endian!r}")

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, config = parse_named_configuration(data, CODEC_NAME, require_configuration=False)
        return cls(**(config or {}))

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": CODEC_NAME,
            "configuration": {
                "compression": self.compression,
                "predictor": self.predictor,
                "endian": self.endian,
            },
        }

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        data = chunk_bytes.as_numpy_array()
        decode = _DECODE[self.compression]
        if decode is not None:
            data = decode(data)
        dtype = chunk_spec.dtype.to_native_dtype()
        width = chunk_spec.shape[-1]
        rows = np.frombuffer(data, dtype=dtype.newbyteorder(self.endian))
        rows = rows[: rows.size - rows.size % width].reshape(-1, width)
        if self.predictor == 2:
            rows = np.cumsum(rows, axis=-1, dtype=dtype)
        chunk_rows = int(np.prod(chunk_spec.shape[:-1]))
        if rows.shape[0] == chunk_rows:
            out = rows.astype(dtype, copy=False).reshape(chunk_spec.shape)
        else:
            out = np.zeros(chunk_spec.shape, dtype=dtype)
            n = min(rows.shape[0], chunk_rows)
            out.reshape(-1, width)[:n] = rows[:n]
        return chunk_spec.prototype.nd_buffer.from_numpy_array(out)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        # imagecodecs releases the GIL, so threads give real parallel decoding.
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        arr = chunk_array.as_numpy_array()
        rows = arr.reshape(-1, arr.shape[-1])
        if self.predictor == 2:
            rows = np.diff(rows, axis=-1, prepend=np.zeros_like(rows[:, :1]))
        data = np.ascontiguousarray(rows, dtype=rows.dtype.newbyteorder(self.endian)).tobytes()
        encode = _ENCODE[self.compression]
        if encode is not None:
            data = encode(data)
        return chunk_spec.prototype.buffer.from_bytes(data)

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        return await asyncio.to_thread(self._encode_sync, chunk_array, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
