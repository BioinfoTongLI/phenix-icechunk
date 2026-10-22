"""Default shard shape for the arrays this package writes as real chunks.

The virtual pixel arrays cannot be sharded: each of their chunks *is* a byte range of a
TIFF, and a shard needs its own index. Everything else (projections, tables, stored
files) is grouped into shards of about :data:`SHARD_TARGET_BYTES`, so a repo holds a
few large objects instead of many small ones.
"""

from __future__ import annotations

import math

SHARD_TARGET_BYTES = 50 * 1024 * 1024


def shards_for(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    itemsize: float,
    target_bytes: int = SHARD_TARGET_BYTES,
) -> tuple[int, ...] | None:
    """Shard shape of whole chunks holding about ``target_bytes`` uncompressed.

    Chunks are grouped in C order (last axis first), and an axis only grows once every
    later axis is covered, so each shard is one contiguous block of the array. Sizing
    uses uncompressed bytes, so compressed shards end up at or below the target.
    Returns ``None`` when a shard would hold a single chunk: sharding then adds nothing.
    """
    chunk_bytes = max(1.0, math.prod(chunks) * itemsize)
    budget = max(1, int(target_bytes // chunk_bytes))
    shards = list(chunks)
    for axis in reversed(range(len(shape))):
        n_chunks = math.ceil(shape[axis] / chunks[axis]) if shape[axis] else 1
        take = min(budget, n_chunks)
        shards[axis] = chunks[axis] * take
        budget //= take
        if take < n_chunks or budget == 1:
            break
    if tuple(shards) == tuple(chunks):
        return None
    return tuple(shards)
