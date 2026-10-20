"""Store non-pixel metadata inside the icechunk repo.

icechunk only accepts zarr keys, so everything is kept as zarr nodes:

* tables -> a group with one 1-D array per column (strings as vlen UTF-8)
* files  -> 1-D ``uint8`` arrays holding the exact bytes, zstd-compressed
"""

from __future__ import annotations

import enum
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .index_xml import flatten, local_name

# StripOffsets / StripByteCounts become the virtual chunk refs themselves.
_SKIP_TAGS = {273, 279}
_FILE_CHUNK = 4 * 1024 * 1024
_TABLE_CHUNK = 65536


def tiff_tags(page) -> dict[str, Any]:
    """All tags of a tifffile page as JSON-friendly values, keyed by tag name (or code)."""
    out: dict[str, Any] = {}
    for tag in page.tags.values():
        if tag.code in _SKIP_TAGS:
            continue
        value = tag.value
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace").strip("\x00").strip()
        elif isinstance(value, enum.Enum):
            value = value.name
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, (tuple, np.ndarray)):
            value = np.asarray(value).tolist()
        elif isinstance(value, str):
            value = value.strip()
        out[tag.name if not tag.name.isdigit() else f"tag{tag.code}"] = value
    return out


def expand_tags(tags: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Flatten embedded XML / JSON tag values (Harmony TiffImageInfo, stats) into columns.

    The raw tag value is kept too, so nothing is lost if the flattening is lossy.
    """
    values: dict[str, Any] = {}
    units: dict[str, str] = {}
    for name, value in tags.items():
        values[name] = value
        if not isinstance(value, str):
            continue
        if value.startswith("<"):
            try:
                root = ET.fromstring(value)
            except ET.ParseError:
                continue
            label = local_name(root.tag) if name.startswith("tag") else name
            sub, sub_units = flatten(root)
            values.update({f"{label}.{k}": v for k, v in sub.items()})
            units.update({f"{label}.{k}": u for k, u in sub_units.items()})
        elif value.startswith("{"):
            try:
                sub = json.loads(value)
            except ValueError:
                continue
            if isinstance(sub, dict):
                values.update({f"{name}.{k}": v for k, v in sub.items()})
    return values, units


def _column(values: list[Any]) -> np.ndarray:
    present = [v for v in values if v is not None]
    missing = len(present) != len(values)
    if present and all(isinstance(v, bool) for v in present) and not missing:
        return np.array(values, dtype=bool)
    if present and all(isinstance(v, int) and not isinstance(v, bool) for v in present) and not missing:
        return np.array(values, dtype=np.int64)
    if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    return np.array(
        ["" if v is None else v if isinstance(v, str) else json.dumps(v) for v in values],
        dtype=object,
    )


def write_table(
    group: zarr.Group, rows: list[dict[str, Any]], units: dict[str, str], description: str
) -> None:
    """One array per column; read back with :func:`read_table`."""
    columns = list(dict.fromkeys(k for row in rows for k in row))
    for name in columns:
        data = _column([row.get(name) for row in rows])
        is_str = data.dtype == object
        arr = group.create_array(
            name,
            shape=data.shape,
            chunks=(max(1, min(len(data), _TABLE_CHUNK)),),
            dtype=str if is_str else data.dtype,
            fill_value="" if is_str else (np.nan if data.dtype.kind == "f" else 0),
        )
        arr[:] = data
        if name in units:
            arr.attrs["unit"] = units[name]
    group.attrs.update(
        {
            "table": {
                "description": description,
                "n_rows": len(rows),
                "columns": columns,
                "units": {c: units[c] for c in columns if c in units},
            }
        }
    )


def write_file(group: zarr.Group, node: str, data: bytes, info: dict[str, Any]) -> None:
    """Store a file's exact bytes at ``group[node]``; read back with :func:`read_file`."""
    arr = group.create_array(
        node,
        shape=(len(data),),
        chunks=(max(1, min(len(data), _FILE_CHUNK)),),
        dtype="uint8",
        fill_value=0,
    )
    if data:
        arr[:] = np.frombuffer(data, dtype=np.uint8)
    arr.attrs.update(
        {
            "file": {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                **info,
            }
        }
    )


def store_source_file(group: zarr.Group, path: Path, relpath: str) -> None:
    st = path.stat()
    write_file(
        group,
        relpath,
        path.read_bytes(),
        {
            "name": path.name,
            "relative_path": relpath,
            "source_path": str(path),
            "mtime": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(),
        },
    )


def read_file(node: zarr.Array) -> bytes:
    return node[:].tobytes()


def read_table(group: zarr.Group):
    """Return the table as a pandas DataFrame (or a dict of numpy arrays without pandas)."""
    columns = group.attrs["table"]["columns"]
    data = {c: group[c][:] for c in columns}
    try:
        import pandas as pd
    except ImportError:
        return data
    return pd.DataFrame(data, columns=columns)


__all__ = [
    "expand_tags",
    "read_file",
    "read_table",
    "store_source_file",
    "tiff_tags",
    "write_file",
    "write_table",
]
