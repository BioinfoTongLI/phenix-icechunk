"""Parse a Harmony / Opera Phenix ``Index.xml`` (``Index.idx.xml``) export.

Besides the typed fields the converter needs, every element is also kept as a
flat ``{dotted.key: value}`` dict (with a parallel ``{key: unit}`` dict for
``Unit=`` attributes), so fields we don't know about -- other Harmony versions,
FLIM, brightfield, ... -- still end up in the icechunk repo.
"""

from __future__ import annotations

import base64
import gzip
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def typed(text: str) -> Any:
    """'12' -> 12, '2E-06' -> 2e-06, 'true' -> True; ids like '0104' stay strings."""
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    try:
        value = int(text)
        return value if str(value) == text else text
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _decode_special(key: str, text: str) -> Any:
    """Decode the opaque blobs Harmony stores inside ``<Maps>``."""
    if key == "FlatfieldProfile":
        # Acapella writes a YAML flow mapping: {Background: {...}, Channel: 1, ...}
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text
    if key == "SkewcropParameters":
        # base64(gzip(Acapella EventXml))
        try:
            return gzip.decompress(base64.b64decode(text)).decode("utf-8")
        except (ValueError, OSError):
            return text
    if text.startswith("["):
        # OrientationMatrix, CropArea: JSON-style nested lists
        try:
            return json.loads(text)
        except ValueError:
            return text
    return typed(text)


def flatten(elem: ET.Element, prefix: str = "") -> tuple[dict[str, Any], dict[str, str]]:
    """Flatten an element's children to dotted keys; repeated keys become lists."""
    values: dict[str, list[Any]] = {}
    units: dict[str, str] = {}

    def put(key: str, value: Any) -> None:
        values.setdefault(key, []).append(value)

    for attr, value in elem.attrib.items():
        if attr != "Unit" and not prefix:
            put(f"@{attr}", typed(value))
    for child in elem:
        key = prefix + local_name(child.tag)
        if len(child):
            sub_values, sub_units = flatten(child, key + ".")
            for attr, value in child.attrib.items():
                put(f"{key}.@{attr}", typed(value))
            for k, v in sub_values.items():
                put(k, v)
            units.update(sub_units)
            continue
        text = (child.text or "").strip()
        attrs = {k: v for k, v in child.attrib.items() if k != "Unit"}
        if text:
            put(key, _decode_special(local_name(child.tag), text))
        elif len(attrs) == 1:
            put(key, typed(next(iter(attrs.values()))))
        elif attrs:
            put(key, {k: typed(v) for k, v in attrs.items()})
        else:
            put(key, None)
        if "Unit" in child.attrib:
            units[key] = child.attrib["Unit"]
    return {k: v[0] if len(v) == 1 else v for k, v in values.items()}, units


@dataclass(frozen=True)
class PhenixImage:
    """One ``<Image>``: a single (well, field, plane, timepoint, channel) TIFF."""

    row: int
    col: int
    field: int
    plane: int
    timepoint: int
    channel: int
    url: str
    ok: bool
    pos_x_m: float
    pos_y_m: float
    pos_z_m: float
    abs_time: str
    time_offset_s: float
    # Every child element of <Image>, flattened (includes the typed fields above).
    fields: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class PhenixChannel:
    id: int
    name: str
    size_x: int
    size_y: int
    resolution_x_m: float
    resolution_y_m: float
    excitation_nm: float | None
    emission_nm: float | None
    exposure_s: float | None
    acquisition_type: str | None
    channel_type: str | None
    binning_x: int | None
    binning_y: int | None
    camera_type: str | None
    objective_magnification: float | None
    objective_na: float | None
    # All <Map>/<Entry ChannelID=..> children merged (FlatfieldProfile parsed,
    # SkewcropParameters decoded), plus their units.
    entries: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    units: dict[str, str] = field(default_factory=dict, compare=False, hash=False)


@dataclass
class PhenixIndex:
    path: Path
    version: str
    user: str
    instrument_type: str
    plate_id: str
    plate_name: str
    plate_type: str
    measurement_id: str
    measurement_start: str
    n_rows: int
    n_cols: int
    plate: dict[str, Any] = field(default_factory=dict)
    plate_units: dict[str, str] = field(default_factory=dict)
    channels: dict[int, PhenixChannel] = field(default_factory=dict)
    images: list[PhenixImage] = field(default_factory=list)
    image_units: dict[str, str] = field(default_factory=dict)
    # Top-level elements other than Plates/Wells/Maps/Images (future-proofing).
    other: dict[str, Any] = field(default_factory=dict)


def parse_index(path: str | Path, data: bytes | None = None) -> PhenixIndex:
    """Parse ``path``, or ``data`` (the file's bytes, e.g. the copy stored in the repo) if given."""
    path = Path(path)
    root = ET.fromstring(data) if data is not None else ET.parse(path).getroot()
    # Harmony uses a default namespace whose version varies (HarmonyV5, V6, ...).
    ns = {"h": root.tag[1:].split("}", 1)[0]} if root.tag.startswith("{") else {"h": ""}

    plate_elem = root.find("h:Plates/h:Plate", ns)
    if plate_elem is None:
        raise ValueError(f"{path}: no <Plates>/<Plate> element")
    plate, plate_units = flatten(plate_elem)
    top_level, _ = flatten(root)
    index = PhenixIndex(
        path=path,
        version=str(root.get("Version", "")),
        user=str(top_level.get("User", "")),
        instrument_type=str(top_level.get("InstrumentType", "")),
        plate_id=str(plate.get("PlateID", "")),
        plate_name=str(plate.get("Name", "")),
        plate_type=str(plate.get("PlateTypeName", "")),
        measurement_id=str(plate.get("MeasurementID", "")),
        measurement_start=str(plate.get("MeasurementStartTime", "")),
        n_rows=int(plate.get("PlateRows") or 0),
        n_cols=int(plate.get("PlateColumns") or 0),
        plate=plate,
        plate_units=plate_units,
        other={
            k: v
            for k, v in top_level.items()
            if k.split(".", 1)[0] not in ("Plates", "Wells", "Maps", "Images")
        },
    )

    entries: dict[int, dict[str, Any]] = {}
    entry_units: dict[int, dict[str, str]] = {}
    for entry in root.iterfind("h:Maps/h:Map/h:Entry", ns):
        cid = int(entry.get("ChannelID"))
        values, units = flatten(entry)
        values.pop("@ChannelID", None)
        entries.setdefault(cid, {}).update(values)
        entry_units.setdefault(cid, {}).update(units)
    for cid, e in entries.items():
        if "ChannelName" not in e:
            continue

        def opt(key: str, cast=float):
            return cast(e[key]) if e.get(key) is not None else None

        index.channels[cid] = PhenixChannel(
            id=cid,
            name=str(e["ChannelName"]),
            size_x=int(e["ImageSizeX"]),
            size_y=int(e["ImageSizeY"]),
            resolution_x_m=float(e["ImageResolutionX"]),
            resolution_y_m=float(e["ImageResolutionY"]),
            excitation_nm=opt("MainExcitationWavelength"),
            emission_nm=opt("MainEmissionWavelength"),
            exposure_s=opt("ExposureTime"),
            acquisition_type=opt("AcquisitionType", str),
            channel_type=opt("ChannelType", str),
            binning_x=opt("BinningX", int),
            binning_y=opt("BinningY", int),
            camera_type=opt("CameraType", str),
            objective_magnification=opt("ObjectiveMagnification"),
            objective_na=opt("ObjectiveNA"),
            entries=e,
            units=entry_units[cid],
        )

    for img in root.iterfind("h:Images/h:Image", ns):
        f, units = flatten(img)
        index.image_units.update(units)
        index.images.append(
            PhenixImage(
                row=int(f["Row"]),
                col=int(f["Col"]),
                field=int(f["FieldID"]),
                plane=int(f["PlaneID"]),
                timepoint=int(f["TimepointID"]),
                channel=int(f["ChannelID"]),
                url=str(f["URL"]),
                ok=f.get("State", "Ok") == "Ok",
                pos_x_m=float(f.get("PositionX") or 0),
                pos_y_m=float(f.get("PositionY") or 0),
                pos_z_m=float(f.get("PositionZ") or 0),
                abs_time=str(f.get("AbsTime", "")),
                time_offset_s=float(f.get("MeasurementTimeOffset") or 0),
                fields=f,
            )
        )
    if not index.images:
        raise ValueError(f"{path}: no <Images>/<Image> entries")
    return index


def parse_assay_layout(path: str | Path) -> dict[str, Any]:
    """Harmony ``Assaylayout/*.xml``: named layers with a value per well."""
    root = ET.parse(path).getroot()
    out: dict[str, Any] = {"file": Path(path).name, "layers": []}
    for child in root:
        name = local_name(child.tag)
        if name != "Layer":
            out[name] = typed((child.text or "").strip())
            continue
        layer: dict[str, Any] = {"wells": []}
        for item in child:
            item_name = local_name(item.tag)
            if item_name == "Well":
                well, _ = flatten(item)
                layer["wells"].append(well)
            else:
                layer[item_name] = typed((item.text or "").strip())
        out["layers"].append(layer)
    return out
