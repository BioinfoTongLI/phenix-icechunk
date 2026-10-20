"""Build an OME-XML (2016-06) document describing the plate, like bioformats2raw's
``OME/METADATA.ome.xml``, so OME tooling sees instrument, channel, plane and
plate-layout metadata in a standard form.

Image ``n`` in the XML is the ``n``-th field in ``fields`` (the same order the
converter writes them), and each field group records its ``ome_xml_image_id``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .index_xml import PhenixChannel, PhenixImage, PhenixIndex

OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
ANNOTATION_NS = "phenix-icechunk"
M_TO_UM = 1e6
# numpy dtype.str (without byte order) -> OME Pixels Type
PIXEL_TYPES = {"u1": "uint8", "u2": "uint16", "u4": "uint32", "i1": "int8", "i2": "int16",
               "i4": "int32", "f4": "float", "f8": "double"}

_ACQUISITION_MODE = {
    "NipkowConfocal": "SpinningDiskConfocal",
    "NonConfocal": "WideField",
}
_CONTRAST_METHOD = {"Fluorescence": "Fluorescence", "BrightField": "Brightfield", "Brightfield": "Brightfield"}
_ILLUMINATION = {"Epifluorescence", "Transmitted", "Oblique", "NonLinear"}
_BINNING = {1: "1x1", 2: "2x2", 4: "4x4", 8: "8x8"}
# Harmony TiffImageInfo DiagnosticData -> OME ImagingEnvironment
_ENV = "TiffImageInfo.DiagnosticData."


@dataclass(frozen=True)
class OmeField:
    """One field of view = one OME Image."""

    well_name: str  # e.g. "A4"
    row: int  # 1-based
    col: int  # 1-based
    field_id: int
    images: list[PhenixImage]
    physical_size_z_um: float


def _rgba_int(hex_rgb: str) -> int:
    value = (int(hex_rgb, 16) << 8) | 0xFF
    return value - (1 << 32) if value >= 1 << 31 else value


def _fmt(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def _map_annotation(parent: ET.Element, ann_id: str, namespace: str, values: dict[str, Any]) -> None:
    ann = ET.SubElement(parent, "MapAnnotation", ID=ann_id, Namespace=namespace)
    value = ET.SubElement(ann, "Value")
    for k, v in values.items():
        m = ET.SubElement(value, "M", K=k)
        m.text = "" if v is None else _fmt(v)


def _seconds_between(start: str, end: str) -> float | None:
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return None


def build_ome_xml(
    index: PhenixIndex,
    fields: list[OmeField],
    channel_ids: list[int],
    plane_ids: list[int],
    timepoint_ids: list[int],
    pixel_type: str,
    colors: dict[int, str],
    tiff_values: dict[str, dict[str, Any]],
    assay_layout: dict[str, Any] | None,
    creator: str,
    tiff_file: tuple[str, str] | None = None,
    image_description: str | None = None,
) -> str:
    """OME-XML for ``fields``. With ``tiff_file=(file_name, "urn:uuid:...")`` the pixels are
    described as an OME-TIFF where image ``i`` occupies IFDs ``i*P .. (i+1)*P-1`` (P = planes
    per image, XYZCT order); otherwise as ``MetadataOnly``."""
    ET.register_namespace("", OME_NS)
    ome = ET.Element("OME", {"xmlns": OME_NS, "Creator": creator})
    if tiff_file:
        ome.set("UUID", tiff_file[1])
    planes_per_image = len(channel_ids) * len(plane_ids) * len(timepoint_ids)
    channels: list[PhenixChannel] = [index.channels[c] for c in channel_ids]

    # --- annotations (collected first so elements can reference them) ---
    annotations: list[tuple[str, str, dict[str, Any]]] = []

    def annotation(namespace: str, values: dict[str, Any]) -> str:
        ann_id = f"Annotation:{len(annotations)}"
        annotations.append((ann_id, namespace, values))
        return ann_id

    plate_ann = annotation(
        f"{ANNOTATION_NS}/plate",
        {
            "IndexVersion": index.version,
            "User": index.user,
            "InstrumentType": index.instrument_type,
            "PlateID": index.plate_id,
            "PlateTypeName": index.plate_type,
            "MeasurementID": index.measurement_id,
            "MeasurementStartTime": index.measurement_start,
        },
    )
    channel_ann = {
        ch.id: annotation(
            f"{ANNOTATION_NS}/channel",
            {k: v for k, v in ch.entries.items() if k != "SkewcropParameters"}
            | {f"{k}.Unit": u for k, u in ch.units.items() if u},
        )
        for ch in channels
    }
    well_ann: dict[tuple[int, int], str] = {}
    if assay_layout:
        per_well: dict[tuple[int, int], dict[str, Any]] = {}
        for layer in assay_layout.get("layers", []):
            for w in layer.get("wells", []):
                per_well.setdefault((int(w["Row"]), int(w["Col"])), {})[str(layer.get("Name"))] = w.get("Value")
        for key in dict.fromkeys((f.row, f.col) for f in fields):
            if key in per_well:
                well_ann[key] = annotation(f"{ANNOTATION_NS}/assay-layout", per_well[key])

    # --- Plate ---
    rows = max(index.n_rows, max(f.row for f in fields))
    cols = max(index.n_cols, max(f.col for f in fields))
    plate = ET.SubElement(
        ome,
        "Plate",
        ID="Plate:0",
        Name=index.plate_name or index.plate_id,
        ExternalIdentifier=index.plate_id,
        Rows=str(rows),
        Columns=str(cols),
        RowNamingConvention="letter",
        ColumnNamingConvention="number",
    )
    if index.plate_type:
        ET.SubElement(plate, "Description").text = index.plate_type
    wells: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(fields):
        wells.setdefault((f.row, f.col), []).append(i)
    well_sample_ids: list[str] = []
    for w, ((row, col), image_indices) in enumerate(wells.items()):
        well = ET.SubElement(plate, "Well", ID=f"Well:0:{w}", Row=str(row - 1), Column=str(col - 1))
        for k, i in enumerate(image_indices):
            first = fields[i].images[0]
            ws_id = f"WellSample:0:{w}:{k}"
            well_sample_ids.append(ws_id)
            ws = ET.SubElement(
                well,
                "WellSample",
                ID=ws_id,
                Index=str(i),
                PositionX=str(first.pos_x_m * M_TO_UM),
                PositionXUnit="µm",
                PositionY=str(first.pos_y_m * M_TO_UM),
                PositionYUnit="µm",
            )
            ET.SubElement(ws, "ImageRef", ID=f"Image:{i}")
        if (row, col) in well_ann:
            ET.SubElement(well, "AnnotationRef", ID=well_ann[(row, col)])
    ET.SubElement(plate, "AnnotationRef", ID=plate_ann)
    times = sorted(im.abs_time for f in fields for im in f.images if im.abs_time)
    acquisition = ET.SubElement(
        plate,
        "PlateAcquisition",
        ID="PlateAcquisition:0:0",
        Name=index.measurement_id,
        MaximumFieldCount=str(max(len(v) for v in wells.values())),
        **({"StartTime": times[0], "EndTime": times[-1]} if times else {}),
    )
    for ws_id in well_sample_ids:
        ET.SubElement(acquisition, "WellSampleRef", ID=ws_id)

    # --- Experimenter / Instrument ---
    if index.user:
        ET.SubElement(ome, "Experimenter", ID="Experimenter:0", UserName=index.user)
    instrument = ET.SubElement(ome, "Instrument", ID="Instrument:0")
    ET.SubElement(instrument, "Microscope", Manufacturer="PerkinElmer", Model=index.instrument_type or "Phenix")
    detectors = {ch.camera_type for ch in channels}
    detector_id = {cam: f"Detector:0:{i}" for i, cam in enumerate(sorted(d or "" for d in detectors))}
    for cam, det_id in detector_id.items():
        ET.SubElement(instrument, "Detector", ID=det_id, **({"Model": cam} if cam else {}))
    objectives = {(ch.objective_magnification, ch.objective_na) for ch in channels}
    objective_id = {obj: f"Objective:0:{i}" for i, obj in enumerate(sorted(objectives, key=str))}
    for (mag, na), obj_id in objective_id.items():
        attrs = {"ID": obj_id}
        if mag is not None:
            attrs["NominalMagnification"] = str(mag)
        if na is not None:
            attrs["LensNA"] = str(na)
        ET.SubElement(instrument, "Objective", attrs)

    # --- Images ---
    ch0 = channels[0]
    for i, f in enumerate(fields):
        first = min(f.images, key=lambda im: im.abs_time or "~")
        image = ET.SubElement(ome, "Image", ID=f"Image:{i}", Name=f"{f.well_name}, Field {f.field_id}")
        if first.abs_time:
            ET.SubElement(image, "AcquisitionDate").text = first.abs_time
        if index.user:
            ET.SubElement(image, "ExperimenterRef", ID="Experimenter:0")
        if image_description:
            ET.SubElement(image, "Description").text = image_description
        ET.SubElement(image, "InstrumentRef", ID="Instrument:0")
        ET.SubElement(
            image, "ObjectiveSettings", ID=objective_id[(ch0.objective_magnification, ch0.objective_na)]
        )
        env = tiff_values.get(first.url, {})
        env_attrs = {}
        if isinstance(env.get(_ENV + "CurrentTemperature"), (int, float)):
            env_attrs |= {"Temperature": str(env[_ENV + "CurrentTemperature"]), "TemperatureUnit": "°C"}
        if isinstance(env.get(_ENV + "CurrentCO2"), (int, float)):
            env_attrs["CO2Percent"] = str(env[_ENV + "CurrentCO2"] / 100)
        if isinstance(env.get(_ENV + "CurrentHumidity"), (int, float)):
            env_attrs["Humidity"] = str(env[_ENV + "CurrentHumidity"] / 100)
        if env_attrs:
            ET.SubElement(image, "ImagingEnvironment", env_attrs)
        ET.SubElement(
            image,
            "StageLabel",
            Name=f"{f.well_name} field {f.field_id}",
            X=str(first.pos_x_m * M_TO_UM),
            XUnit="µm",
            Y=str(first.pos_y_m * M_TO_UM),
            YUnit="µm",
            Z=str(first.pos_z_m * M_TO_UM),
            ZUnit="µm",
        )
        pixels = ET.SubElement(
            image,
            "Pixels",
            ID=f"Pixels:{i}",
            DimensionOrder="XYZCT",
            Type=pixel_type,
            SizeX=str(ch0.size_x),
            SizeY=str(ch0.size_y),
            SizeZ=str(len(plane_ids)),
            SizeC=str(len(channel_ids)),
            SizeT=str(len(timepoint_ids)),
            PhysicalSizeX=str(ch0.resolution_x_m * M_TO_UM),
            PhysicalSizeXUnit="µm",
            PhysicalSizeY=str(ch0.resolution_y_m * M_TO_UM),
            PhysicalSizeYUnit="µm",
            PhysicalSizeZ=str(f.physical_size_z_um),
            PhysicalSizeZUnit="µm",
            BigEndian="false",
        )
        for c, ch in enumerate(channels):
            attrs = {"ID": f"Channel:{i}:{c}", "Name": ch.name, "SamplesPerPixel": "1"}
            illumination = ch.entries.get("IlluminationType")
            if illumination in _ILLUMINATION:
                attrs["IlluminationType"] = illumination
            if ch.acquisition_type:
                attrs["AcquisitionMode"] = _ACQUISITION_MODE.get(ch.acquisition_type, "Other")
            if ch.channel_type:
                attrs["ContrastMethod"] = _CONTRAST_METHOD.get(ch.channel_type, "Other")
            if ch.excitation_nm:
                attrs |= {"ExcitationWavelength": str(ch.excitation_nm), "ExcitationWavelengthUnit": "nm"}
            if ch.emission_nm:
                attrs |= {"EmissionWavelength": str(ch.emission_nm), "EmissionWavelengthUnit": "nm"}
            attrs["Color"] = str(_rgba_int(colors[ch.id]))
            channel = ET.SubElement(pixels, "Channel", attrs)
            ET.SubElement(
                channel,
                "DetectorSettings",
                ID=detector_id[ch.camera_type or ""],
                Binning=_BINNING.get(ch.binning_x, "Other") if ch.binning_x == ch.binning_y else "Other",
            )
            ET.SubElement(channel, "AnnotationRef", ID=channel_ann[ch.id])
        if tiff_file:
            tiff_data = ET.SubElement(
                pixels, "TiffData", IFD=str(i * planes_per_image), PlaneCount=str(planes_per_image)
            )
            ET.SubElement(tiff_data, "UUID", FileName=tiff_file[0]).text = tiff_file[1]
        else:
            ET.SubElement(pixels, "MetadataOnly")
        c_idx = {c: k for k, c in enumerate(channel_ids)}
        z_idx = {p: k for k, p in enumerate(plane_ids)}
        t_idx = {t: k for k, t in enumerate(timepoint_ids)}
        for im in sorted(f.images, key=lambda im: (im.timepoint, im.channel, im.plane)):
            attrs = {
                "TheZ": str(z_idx[im.plane]),
                "TheT": str(t_idx[im.timepoint]),
                "TheC": str(c_idx[im.channel]),
            }
            delta = _seconds_between(index.measurement_start, im.abs_time) if im.abs_time else None
            if delta is not None:
                attrs |= {"DeltaT": str(delta), "DeltaTUnit": "s"}
            exposure = index.channels[im.channel].exposure_s
            if exposure is not None:
                attrs |= {"ExposureTime": str(exposure), "ExposureTimeUnit": "s"}
            attrs |= {
                "PositionX": str(im.pos_x_m * M_TO_UM),
                "PositionXUnit": "µm",
                "PositionY": str(im.pos_y_m * M_TO_UM),
                "PositionYUnit": "µm",
                "PositionZ": str(im.pos_z_m * M_TO_UM),
                "PositionZUnit": "µm",
            }
            ET.SubElement(pixels, "Plane", attrs)

    structured = ET.SubElement(ome, "StructuredAnnotations")
    for ann_id, namespace, values in annotations:
        _map_annotation(structured, ann_id, namespace, values)

    ET.indent(ome)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(ome, encoding="unicode")
