"""Best-effort EXIF, XMP, IPTC and ICC extraction with GPS redaction."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import ExifTags, Image, ImageCms

EXIF_FIELDS = {
    "Make",
    "Model",
    "LensModel",
    "DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "PhotographicSensitivity",
    "FocalLength",
    "Flash",
    "Orientation",
    "Software",
    "Artist",
    "Copyright",
    "ImageDescription",
    "UserComment",
}
AI_MARKERS = re.compile(
    r"(?:stable diffusion|midjourney|dall.?e|firefly|comfyui|generative fill|text.?to.?image)",
    re.IGNORECASE,
)
EDIT_MARKERS = re.compile(
    r"(?:photoshop|lightroom|gimp|affinity|capture one|pixelmator)", re.IGNORECASE
)


def _safe_value(value: object) -> object:
    if isinstance(value, bytes):
        return value[:512].decode("utf-8", errors="replace").strip("\x00")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:1000] if isinstance(value, str) else value
    if isinstance(value, (tuple, list)):
        return [_safe_value(v) for v in value[:32]]
    try:
        return str(value)[:256]
    except Exception:  # noqa: BLE001 - malformed metadata values must not break report generation.
        return "[unavailable]"


def _parse_xmp(image: Image.Image) -> dict[str, str]:
    raw = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
    if not raw:
        return {}
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")
    try:
        root = ET.fromstring(raw[:1_000_000])
    except (ET.ParseError, ValueError):
        return {"parse_status": "present_but_unreadable"}
    result: dict[str, str] = {}
    location_present = False
    for node in root.iter():
        if node.text and node.text.strip():
            key = node.tag.rsplit("}", 1)[-1]
            if re.search(
                r"(?:gps|latitude|longitude|geotag|geolocation|location)",
                key,
                re.IGNORECASE,
            ):
                location_present = True
                continue
            if key not in {"RDF", "Description", "li"}:
                result[key[:80]] = node.text.strip()[:500]
        for key, value in node.attrib.items():
            local = key.rsplit("}", 1)[-1]
            if re.search(
                r"(?:gps|latitude|longitude|geotag|geolocation|location)",
                local,
                re.IGNORECASE,
            ):
                location_present = True
                continue
            if local not in {"about", "parseType"}:
                result[local[:80]] = str(value)[:500]
        if len(result) >= 100:
            break
    if location_present:
        result["location_metadata_present"] = "true; coordinate values redacted"
    return result


def _exiftool_metadata(raw_bytes: bytes | None) -> dict[str, object]:
    """Use ExifTool when installed, with a random temporary file and strict timeout."""
    executable = shutil.which("exiftool")
    if not executable or raw_bytes is None:
        return {"available": bool(executable), "used": False}
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="image-forensics-", suffix=".bin", delete=False
        ) as temporary:
            temporary.write(raw_bytes)
            path = Path(temporary.name)
        completed = subprocess.run(
            [executable, "-json", "-G1", "-n", "-q", str(path)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode != 0:
            return {"available": True, "used": False, "status": "tool_error"}
        entries = json.loads(completed.stdout)
        if not entries or not isinstance(entries[0], dict):
            return {"available": True, "used": False, "status": "empty_result"}
        selected: dict[str, object] = {}
        location_present = False
        for raw_key, value in entries[0].items():
            group, _, key = str(raw_key).partition(":")
            if re.search(
                r"(?:gps|latitude|longitude|geotag|geolocation|location)",
                key,
                re.IGNORECASE,
            ):
                location_present = True
                continue
            if not key:
                continue
            group_upper = group.upper()
            if "EXIF" in group_upper or group_upper.startswith(
                ("XMP", "IPTC", "ICC_PROFILE", "PHOTOSHOP")
            ):
                selected[f"{group}:{key}"] = _safe_value(value)
            if len(selected) >= 100:
                break
        return {
            "available": True,
            "used": True,
            "location_present": location_present,
            "tags": selected,
        }
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"available": True, "used": False, "status": "unavailable_or_timeout"}
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _parse_iptc(image: Image.Image) -> dict[str, object]:
    raw = image.info.get("photoshop")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {
            str(key)[:80]: _safe_value(value) for key, value in list(raw.items())[:50]
        }
    return {"present": True}


def _parse_icc(image: Image.Image) -> dict[str, object]:
    profile = image.info.get("icc_profile")
    if not profile:
        return {}
    result: dict[str, object] = {"byte_length": len(profile)}
    try:
        parsed = ImageCms.ImageCmsProfile(BytesIO(profile))
        result["profile_name"] = ImageCms.getProfileName(parsed).strip()[:200]
        result["profile_description"] = ImageCms.getProfileDescription(parsed).strip()[
            :300
        ]
        color_space = getattr(parsed.profile, "xcolor_space", None)
        if color_space:
            result["color_space"] = str(color_space).strip()[:80]
    except (ImageCms.PyCMSError, OSError, TypeError, ValueError):
        result["parse_status"] = "present_but_unreadable"
    return result


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        # EXIF timestamps are local wall-clock values and do not carry a timezone.
        return datetime.strptime(value[:19], "%Y:%m:%d %H:%M:%S")  # noqa: DTZ007
    except ValueError:
        return None


def _bit_depth(image: Image.Image) -> int | None:
    if image.mode == "1":
        return 1
    if image.mode.startswith("I;16"):
        return 16
    if image.mode in {"I", "F"}:
        return 32
    if image.mode in {"L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV"}:
        return 8
    return None


def extract_metadata(
    image: Image.Image,
    filename: str,
    mime_type: str,
    byte_size: int,
    raw_bytes: bytes | None = None,
) -> dict:
    """Extract common metadata and safe file properties from a decoded image."""
    exif_data: dict[str, object] = {}
    gps_present = False
    try:
        exif = image.getexif()
        all_tags = dict(exif.items())
        try:
            all_tags.update(exif.get_ifd(ExifTags.IFD.Exif))
        except (AttributeError, KeyError, TypeError):
            pass
        try:
            gps_present = bool(exif.get_ifd(ExifTags.IFD.GPSInfo))
        except (AttributeError, KeyError, TypeError):
            gps_present = 34853 in all_tags
        for tag_id, value in all_tags.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if name == "GPSInfo":
                gps_present = bool(value)
            elif name in EXIF_FIELDS:
                exif_data[name] = _safe_value(value)
    except Exception:  # noqa: BLE001 - partially corrupt EXIF should not prevent image analysis.
        exif_data = {}
    has_exif_tags = bool(exif_data)
    if has_exif_tags or gps_present:
        exif_data["GPSPresent"] = gps_present

    xmp = _parse_xmp(image)
    iptc = _parse_iptc(image)
    icc = _parse_icc(image)
    exiftool = _exiftool_metadata(raw_bytes)
    for combined_key, value in exiftool.get("tags", {}).items():
        group, key = combined_key.split(":", 1)
        if "EXIF" in group.upper() and key in EXIF_FIELDS and key not in exif_data:
            exif_data[key] = value
        elif group.upper().startswith("XMP"):
            xmp.setdefault(key, str(value)[:500])
        elif group.upper().startswith(("IPTC", "PHOTOSHOP")):
            iptc.setdefault(key, value)
        elif group.upper().startswith("ICC_PROFILE"):
            icc.setdefault(key, value)
    if exiftool.get("location_present"):
        exif_data["GPSPresent"] = True
    has_descriptive_exif = any(key != "GPSPresent" for key in exif_data)
    software_values = [str(exif_data.get("Software", ""))]
    software_values.extend(
        value
        for key, value in xmp.items()
        if key.casefold() in {"creator tool", "creatortool", "producer", "software"}
    )
    software = list(
        dict.fromkeys(
            value.strip() for value in software_values if value and value.strip()
        )
    )
    findings: list[dict] = []
    all_metadata_text = " ".join([*software, *(str(v) for v in xmp.values())])
    if AI_MARKERS.search(all_metadata_text):
        findings.append(
            {
                "category": "METADATA",
                "severity": "HIGH",
                "finding": "Generation-related software or metadata marker found",
                "evidence": AI_MARKERS.search(all_metadata_text).group(0),
                "score_contribution": 0.35,
                "explanation": "A metadata string matches a known image-generation tool name; this can be edited and should be corroborated.",
            }
        )
    if any(EDIT_MARKERS.search(item) for item in software):
        findings.append(
            {
                "category": "METADATA",
                "severity": "MEDIUM",
                "finding": "Editing software is recorded",
                "evidence": "; ".join(software)[:200],
                "score_contribution": 0.08,
                "explanation": "Editing software can be used on both authentic and generated images and is not proof of manipulation.",
            }
        )
    original = _timestamp(exif_data.get("DateTimeOriginal"))
    modified = _timestamp(exif_data.get("DateTime"))
    if original and modified and original > modified:
        findings.append(
            {
                "category": "METADATA",
                "severity": "MEDIUM",
                "finding": "EXIF capture time is later than the file modification time",
                "evidence": f"DateTimeOriginal={original.isoformat()}; DateTime={modified.isoformat()}",
                "score_contribution": 0.05,
                "explanation": "The timestamp ordering is inconsistent; device clocks and metadata edits can also cause this.",
            }
        )
    if not has_descriptive_exif and not xmp and not iptc:
        findings.append(
            {
                "category": "METADATA",
                "severity": "LOW",
                "finding": "No common descriptive metadata was found",
                "evidence": "EXIF, XMP and IPTC fields absent",
                "score_contribution": 0.02,
                "explanation": "Metadata is often removed during sharing or export; absence is weak evidence.",
            }
        )

    return {
        "file": {
            "filename": filename,
            "extension": filename.rsplit(".", 1)[-1].lower() if "." in filename else "",
            "mime_type": mime_type,
            "byte_size": byte_size,
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "color_mode": image.mode,
            "bit_depth": _bit_depth(image),
            "compression": image.info.get("compression", "unknown"),
        },
        "exif": exif_data,
        "xmp": xmp,
        "iptc": iptc,
        "icc": icc,
        "exiftool": {key: value for key, value in exiftool.items() if key != "tags"},
        "software": software,
        "findings": findings,
    }
