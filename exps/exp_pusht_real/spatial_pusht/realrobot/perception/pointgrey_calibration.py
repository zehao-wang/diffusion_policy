"""Helpers for loading and applying PointGrey camera calibration files."""

from __future__ import annotations

import json
from pathlib import Path


def load_pointgrey_calibration(calibration_path: str | None) -> dict | None:
    """Load a saved calibration JSON file."""
    if calibration_path is None:
        return None

    path = Path(calibration_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration JSON must contain an object: {path}")
    return payload


def merge_pointgrey_camera_info(
    camera_info: dict | None,
    calibration: dict | None,
    *,
    calibration_path: str | None = None,
) -> dict | None:
    """Merge solved intrinsics into runtime camera info."""
    if camera_info is None and calibration is None:
        return None

    merged = {} if camera_info is None else dict(camera_info)
    if calibration is None:
        return merged

    base_serial = str(merged.get("serial") or "").strip()
    calib_serial = str(calibration.get("serial") or "").strip()
    if base_serial and calib_serial and base_serial != calib_serial:
        raise ValueError(
            f"Calibration serial mismatch: runtime={base_serial}, file={calib_serial}"
        )

    base_resolution = merged.get("resolution")
    calib_resolution = calibration.get("resolution")
    if isinstance(base_resolution, dict) and isinstance(calib_resolution, dict):
        base_size = (
            int(base_resolution.get("width", 0)),
            int(base_resolution.get("height", 0)),
        )
        calib_size = (
            int(calib_resolution.get("width", 0)),
            int(calib_resolution.get("height", 0)),
        )
        if all(v > 0 for v in base_size) and all(v > 0 for v in calib_size) and base_size != calib_size:
            raise ValueError(
                "Calibration resolution mismatch: "
                f"runtime={base_size[0]}x{base_size[1]}, "
                f"file={calib_size[0]}x{calib_size[1]}"
            )

    merged.setdefault("backend", calibration.get("backend", "pointgrey"))
    merged.setdefault("camera_model", calibration.get("camera_model", "PointGrey"))
    if calib_serial:
        merged["serial"] = calib_serial
    if isinstance(calib_resolution, dict):
        merged["resolution"] = {
            "width": int(calib_resolution["width"]),
            "height": int(calib_resolution["height"]),
        }
    if "fps" in calibration:
        merged["fps"] = float(calibration["fps"])
    if "stream" in calibration:
        merged["stream"] = calibration["stream"]
    if "color_space" in calibration:
        merged["color_space"] = calibration["color_space"]
    if "image_rectified" in calibration:
        merged["image_rectified"] = bool(calibration["image_rectified"])
    if "intrinsics" in calibration:
        merged["intrinsics"] = dict(calibration["intrinsics"])
    if "distortion_model" in calibration:
        merged["distortion_model"] = calibration["distortion_model"]
    if "distortion" in calibration:
        merged["distortion"] = dict(calibration["distortion"])

    metadata = dict(merged.get("calibration_metadata") or {})
    metadata.update(dict(calibration.get("calibration_metadata") or {}))
    if calibration_path is not None:
        metadata["calibration_path"] = str(Path(calibration_path).expanduser().resolve())
    merged["calibration_metadata"] = metadata
    return merged
