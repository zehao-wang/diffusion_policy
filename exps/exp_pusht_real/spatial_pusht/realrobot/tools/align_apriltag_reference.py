from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ANCHOR_TAG_IDS = ("98", "99", "100")
KEEP_REFERENCE_TAG_IDS = ("76", "53", "77", "101")


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_correspondences(
    model_tags: dict[str, dict],
    reference_tags: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    model_points = []
    reference_points = []
    for tag_id in ANCHOR_TAG_IDS:
        if tag_id not in model_tags or tag_id not in reference_tags:
            raise KeyError(f"Missing anchor tag {tag_id} in model or reference JSON.")
        for corner_idx in ("0", "1", "2", "3"):
            model_points.append(model_tags[tag_id][corner_idx]["xyz_m"])
            reference_points.append(reference_tags[tag_id][corner_idx]["xyz_m"])
    return np.asarray(model_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)


def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    force_unit_scale: bool = False,
) -> SimilarityTransform:
    if source_points.shape != target_points.shape or source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Expected Nx3 source/target point arrays with identical shapes.")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = (target_centered.T @ source_centered) / source_points.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    if source_variance <= 0.0:
        raise ValueError("Degenerate source points; cannot estimate scale.")

    if force_unit_scale:
        scale = 1.0
    else:
        scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def transform_tag_payload(tag_payload: dict, transform: SimilarityTransform) -> dict:
    transformed = {}
    for corner_idx, corner_payload in tag_payload.items():
        xyz = np.asarray(corner_payload["xyz_m"], dtype=np.float64)
        xyz_transformed = transform.apply(xyz.reshape(1, 3))[0]
        transformed[corner_idx] = {
            "xyz_m": [float(value) for value in xyz_transformed],
            "sample_count": int(corner_payload.get("sample_count", 0)),
        }
    return transformed


def build_merged_payload(model_data: dict, reference_data: dict, transform: SimilarityTransform) -> dict:
    merged = {
        key: value
        for key, value in model_data.items()
        if key != "tags"
    }
    merged["tags"] = {
        str(tag_id): {
            str(corner_idx): {
                "xyz_m": [float(value) for value in corner_payload["xyz_m"]],
                "sample_count": int(corner_payload.get("sample_count", 0)),
            }
            for corner_idx, corner_payload in tag_payload.items()
        }
        for tag_id, tag_payload in model_data["tags"].items()
    }

    for tag_id in KEEP_REFERENCE_TAG_IDS:
        if tag_id not in reference_data["tags"]:
            continue
        merged["tags"][tag_id] = transform_tag_payload(reference_data["tags"][tag_id], transform)

    model_points, reference_points = collect_correspondences(model_data["tags"], reference_data["tags"])
    aligned_reference = transform.apply(reference_points)
    anchor_rmse = float(np.sqrt(np.mean(np.sum((aligned_reference - model_points) ** 2, axis=1))))

    merged["reference_alignment"] = {
        "anchor_tag_ids": [int(tag_id) for tag_id in ANCHOR_TAG_IDS],
        "kept_reference_tag_ids": [int(tag_id) for tag_id in KEEP_REFERENCE_TAG_IDS if tag_id in reference_data["tags"]],
        "source_json": "reference.json",
        "target_json": "model.json",
        "transform_type": "rigid" if np.isclose(transform.scale, 1.0) else "similarity",
        "scale": transform.scale,
        "rotation": transform.rotation.tolist(),
        "translation": transform.translation.tolist(),
        "matrix_4x4": transform.matrix().tolist(),
        "anchor_rmse_m": anchor_rmse,
    }
    return merged


def color_for_tag(tag_id: str) -> tuple[int, int, int]:
    if tag_id in KEEP_REFERENCE_TAG_IDS:
        return 255, 140, 0
    if tag_id in ANCHOR_TAG_IDS:
        return 80, 255, 80
    return 220, 220, 220


def write_ply(path: Path, tags: dict[str, dict]) -> None:
    vertices: list[tuple[float, float, float, int, int, int]] = []
    for tag_id in sorted(tags, key=lambda value: int(value)):
        rgb = color_for_tag(tag_id)
        for corner_idx in ("0", "1", "2", "3"):
            xyz = tags[tag_id][corner_idx]["xyz_m"]
            vertices.append((float(xyz[0]), float(xyz[1]), float(xyz[2]), *rgb))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for x, y, z, r, g, b in vertices:
            handle.write(f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}\n")


def parse_args() -> argparse.Namespace:
    default_input_dir = Path("/Users/wuminye/code/robodata_Agilex/data/records/apriltag_reconstruction")
    default_output_dir = Path("/Users/wuminye/code/robodata_Agilex/apriltag_reconstruction_outputs")
    parser = argparse.ArgumentParser(
        description="Align reference AprilTag corners to the model frame and export merged JSON/PLY.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing model.json and reference.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where the aligned JSON and PLY will be written.",
    )
    parser.add_argument(
        "--force-unit-scale",
        action="store_true",
        help="Use a rigid transform with scale fixed to 1.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = input_dir / "model.json"
    reference_path = input_dir / "reference.json"
    output_json_path = output_dir / "reference_aligned_to_model_filtered.json"
    output_ply_path = output_dir / "reference_aligned_to_model_filtered.ply"

    model_data = load_json(model_path)
    reference_data = load_json(reference_path)
    model_points, reference_points = collect_correspondences(model_data["tags"], reference_data["tags"])
    transform = estimate_similarity_transform(
        reference_points,
        model_points,
        force_unit_scale=args.force_unit_scale,
    )

    merged_payload = build_merged_payload(model_data, reference_data, transform)
    save_json(output_json_path, merged_payload)
    write_ply(output_ply_path, merged_payload["tags"])

    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote PLY: {output_ply_path}")
    print(f"Scale: {transform.scale:.12f}")
    print("Translation:", " ".join(f"{value:.12f}" for value in transform.translation))
    print(f"Anchor RMSE (m): {merged_payload['reference_alignment']['anchor_rmse_m']:.9f}")


if __name__ == "__main__":
    main()
