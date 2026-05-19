from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


BACKGROUND_TAG_IDS = {98, 99, 100, 53, 101, 77, 76}


@dataclass(frozen=True)
class RigidTransform:
    rotation: np.ndarray
    translation: np.ndarray

    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.rotation.T + self.translation


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_binary_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    blob = path.read_bytes()
    header_end = blob.index(b"end_header\n") + len(b"end_header\n")
    header = blob[:header_end].decode("ascii", errors="ignore").splitlines()

    vertex_count = None
    face_count = None
    for line in header:
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face "):
            face_count = int(line.split()[-1])

    if vertex_count is None or face_count is None:
        raise ValueError(f"Unable to parse vertex/face counts from {path}.")

    body = memoryview(blob)[header_end:]
    offset = 0

    vertices = []
    for _ in range(vertex_count):
        vertices.append(struct.unpack_from("<fff", body, offset))
        offset += 12

    faces = []
    for _ in range(face_count):
        face_size = struct.unpack_from("<B", body, offset)[0]
        offset += 1
        face = struct.unpack_from("<" + "i" * face_size, body, offset)
        offset += 4 * face_size
        if face_size != 3:
            raise ValueError(f"Only triangular faces are supported, found face size {face_size}.")
        faces.append(face)

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def write_ascii_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def collect_object_tag_corners(tag_payload: dict[str, dict]) -> tuple[np.ndarray, list[dict[str, int]]]:
    points = []
    metadata = []
    for tag_id in sorted(tag_payload, key=lambda value: int(value)):
        if int(tag_id) in BACKGROUND_TAG_IDS:
            continue
        for corner_idx in ("0", "1", "2", "3"):
            xyz = tag_payload[tag_id][corner_idx]["xyz_m"]
            points.append(xyz)
            metadata.append({"tag_id": int(tag_id), "corner_idx": int(corner_idx)})
    return np.asarray(points, dtype=np.float64), metadata


def point_to_triangle_distance_squared(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[float, np.ndarray]:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(ab.dot(ap))
    d2 = float(ac.dot(ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(ap.dot(ap)), a

    bp = point - b
    d3 = float(ab.dot(bp))
    d4 = float(ac.dot(bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(bp.dot(bp)), b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        projection = a + v * ab
        delta = point - projection
        return float(delta.dot(delta)), projection

    cp = point - c
    d5 = float(ab.dot(cp))
    d6 = float(ac.dot(cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(cp.dot(cp)), c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        projection = a + w * ac
        delta = point - projection
        return float(delta.dot(delta)), projection

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        projection = b + w * (c - b)
        delta = point - projection
        return float(delta.dot(delta)), projection

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    projection = a + v * ab + w * ac
    delta = point - projection
    return float(delta.dot(delta)), projection


def closest_point_on_mesh(
    point: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[float, np.ndarray]:
    best_distance_sq = np.inf
    best_projection = None
    for face in faces:
        distance_sq, projection = point_to_triangle_distance_squared(
            point,
            vertices[face[0]],
            vertices[face[1]],
            vertices[face[2]],
        )
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_projection = projection
    if best_projection is None:
        raise ValueError("Mesh contains no faces.")
    return best_distance_sq, best_projection


def evaluate_alignment(
    tag_corners_world: np.ndarray,
    mesh_vertices_world: np.ndarray,
    mesh_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    squared_distances = []
    closest_points = []
    for point in tag_corners_world:
        distance_sq, projection = closest_point_on_mesh(point, mesh_vertices_world, mesh_faces)
        squared_distances.append(distance_sq)
        closest_points.append(projection)
    return np.asarray(squared_distances, dtype=np.float64), np.asarray(closest_points, dtype=np.float64)


def solve_rigid_transform(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    tag_corners_world: np.ndarray,
) -> tuple[RigidTransform, np.ndarray, np.ndarray]:
    def transformed_vertices(params: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(params[:3]).as_matrix()
        translation = params[3:]
        return mesh_vertices @ rotation.T + translation

    def objective(params: np.ndarray) -> float:
        distances_sq, _ = evaluate_alignment(tag_corners_world, transformed_vertices(params), mesh_faces)
        return float(distances_sq.mean())

    result = minimize(
        objective,
        x0=np.zeros(6, dtype=np.float64),
        method="Powell",
        options={"xtol": 1e-8, "ftol": 1e-12, "maxiter": 10_000, "maxfev": 100_000},
    )
    if not result.success:
        raise RuntimeError(f"Rigid alignment failed: {result.message}")

    rotation = Rotation.from_rotvec(result.x[:3]).as_matrix()
    translation = result.x[3:]
    transform = RigidTransform(rotation=rotation, translation=translation)
    aligned_vertices = transform.apply(mesh_vertices)
    distances_sq, closest_points = evaluate_alignment(tag_corners_world, aligned_vertices, mesh_faces)
    return transform, distances_sq, closest_points


def parse_args() -> argparse.Namespace:
    default_model_dir = Path("/Users/wuminye/code/robodata_Agilex/data/model")
    parser = argparse.ArgumentParser(
        description="Solve a global SE3 transform that aligns tblock.ply to object AprilTag corner points.",
    )
    parser.add_argument(
        "--tag-json",
        type=Path,
        default=default_model_dir / "reference_aligned_to_model_filtered.json",
        help="JSON file containing aligned AprilTag corner coordinates.",
    )
    parser.add_argument(
        "--mesh-ply",
        type=Path,
        default=default_model_dir / "tblock.ply",
        help="PLY mesh for the tblock object in its current pose estimate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_model_dir / "tblock_mesh_alignment.json",
        help="Path for the solved SE3 and fit statistics.",
    )
    parser.add_argument(
        "--output-mesh",
        type=Path,
        default=default_model_dir / "tblock_aligned_to_tags.ply",
        help="Path for the transformed mesh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag_data = load_json(args.tag_json.resolve())
    mesh_vertices, mesh_faces = load_binary_ply_mesh(args.mesh_ply.resolve())
    tag_corners_world, tag_corner_metadata = collect_object_tag_corners(tag_data["tags"])

    transform, distances_sq, closest_points = solve_rigid_transform(
        mesh_vertices,
        mesh_faces,
        tag_corners_world,
    )

    aligned_vertices = transform.apply(mesh_vertices)
    write_ascii_ply_mesh(args.output_mesh.resolve(), aligned_vertices, mesh_faces)

    distances = np.sqrt(distances_sq)
    payload = {
        "mesh_ply": str(args.mesh_ply.resolve()),
        "tag_json": str(args.tag_json.resolve()),
        "background_tag_ids": sorted(BACKGROUND_TAG_IDS),
        "object_tag_ids": sorted({item["tag_id"] for item in tag_corner_metadata}),
        "object_corner_count": int(len(tag_corners_world)),
        "transform_type": "rigid",
        "rotation": transform.rotation.tolist(),
        "translation": transform.translation.tolist(),
        "matrix_4x4": transform.matrix().tolist(),
        "fit_metrics_m": {
            "rmse": float(np.sqrt(distances_sq.mean())),
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "max": float(distances.max()),
        },
        "per_corner_distances_m": [
            {
                **metadata,
                "distance_m": float(distance),
                "closest_point_xyz_m": [float(value) for value in closest_point],
                "corner_xyz_m": [float(value) for value in corner],
            }
            for metadata, distance, closest_point, corner in zip(
                tag_corner_metadata,
                distances,
                closest_points,
                tag_corners_world,
                strict=True,
            )
        ],
    }
    save_json(args.output_json.resolve(), payload)

    print(f"Wrote transform JSON: {args.output_json.resolve()}")
    print(f"Wrote aligned mesh: {args.output_mesh.resolve()}")
    print("Translation:", " ".join(f"{value:.12f}" for value in transform.translation))
    print(f"RMSE (m): {payload['fit_metrics_m']['rmse']:.9f}")
    print(f"Max corner-to-surface distance (m): {payload['fit_metrics_m']['max']:.9f}")


if __name__ == "__main__":
    main()
