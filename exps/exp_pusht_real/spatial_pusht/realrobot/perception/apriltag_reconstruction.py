"""Pure AprilTag multi-view reconstruction helpers."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import least_squares


ANCHOR_TAG_IDS = (98, 99, 100)
COPLANAR_TAG_IDS = (98, 99, 100, 76, 53, 77, 101)
COPLANAR_TAG_IDS_GROUP_2 = (73, 48, 49)
BACKGROUND_TAG_IDS = (99, 98, 100, 76, 53, 101, 77)
DEFAULT_MODEL_DIR = Path("/Users/wuminye/code/robodata_Agilex/data/model")
DEFAULT_MODEL_JSON_PATH = DEFAULT_MODEL_DIR / "reference_aligned_to_model_filtered.json"
DEFAULT_MODEL_MESH_PATH = DEFAULT_MODEL_DIR / "tblock.ply"


@dataclass(frozen=True)
class AprilTagDetection2D:
    """Minimal 2D detection payload used by reconstruction."""

    tag_id: int
    corners: np.ndarray


@dataclass(frozen=True)
class AprilTagAnchorGeometry:
    """Tag corner geometry in the final aligned world frame."""

    tag_size_m: float
    corner_points_by_tag: dict[int, np.ndarray]
    reference_points: dict[str, np.ndarray]
    sample_counts_by_corner: dict[tuple[int, int], int]
    scale_factor: float
    mean_reproj_error_px: float
    max_reproj_error_px: float
    optimized_views: int


@dataclass(frozen=True)
class AprilTagPoseResult:
    """Camera pose relative to a reconstructed tag geometry."""

    rvec: np.ndarray
    tvec: np.ndarray
    reproj_error_px: float
    camera_center_world: np.ndarray


@dataclass(frozen=True)
class RigidTransform:
    """SE3 transform using the convention x_out = R * x_in + t."""

    rotation: np.ndarray
    translation: np.ndarray

    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> "RigidTransform":
        rotation_inv = self.rotation.T
        translation_inv = -rotation_inv @ self.translation
        return RigidTransform(rotation=rotation_inv, translation=translation_inv)

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        return RigidTransform(
            rotation=self.rotation @ other.rotation,
            translation=self.rotation @ other.translation + self.translation,
        )

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.rotation.T + self.translation

    @classmethod
    def from_rvec_tvec(cls, rvec: np.ndarray, tvec: np.ndarray) -> "RigidTransform":
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        return cls(rotation=rotation, translation=translation)


@dataclass(frozen=True)
class AprilTagStaticReconstructionModel:
    """Global AprilTag corners and object mesh after the one-time T1 alignment."""

    model_dir: Path
    json_path: Path
    mesh_path: Path
    T1: RigidTransform
    corner_points_by_tag: dict[int, np.ndarray]
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    background_tag_ids: tuple[int, ...]
    object_tag_ids: tuple[int, ...]


@dataclass(frozen=True)
class AprilTagMeshReconstructionResult:
    """Per-frame object mesh pose recovered from two PnP solves."""

    mesh_vertices_world: np.ndarray
    mesh_faces: np.ndarray
    T_world_from_object: RigidTransform
    camera_from_world: RigidTransform
    camera_from_object: RigidTransform
    camera_center_world: np.ndarray
    world_reproj_error_px: float
    object_reproj_error_px: float
    visible_background_tag_ids: tuple[int, ...]
    visible_object_tag_ids: tuple[int, ...]


@dataclass(frozen=True)
class ConstantVelocityKalmanConfig:
    """3D constant-velocity Kalman filter parameters."""

    process_accel_std: float
    measurement_std: float
    initial_position_std: float
    initial_velocity_std: float


class ConstantVelocityKalman3D:
    """Low-latency 3D position filter with a constant-velocity motion model."""

    def __init__(self, config: ConstantVelocityKalmanConfig):
        self._config = config
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._state is not None and self._covariance is not None

    def reset(self) -> None:
        self._state = None
        self._covariance = None

    def initialize(self, position: np.ndarray) -> None:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        self._state = np.zeros(6, dtype=np.float64)
        self._state[:3] = position
        self._covariance = np.diag(
            np.array(
                [self._config.initial_position_std**2] * 3
                + [self._config.initial_velocity_std**2] * 3,
                dtype=np.float64,
            )
        )

    def predict(self, dt_s: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Kalman filter must be initialized before predict().")

        dt = float(max(dt_s, 1e-4))
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3, dtype=np.float64) * dt

        noise_gain = np.zeros((6, 3), dtype=np.float64)
        noise_gain[:3, :] = 0.5 * dt * dt * np.eye(3, dtype=np.float64)
        noise_gain[3:, :] = dt * np.eye(3, dtype=np.float64)
        process_cov = (
            noise_gain
            @ noise_gain.T
            * (self._config.process_accel_std**2)
        )

        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process_cov
        return self._state[:3].copy()

    def update(
        self,
        position: np.ndarray,
        *,
        measurement_std: float | None = None,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if not self.initialized:
            self.initialize(position)
            return position.copy()

        measure_std = float(
            self._config.measurement_std if measurement_std is None else measurement_std
        )
        measurement_matrix = np.zeros((3, 6), dtype=np.float64)
        measurement_matrix[:, :3] = np.eye(3, dtype=np.float64)
        measurement_cov = np.eye(3, dtype=np.float64) * (measure_std**2)

        innovation = position - measurement_matrix @ self._state
        innovation_cov = measurement_matrix @ self._covariance @ measurement_matrix.T + measurement_cov
        kalman_gain = self._covariance @ measurement_matrix.T @ np.linalg.inv(innovation_cov)
        self._state = self._state + kalman_gain @ innovation
        identity = np.eye(6, dtype=np.float64)
        self._covariance = (identity - kalman_gain @ measurement_matrix) @ self._covariance
        return self._state[:3].copy()

    def current_position(self) -> np.ndarray | None:
        if not self.initialized:
            return None
        return self._state[:3].copy()


@dataclass(frozen=True)
class AprilTagPoseKalmanSmootherConfig:
    """Default-tuned pose smoother for low-latency AprilTag mesh tracking."""

    translation: ConstantVelocityKalmanConfig = ConstantVelocityKalmanConfig(
        process_accel_std=1.1,
        measurement_std=0.010,
        initial_position_std=0.02,
        initial_velocity_std=0.25,
    )
    rotation: ConstantVelocityKalmanConfig = ConstantVelocityKalmanConfig(
        process_accel_std=6.0,
        measurement_std=0.12,
        initial_position_std=0.2,
        initial_velocity_std=1.0,
    )
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.15
    reset_timeout_s: float = 0.6
    reproj_error_scale: float = 0.18
    max_reproj_error_px_for_update: float = 6.0
    snap_reproj_error_px_threshold: float = 2.5
    snap_translation_error_m: float = 0.03
    snap_rotation_error_rad: float = 0.35
    relock_reproj_error_px_threshold: float = 4.5
    relock_translation_consistency_m: float = 0.02
    relock_rotation_consistency_rad: float = 0.20
    relock_required_frames: int = 2


class AprilTagPoseKalmanSmoother:
    """Smooth SE3 poses while keeping up with motion using a constant-velocity prior."""

    def __init__(self, config: AprilTagPoseKalmanSmootherConfig | None = None):
        self._config = config or AprilTagPoseKalmanSmootherConfig()
        self._translation_filter = ConstantVelocityKalman3D(self._config.translation)
        self._rotation_filter = ConstantVelocityKalman3D(self._config.rotation)
        self._last_timestamp_s: float | None = None
        self._candidate_translation: np.ndarray | None = None
        self._candidate_rotation: np.ndarray | None = None
        self._candidate_count = 0

    def reset(self) -> None:
        self._translation_filter.reset()
        self._rotation_filter.reset()
        self._last_timestamp_s = None
        self._candidate_translation = None
        self._candidate_rotation = None
        self._candidate_count = 0

    def _clear_candidate(self) -> None:
        self._candidate_translation = None
        self._candidate_rotation = None
        self._candidate_count = 0

    def _adaptive_measurement_std(
        self,
        base_std: float,
        reproj_error_px: float,
        visible_tag_count: int,
    ) -> float:
        error_scale = 1.0 + self._config.reproj_error_scale * max(float(reproj_error_px), 0.0)
        tag_scale = 1.0 + 0.25 * max(0, 2 - int(visible_tag_count))
        return base_std * error_scale * tag_scale

    def update(
        self,
        pose: RigidTransform,
        *,
        timestamp_s: float,
        reproj_error_px: float,
        visible_tag_count: int,
    ) -> RigidTransform:
        timestamp = float(timestamp_s)
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp
        else:
            gap = max(0.0, timestamp - self._last_timestamp_s)
            if gap > self._config.reset_timeout_s:
                self.reset()
                self._last_timestamp_s = timestamp

        translation_meas = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        rotation_meas, _ = cv2.Rodrigues(np.asarray(pose.rotation, dtype=np.float64))
        rotation_meas = rotation_meas.reshape(3)

        predicted_translation = None
        predicted_rot = None
        if self._translation_filter.initialized and self._rotation_filter.initialized:
            dt = float(
                np.clip(
                    timestamp - float(self._last_timestamp_s),
                    self._config.min_dt_s,
                    self._config.max_dt_s,
                )
            )
            self._translation_filter.predict(dt)
            self._rotation_filter.predict(dt)
            predicted_translation = self._translation_filter.current_position()
            predicted_rot = self._rotation_filter.current_position()
            if predicted_rot is not None:
                if np.linalg.norm(rotation_meas - predicted_rot) > np.linalg.norm(
                    -rotation_meas - predicted_rot
                ):
                    rotation_meas = -rotation_meas

        if (
            predicted_translation is not None
            and predicted_rot is not None
            and float(reproj_error_px) > self._config.max_reproj_error_px_for_update
        ):
            self._clear_candidate()
            rotation_filtered_matrix, _ = cv2.Rodrigues(predicted_rot.reshape(3, 1))
            self._last_timestamp_s = timestamp
            return RigidTransform(
                rotation=rotation_filtered_matrix,
                translation=predicted_translation,
            )

        if predicted_translation is not None and predicted_rot is not None:
            translation_error = float(np.linalg.norm(translation_meas - predicted_translation))
            rotation_error = float(np.linalg.norm(rotation_meas - predicted_rot))
            should_snap_to_measurement = (
                float(reproj_error_px) <= self._config.snap_reproj_error_px_threshold
                and (
                    translation_error >= self._config.snap_translation_error_m
                    or rotation_error >= self._config.snap_rotation_error_rad
                )
            )
            if should_snap_to_measurement:
                self._translation_filter.reset()
                self._rotation_filter.reset()
                self._clear_candidate()
            elif (
                float(reproj_error_px) <= self._config.relock_reproj_error_px_threshold
                and (
                    translation_error >= self._config.snap_translation_error_m
                    or rotation_error >= self._config.snap_rotation_error_rad
                )
            ):
                if (
                    self._candidate_translation is not None
                    and self._candidate_rotation is not None
                    and float(np.linalg.norm(translation_meas - self._candidate_translation))
                    <= self._config.relock_translation_consistency_m
                    and float(np.linalg.norm(rotation_meas - self._candidate_rotation))
                    <= self._config.relock_rotation_consistency_rad
                ):
                    self._candidate_count += 1
                else:
                    self._candidate_translation = translation_meas.copy()
                    self._candidate_rotation = rotation_meas.copy()
                    self._candidate_count = 1

                if self._candidate_count >= self._config.relock_required_frames:
                    self._translation_filter.reset()
                    self._rotation_filter.reset()
                    self._clear_candidate()
            else:
                self._clear_candidate()

        translation_filtered = self._translation_filter.update(
            translation_meas,
            measurement_std=self._adaptive_measurement_std(
                self._config.translation.measurement_std,
                reproj_error_px,
                visible_tag_count,
            ),
        )
        rotation_filtered = self._rotation_filter.update(
            rotation_meas,
            measurement_std=self._adaptive_measurement_std(
                self._config.rotation.measurement_std,
                reproj_error_px,
                visible_tag_count,
            ),
        )
        self._last_timestamp_s = timestamp
        rotation_filtered_matrix, _ = cv2.Rodrigues(rotation_filtered.reshape(3, 1))
        return RigidTransform(
            rotation=rotation_filtered_matrix,
            translation=translation_filtered,
        )


@dataclass(frozen=True)
class MeshVertexKalmanSmootherConfig:
    """Per-vertex smoother for mesh vertices in world coordinates."""

    vertex: ConstantVelocityKalmanConfig = ConstantVelocityKalmanConfig(
        process_accel_std=0.8,
        measurement_std=0.010,
        initial_position_std=0.02,
        initial_velocity_std=0.20,
    )
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.20
    reset_timeout_s: float = 0.7
    reproj_error_scale: float = 0.18


class MeshVertexKalmanSmoother:
    """Apply one 3D constant-velocity Kalman filter per mesh vertex."""

    def __init__(self, config: MeshVertexKalmanSmootherConfig | None = None):
        self._config = config or MeshVertexKalmanSmootherConfig()
        self._filters: list[ConstantVelocityKalman3D] = []
        self._last_timestamp_s: float | None = None

    def reset(self) -> None:
        self._filters = []
        self._last_timestamp_s = None

    def _measurement_std(self, reproj_error_px: float) -> float:
        return self._config.vertex.measurement_std * (
            1.0 + self._config.reproj_error_scale * max(float(reproj_error_px), 0.0)
        )

    def update(
        self,
        vertices_world: np.ndarray,
        *,
        timestamp_s: float,
        reproj_error_px: float,
    ) -> np.ndarray:
        vertices_world = np.asarray(vertices_world, dtype=np.float64)
        if vertices_world.ndim != 2 or vertices_world.shape[1] != 3:
            raise ValueError("Expected mesh vertices as an Nx3 array.")

        timestamp = float(timestamp_s)
        if self._last_timestamp_s is not None:
            gap = max(0.0, timestamp - self._last_timestamp_s)
            if gap > self._config.reset_timeout_s:
                self.reset()

        if len(self._filters) != len(vertices_world):
            self.reset()
            self._filters = [
                ConstantVelocityKalman3D(self._config.vertex) for _ in range(len(vertices_world))
            ]

        measurement_std = self._measurement_std(reproj_error_px)
        filtered_vertices = np.zeros_like(vertices_world, dtype=np.float64)

        if self._last_timestamp_s is not None:
            dt = float(
                np.clip(
                    timestamp - self._last_timestamp_s,
                    self._config.min_dt_s,
                    self._config.max_dt_s,
                )
            )
            for vertex_filter in self._filters:
                if vertex_filter.initialized:
                    vertex_filter.predict(dt)

        for idx, vertex_world in enumerate(vertices_world):
            filtered_vertices[idx] = self._filters[idx].update(
                vertex_world,
                measurement_std=measurement_std,
            )

        self._last_timestamp_s = timestamp
        return filtered_vertices


def reference_points(axis_length_m: float) -> dict[str, np.ndarray]:
    s = float(axis_length_m)
    return {
        "origin": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "x_tip": np.array([s, 0.0, 0.0], dtype=np.float64),
        "y_tip": np.array([0.0, s, 0.0], dtype=np.float64),
        "z_tip": np.array([0.0, 0.0, s], dtype=np.float64),
    }


def camera_calibration_from_info(
    camera_info: dict | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if camera_info is None:
        return None, None, "Camera intrinsics unavailable"
    intrinsics = camera_info.get("intrinsics")
    if not isinstance(intrinsics, dict):
        return None, None, "Camera intrinsics unavailable"
    required = ("fx", "fy", "cx", "cy")
    if any(key not in intrinsics for key in required):
        return None, None, "Camera intrinsics incomplete"

    camera_matrix = np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion_info = camera_info.get("distortion") or {}
    dist_coeffs = np.array(
        [
            float(distortion_info.get("k1", 0.0)),
            float(distortion_info.get("k2", 0.0)),
            float(distortion_info.get("p1", 0.0)),
            float(distortion_info.get("p2", 0.0)),
            float(distortion_info.get("k3", 0.0)),
        ],
        dtype=np.float64,
    ).reshape(-1, 1)
    return camera_matrix, dist_coeffs, None


def _load_corner_points_by_tag_from_json(json_path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    tags_payload = payload.get("tags")
    if not isinstance(tags_payload, dict):
        raise ValueError(f"JSON does not contain a 'tags' object: {json_path}")

    corner_points_by_tag: dict[int, np.ndarray] = {}
    for tag_id_str, tag_payload in tags_payload.items():
        tag_id = int(tag_id_str)
        tag_points = []
        for corner_idx in range(4):
            corner_payload = tag_payload.get(str(corner_idx))
            if not isinstance(corner_payload, dict) or "xyz_m" not in corner_payload:
                raise ValueError(
                    f"Missing tag {tag_id} corner {corner_idx} xyz_m in {json_path}"
                )
            tag_points.append(np.asarray(corner_payload["xyz_m"], dtype=np.float64))
        corner_points_by_tag[tag_id] = np.asarray(tag_points, dtype=np.float64)
    return corner_points_by_tag


def _load_binary_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Encountered a degenerate direction while building T1.")
    return vector / norm


def _compute_t1_from_tag_100(corner_points_by_tag: dict[int, np.ndarray]) -> RigidTransform:
    tag_100 = corner_points_by_tag.get(100)
    if tag_100 is None:
        raise ValueError("Global model JSON is missing tag 100, required to build T1.")

    origin = np.asarray(tag_100[2], dtype=np.float64)
    x_axis = _normalize_vector(origin - tag_100[3])
    y_seed = origin - tag_100[1]
    z_axis = _normalize_vector(np.cross(x_axis, y_seed))
    y_axis = _normalize_vector(np.cross(z_axis, x_axis))

    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
    translation = -rotation @ origin
    return RigidTransform(rotation=rotation, translation=translation)


def _transform_corner_points_by_tag(
    corner_points_by_tag: dict[int, np.ndarray],
    transform: RigidTransform,
) -> dict[int, np.ndarray]:
    return {
        int(tag_id): transform.apply_points(tag_points)
        for tag_id, tag_points in corner_points_by_tag.items()
    }


def load_static_reconstruction_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> AprilTagStaticReconstructionModel:
    model_dir = Path(model_dir)
    json_path = model_dir / DEFAULT_MODEL_JSON_PATH.name
    mesh_path = model_dir / DEFAULT_MODEL_MESH_PATH.name

    corner_points_by_tag_raw = _load_corner_points_by_tag_from_json(json_path)
    mesh_vertices_raw, mesh_faces = _load_binary_ply_mesh(mesh_path)
    T1 = _compute_t1_from_tag_100(corner_points_by_tag_raw)

    corner_points_by_tag = _transform_corner_points_by_tag(corner_points_by_tag_raw, T1)
    mesh_vertices = T1.apply_points(mesh_vertices_raw)
    object_tag_ids = tuple(
        tag_id for tag_id in sorted(corner_points_by_tag) if tag_id not in BACKGROUND_TAG_IDS
    )

    return AprilTagStaticReconstructionModel(
        model_dir=model_dir.resolve(),
        json_path=json_path.resolve(),
        mesh_path=mesh_path.resolve(),
        T1=T1,
        corner_points_by_tag=corner_points_by_tag,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        background_tag_ids=tuple(BACKGROUND_TAG_IDS),
        object_tag_ids=object_tag_ids,
    )


def normalized_detections(detections: list) -> list[AprilTagDetection2D]:
    normalized: list[AprilTagDetection2D] = []
    for detection in detections:
        tag_id = int(getattr(detection, "tag_id"))
        corners = np.asarray(getattr(detection, "corners"), dtype=np.float64)
        if corners.shape != (4, 2):
            continue
        normalized.append(AprilTagDetection2D(tag_id=tag_id, corners=corners))
    return normalized


def _frame_observations(
    detections: list[AprilTagDetection2D],
) -> dict[tuple[int, int], np.ndarray]:
    observations = {}
    for detection in detections:
        for corner_idx, corner in enumerate(detection.corners):
            observations[(detection.tag_id, corner_idx)] = np.asarray(corner, dtype=np.float64)
    return observations


def _undistort_points(
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points_2d, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
    ).reshape(-1, 2)


def _project_point(
    point_world: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(point_world, dtype=np.float64).reshape(1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(2)


def _project_points_batch(
    points_world: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(points_world, dtype=np.float64).reshape(-1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(-1, 2)


def _triangulate_two_view(
    point_a: np.ndarray,
    point_b: np.ndarray,
    pose_a: tuple[np.ndarray, np.ndarray],
    pose_b: tuple[np.ndarray, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray | None:
    und_a = _undistort_points(np.asarray([point_a]), camera_matrix, dist_coeffs)
    und_b = _undistort_points(np.asarray([point_b]), camera_matrix, dist_coeffs)
    rvec_a, tvec_a = pose_a
    rvec_b, tvec_b = pose_b
    R_a, _ = cv2.Rodrigues(rvec_a.reshape(3, 1))
    R_b, _ = cv2.Rodrigues(rvec_b.reshape(3, 1))
    P_a = np.concatenate([R_a, tvec_a.reshape(3, 1)], axis=1)
    P_b = np.concatenate([R_b, tvec_b.reshape(3, 1)], axis=1)
    homog = cv2.triangulatePoints(
        P_a,
        P_b,
        und_a.T,
        und_b.T,
    )
    if abs(homog[3, 0]) < 1e-12:
        return None
    return (homog[:3, 0] / homog[3, 0]).reshape(3)


def _choose_bootstrap_pair(
    frame_obs: list[dict[tuple[int, int], np.ndarray]],
    camera_matrix: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    best = None
    for i in range(len(frame_obs)):
        keys_i = set(frame_obs[i].keys())
        for j in range(i + 1, len(frame_obs)):
            common_keys = sorted(keys_i & set(frame_obs[j].keys()))
            if len(common_keys) < 8:
                continue
            pts_i = np.asarray([frame_obs[i][key] for key in common_keys], dtype=np.float64)
            pts_j = np.asarray([frame_obs[j][key] for key in common_keys], dtype=np.float64)
            E, mask = cv2.findEssentialMat(
                pts_i,
                pts_j,
                cameraMatrix=camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            if E is None or mask is None:
                continue
            inlier_count = int(mask.ravel().sum())
            if inlier_count < 8:
                continue
            _, R, t, pose_mask = cv2.recoverPose(
                E,
                pts_i,
                pts_j,
                cameraMatrix=camera_matrix,
            )
            pose_inliers = int(pose_mask.ravel().sum())
            if best is None or pose_inliers > best[0]:
                inlier_keys = [
                    key for key, keep in zip(common_keys, pose_mask.ravel() > 0) if keep
                ]
                best = (pose_inliers, i, j, R, t.reshape(3), inlier_keys)
    if best is None:
        return None
    _, i, j, R, t, inlier_keys = best
    return i, j, R, t, inlier_keys


def _reprojection_error_for_pose(
    observations: dict[tuple[int, int], np.ndarray],
    points_by_key: dict[tuple[int, int], np.ndarray],
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    residuals = []
    for key, pixel_xy in observations.items():
        if key not in points_by_key:
            continue
        proj_xy = _project_point(points_by_key[key], rvec, tvec, camera_matrix, dist_coeffs)
        residuals.append(np.linalg.norm(proj_xy - pixel_xy))
    if not residuals:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(residuals))))


def _scale_geometry_if_possible(
    corner_points_by_tag: dict[int, np.ndarray],
    tag_size_m: float,
) -> tuple[dict[int, np.ndarray], float]:
    edge_lengths = []
    for tag_points in corner_points_by_tag.values():
        for idx in range(4):
            edge_lengths.append(float(np.linalg.norm(tag_points[(idx + 1) % 4] - tag_points[idx])))
    if not edge_lengths:
        return corner_points_by_tag, 1.0
    median_edge = float(np.median(edge_lengths))
    if median_edge <= 1e-9:
        return corner_points_by_tag, 1.0
    scale_factor = float(tag_size_m / median_edge)
    scaled_points = {
        tag_id: np.asarray(tag_points, dtype=np.float64) * scale_factor
        for tag_id, tag_points in corner_points_by_tag.items()
    }
    return scaled_points, scale_factor


def _bundle_adjust_tag_reconstruction(
    valid_frame_indices: list[int],
    frame_obs: list[dict[tuple[int, int], np.ndarray]],
    poses_init: dict[int, tuple[np.ndarray, np.ndarray]],
    points_by_key_init: dict[tuple[int, int], np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[tuple[int, int], np.ndarray], float, float]:
    pose_indices = valid_frame_indices
    point_keys = sorted(points_by_key_init)
    point_index = {key: idx for idx, key in enumerate(point_keys)}

    points_init = np.asarray([points_by_key_init[key] for key in point_keys], dtype=np.float64)
    pose_params = []
    for frame_idx in pose_indices[1:]:
        rvec, tvec = poses_init[frame_idx]
        pose_params.append(np.asarray(rvec, dtype=np.float64).reshape(3))
        pose_params.append(np.asarray(tvec, dtype=np.float64).reshape(3))
    x0 = np.concatenate(
        [np.concatenate(pose_params, axis=0), points_init.reshape(-1)],
        axis=0,
    ) if pose_params else points_init.reshape(-1)

    fixed_first_pose = poses_init[pose_indices[0]]
    n_opt_poses = max(0, len(pose_indices) - 1)
    reproj_weight = 1.0
    anchor_plane_weight = 2.0
    eval_counter = 0
    frame_observation_indices: dict[int, np.ndarray] = {}
    frame_observation_pixels: dict[int, np.ndarray] = {}
    for frame_idx in pose_indices:
        indices = []
        pixels = []
        for key, pixel_xy in frame_obs[frame_idx].items():
            idx = point_index.get(key)
            if idx is None:
                continue
            indices.append(idx)
            pixels.append(pixel_xy)
        frame_observation_indices[frame_idx] = np.asarray(indices, dtype=np.int64)
        frame_observation_pixels[frame_idx] = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)

    coplanar_group_indices: list[tuple[np.ndarray, np.ndarray | None]] = []
    for group_tag_ids, basis_keys in (
        (COPLANAR_TAG_IDS, [(100, 2), (100, 1), (100, 3)]),
        (COPLANAR_TAG_IDS_GROUP_2, [(49, 2), (49, 1), (49, 3)]),
    ):
        group_indices = np.asarray(
            [
                point_index[(tag_id, corner_idx)]
                for tag_id in group_tag_ids
                for corner_idx in range(4)
                if (tag_id, corner_idx) in point_index
            ],
            dtype=np.int64,
        )
        if all(key in point_index for key in basis_keys):
            basis_indices = np.asarray([point_index[key] for key in basis_keys], dtype=np.int64)
        else:
            basis_indices = None
        coplanar_group_indices.append((group_indices, basis_indices))

    def unpack(params: np.ndarray):
        poses = {pose_indices[0]: fixed_first_pose}
        offset = 0
        for frame_idx in pose_indices[1:]:
            rvec = params[offset:offset + 3]
            tvec = params[offset + 3:offset + 6]
            poses[frame_idx] = (rvec.reshape(3), tvec.reshape(3))
            offset += 6
        points = params[offset:].reshape(len(point_keys), 3)
        return poses, points

    def residuals(params: np.ndarray) -> np.ndarray:
        nonlocal eval_counter
        eval_counter += 1
        if progress_callback is not None and (eval_counter == 1 or eval_counter % 10 == 0):
            progress_callback(f"Bundle adjustment... eval {eval_counter}")
        poses, points = unpack(params)
        residual_list = []
        for frame_idx in pose_indices:
            rvec, tvec = poses[frame_idx]
            obs_indices = frame_observation_indices[frame_idx]
            if obs_indices.size == 0:
                continue
            observed_points = points[obs_indices]
            proj_xy = _project_points_batch(
                observed_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            residual_list.extend(
                (reproj_weight * (proj_xy - frame_observation_pixels[frame_idx])).reshape(-1)
            )

        def add_coplanar_group_residuals(group_indices: np.ndarray, basis_indices: np.ndarray | None):
            if basis_indices is None or group_indices.size == 0:
                return
            origin = points[basis_indices[0]]
            u_vec = points[basis_indices[1]] - origin
            v_vec = points[basis_indices[2]] - origin
            normal = np.cross(u_vec, v_vec)
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-9:
                return
            normal_unit = normal / normal_norm
            signed_distances = (points[group_indices] - origin) @ normal_unit
            residual_list.extend((anchor_plane_weight * signed_distances).reshape(-1))

        for group_indices, basis_indices in coplanar_group_indices:
            add_coplanar_group_residuals(group_indices, basis_indices)

        return np.asarray(residual_list, dtype=np.float64)

    if x0.size == 0 or n_opt_poses < 1:
        return points_by_key_init, 0.0, 0.0

    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=300,
    )
    if progress_callback is not None:
        progress_callback(f"Bundle adjustment finished in {eval_counter} evals")
    poses_opt, points_opt = unpack(result.x)
    point_map = {
        key: points_opt[point_index[key]].reshape(3)
        for key in point_keys
    }
    reproj_errors = []
    for frame_idx in pose_indices:
        rvec, tvec = poses_opt[frame_idx]
        for key, pixel_xy in frame_obs[frame_idx].items():
            if key not in point_map:
                continue
            proj_xy = _project_point(point_map[key], rvec, tvec, camera_matrix, dist_coeffs)
            reproj_errors.append(float(np.linalg.norm(proj_xy - pixel_xy)))
    mean_err = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    max_err = float(np.max(reproj_errors)) if reproj_errors else 0.0
    return point_map, mean_err, max_err


def reconstruct_anchor_geometry_from_frames(
    detection_frames: list[list[AprilTagDetection2D]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size_m: float,
    progress_callback: Callable[[str], None] | None = None,
) -> AprilTagAnchorGeometry | None:
    """Bootstrap poses/points from multi-view correspondences, then refine with BA."""
    if len(detection_frames) < 2:
        return None

    if progress_callback is not None:
        progress_callback(f"Preparing {len(detection_frames)} captured views")
    frame_obs = [_frame_observations(detections) for detections in detection_frames]
    bootstrap = _choose_bootstrap_pair(frame_obs, camera_matrix)
    if bootstrap is None:
        return None
    frame_a, frame_b, R_b, t_b, bootstrap_keys = bootstrap
    if progress_callback is not None:
        progress_callback(
            f"Bootstrapped from views {frame_a}/{frame_b} with {len(bootstrap_keys)} points"
        )

    poses_world_to_camera: dict[int, tuple[np.ndarray, np.ndarray]] = {
        frame_a: (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
    }
    rvec_b, _ = cv2.Rodrigues(R_b)
    poses_world_to_camera[frame_b] = (rvec_b.reshape(3), np.asarray(t_b, dtype=np.float64).reshape(3))

    points_by_key: dict[tuple[int, int], np.ndarray] = {}
    for key in bootstrap_keys:
        point_world = _triangulate_two_view(
            frame_obs[frame_a][key],
            frame_obs[frame_b][key],
            poses_world_to_camera[frame_a],
            poses_world_to_camera[frame_b],
            camera_matrix,
            dist_coeffs,
        )
        if point_world is not None:
            points_by_key[key] = point_world

    progress = True
    expansion_round = 0
    while progress:
        progress = False
        expansion_round += 1
        if progress_callback is not None:
            progress_callback(
                f"Expanding reconstruction... round {expansion_round}, "
                f"{len(poses_world_to_camera)} poses, {len(points_by_key)} points"
            )

        for frame_idx, observations in enumerate(frame_obs):
            if frame_idx in poses_world_to_camera:
                continue
            common_keys = [key for key in observations if key in points_by_key]
            if len(common_keys) < 6:
                continue
            object_points = np.asarray([points_by_key[key] for key in common_keys], dtype=np.float64)
            image_points = np.asarray([observations[key] for key in common_keys], dtype=np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or inliers is None or len(inliers) < 6:
                continue
            poses_world_to_camera[frame_idx] = (rvec.reshape(3), tvec.reshape(3))
            progress = True

        initialized_frames = sorted(poses_world_to_camera)
        for key in sorted({k for obs in frame_obs for k in obs.keys()}):
            if key in points_by_key:
                continue
            observing_frames = [idx for idx in initialized_frames if key in frame_obs[idx]]
            if len(observing_frames) < 2:
                continue
            best_point = None
            best_baseline = -1.0
            for i in range(len(observing_frames)):
                for j in range(i + 1, len(observing_frames)):
                    fi = observing_frames[i]
                    fj = observing_frames[j]
                    point_world = _triangulate_two_view(
                        frame_obs[fi][key],
                        frame_obs[fj][key],
                        poses_world_to_camera[fi],
                        poses_world_to_camera[fj],
                        camera_matrix,
                        dist_coeffs,
                    )
                    if point_world is None:
                        continue
                    ci = camera_center_from_pose(*poses_world_to_camera[fi])
                    cj = camera_center_from_pose(*poses_world_to_camera[fj])
                    baseline = float(np.linalg.norm(ci - cj))
                    if baseline > best_baseline:
                        best_baseline = baseline
                        best_point = point_world
            if best_point is not None:
                points_by_key[key] = best_point
                progress = True

    valid_frame_indices = sorted(poses_world_to_camera)
    if len(valid_frame_indices) < 2:
        return None
    if not all((tag_id, corner_idx) in points_by_key for tag_id in ANCHOR_TAG_IDS for corner_idx in range(4)):
        return None
    if progress_callback is not None:
        progress_callback(
            f"Running bundle adjustment on {len(valid_frame_indices)} views and {len(points_by_key)} points"
        )

    corner_points_by_tag_init: dict[int, np.ndarray] = {}
    sample_counts_by_corner: dict[tuple[int, int], int] = {}
    observed_tag_ids = sorted({tag_id for tag_id, _ in points_by_key.keys()})
    for tag_id in observed_tag_ids:
        tag_points = []
        for corner_idx in range(4):
            key = (tag_id, corner_idx)
            if key not in points_by_key:
                return None
            tag_points.append(points_by_key[key])
            sample_counts_by_corner[key] = sum(1 for idx in valid_frame_indices if key in frame_obs[idx])
        corner_points_by_tag_init[tag_id] = np.asarray(tag_points, dtype=np.float64)

    points_by_key_init = {
        (tag_id, corner_idx): corner_points_by_tag_init[tag_id][corner_idx]
        for tag_id in corner_points_by_tag_init
        for corner_idx in range(4)
    }

    points_by_key_opt, mean_reproj_error_px, max_reproj_error_px = _bundle_adjust_tag_reconstruction(
        valid_frame_indices,
        frame_obs,
        poses_world_to_camera,
        points_by_key_init,
        camera_matrix,
        dist_coeffs,
        progress_callback=progress_callback,
    )

    corner_points_by_tag_opt: dict[int, np.ndarray] = {}
    for tag_id in observed_tag_ids:
        corner_points_by_tag_opt[tag_id] = np.asarray(
            [points_by_key_opt[(tag_id, corner_idx)] for corner_idx in range(4)],
            dtype=np.float64,
        )

    corner_points_by_tag, scale_factor = _scale_geometry_if_possible(
        corner_points_by_tag_opt,
        tag_size_m,
    )
    axis_length = tag_size_m * max(scale_factor, 1.0)

    return AprilTagAnchorGeometry(
        tag_size_m=float(tag_size_m),
        corner_points_by_tag=corner_points_by_tag,
        reference_points=reference_points(axis_length),
        sample_counts_by_corner=sample_counts_by_corner,
        scale_factor=scale_factor,
        mean_reproj_error_px=mean_reproj_error_px,
        max_reproj_error_px=max_reproj_error_px,
        optimized_views=len(valid_frame_indices),
    )


def save_geometry_point_cloud_ply(
    geometry: AprilTagAnchorGeometry,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices: list[tuple[float, float, float, int, int, int]] = []
    for tag_id in sorted(geometry.corner_points_by_tag):
        if tag_id == 100:
            color = (80, 255, 80)
        elif tag_id == 99:
            color = (0, 180, 255)
        elif tag_id == 98:
            color = (255, 140, 0)
        else:
            color = (220, 220, 220)
        for point in geometry.corner_points_by_tag[tag_id]:
            vertices.append((float(point[0]), float(point[1]), float(point[2]), *color))

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in vertices:
            f.write(f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}\n")
    return path


def save_geometry_points_json(
    geometry: AprilTagAnchorGeometry,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tag_size_m": float(geometry.tag_size_m),
        "scale_factor": float(geometry.scale_factor),
        "mean_reproj_error_px": float(geometry.mean_reproj_error_px),
        "max_reproj_error_px": float(geometry.max_reproj_error_px),
        "optimized_views": int(geometry.optimized_views),
        "tags": {},
    }
    for tag_id in sorted(geometry.corner_points_by_tag):
        tag_points = geometry.corner_points_by_tag[tag_id]
        payload["tags"][str(tag_id)] = {
            str(corner_idx): {
                "xyz_m": [
                    float(point_world[0]),
                    float(point_world[1]),
                    float(point_world[2]),
                ],
                "sample_count": int(
                    geometry.sample_counts_by_corner.get((tag_id, corner_idx), 0)
                ),
            }
            for corner_idx, point_world in enumerate(tag_points)
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")
    return path


def camera_center_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return (-R.T @ np.asarray(tvec, dtype=np.float64).reshape(3)).reshape(3)


def _collect_pnp_correspondences(
    detections: list[AprilTagDetection2D],
    corner_points_by_tag: dict[int, np.ndarray],
    allowed_tag_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    object_points = []
    image_points = []
    visible_tag_ids = []
    for detection in detections:
        if detection.tag_id not in allowed_tag_ids:
            continue
        tag_points = corner_points_by_tag.get(detection.tag_id)
        if tag_points is None:
            continue
        object_points.append(tag_points)
        image_points.append(np.asarray(detection.corners, dtype=np.float64))
        visible_tag_ids.append(int(detection.tag_id))

    if not object_points:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            tuple(),
        )

    return (
        np.concatenate(object_points, axis=0),
        np.concatenate(image_points, axis=0),
        tuple(sorted(set(visible_tag_ids))),
    )


def _estimate_camera_pose_from_tag_subset(
    detections: list[AprilTagDetection2D],
    corner_points_by_tag: dict[int, np.ndarray],
    allowed_tag_ids: tuple[int, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[AprilTagPoseResult | None, tuple[int, ...], str | None]:
    object_points, image_points, visible_tag_ids = _collect_pnp_correspondences(
        detections,
        corner_points_by_tag,
        set(allowed_tag_ids),
    )
    if len(object_points) < 4:
        return None, visible_tag_ids, "need at least 4 visible corners"

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,
        iterationsCount=100,
        confidence=0.999,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, visible_tag_ids, "solvePnP failed"
    elif inliers is not None and len(inliers) >= 4:
        inlier_idx = inliers.reshape(-1)
        ok_refine, rvec, tvec = cv2.solvePnP(
            object_points[inlier_idx],
            image_points[inlier_idx],
            camera_matrix,
            dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok_refine:
            return None, visible_tag_ids, "solvePnP refinement failed"

    reproj, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    reproj = reproj.reshape(-1, 2)
    reproj_error_px = float(np.sqrt(np.mean(np.sum((reproj - image_points) ** 2, axis=1))))
    return (
        AprilTagPoseResult(
            rvec=rvec.reshape(3),
            tvec=tvec.reshape(3),
            reproj_error_px=reproj_error_px,
            camera_center_world=camera_center_from_pose(rvec, tvec),
        ),
        visible_tag_ids,
        None,
    )


def reconstruct_object_mesh_from_detections(
    detections: list[AprilTagDetection2D],
    model: AprilTagStaticReconstructionModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[AprilTagMeshReconstructionResult | None, str]:
    world_pose, visible_background_tag_ids, world_error = _estimate_camera_pose_from_tag_subset(
        detections,
        model.corner_points_by_tag,
        model.background_tag_ids,
        camera_matrix,
        dist_coeffs,
    )
    if world_pose is None:
        return None, f"Background PnP failed: {world_error}"

    object_pose, visible_object_tag_ids, object_error = _estimate_camera_pose_from_tag_subset(
        detections,
        model.corner_points_by_tag,
        model.object_tag_ids,
        camera_matrix,
        dist_coeffs,
    )
    if object_pose is None:
        return None, f"Object PnP failed: {object_error}"

    camera_from_world = RigidTransform.from_rvec_tvec(world_pose.rvec, world_pose.tvec)
    camera_from_object = RigidTransform.from_rvec_tvec(object_pose.rvec, object_pose.tvec)
    T_world_from_object = camera_from_world.inverse().compose(camera_from_object)
    mesh_vertices_world = T_world_from_object.apply_points(model.mesh_vertices)

    return (
        AprilTagMeshReconstructionResult(
            mesh_vertices_world=mesh_vertices_world,
            mesh_faces=model.mesh_faces,
            T_world_from_object=T_world_from_object,
            camera_from_world=camera_from_world,
            camera_from_object=camera_from_object,
            camera_center_world=world_pose.camera_center_world,
            world_reproj_error_px=world_pose.reproj_error_px,
            object_reproj_error_px=object_pose.reproj_error_px,
            visible_background_tag_ids=visible_background_tag_ids,
            visible_object_tag_ids=visible_object_tag_ids,
        ),
        "ok",
    )


def estimate_camera_pose_from_geometry(
    detections: list[AprilTagDetection2D],
    geometry: AprilTagAnchorGeometry,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> AprilTagPoseResult | None:
    object_points = []
    image_points = []
    for detection in detections:
        tag_points = geometry.corner_points_by_tag.get(detection.tag_id)
        if tag_points is None:
            continue
        object_points.append(tag_points)
        image_points.append(detection.corners)

    if not object_points:
        return None

    object_points_np = np.concatenate(object_points, axis=0)
    image_points_np = np.concatenate(image_points, axis=0)
    ok, rvec, tvec = cv2.solvePnP(
        object_points_np,
        image_points_np,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    reproj, _ = cv2.projectPoints(
        object_points_np,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    reproj = reproj.reshape(-1, 2)
    reproj_error_px = float(np.sqrt(np.mean(np.sum((reproj - image_points_np) ** 2, axis=1))))
    return AprilTagPoseResult(
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        reproj_error_px=reproj_error_px,
        camera_center_world=camera_center_from_pose(rvec, tvec),
    )
