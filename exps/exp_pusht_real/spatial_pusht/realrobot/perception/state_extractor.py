"""Camera frame -> spatial-pusht observation (occupancy + agent_pos).

Wraps the vendored apriltag/spatial_language modules into a single
`SpatialStateExtractor.step(color_rgb, pusher_world_xyz)` that returns the
exact obs dict the trained `SpatialPushTOccupancyImageDataset` expects, i.e.
`{"image": (1, H, W) float32, "agent_pos": (2,) float32}` plus the raw
`SpatialLanguageResult` for debugging. The image is tri-valued
({0.0=bg, 0.5=goal, 1.0=T-block}) so the static goal layer must be supplied
to the extractor at construction time via `goal_grid`.

All world-frame quantities are in the AprilTag-world frame defined by tag 100
(see `apriltag_reconstruction._compute_t1_from_tag_100`). The bbox / resolution
that map this frame to voxel indices must match the training episode JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .apriltag_reconstruction import (
    AprilTagPoseKalmanSmoother,
    AprilTagStaticReconstructionModel,
    MeshVertexKalmanSmoother,
    load_static_reconstruction_model,
    normalized_detections,
    reconstruct_object_mesh_from_detections,
)
from .spatial_language import compute_spatial_language, unavailable_result
from exps.exp_pusht_real.spatial_pusht.data.occupancy_utils import rasterize_occupancy


@dataclass
class SpatialObservation:
    """One inference-ready observation packaged with diagnostics."""

    image: np.ndarray            # (1, H, W) float32 in {0, 1}
    agent_pos: np.ndarray        # (2,) float32 voxel coords [x, y]
    available: bool
    status: str
    tblock_pose_world: dict | None  # {translation_m: (3,), wxyz: (4,)} or None
    raw_world_reproj_px: float
    raw_object_reproj_px: float
    visible_background_tags: tuple
    visible_object_tags: tuple


class SpatialStateExtractor:
    """Detects tags, recovers the T-block mesh pose, voxelizes the scene."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        resolution_xyz: np.ndarray,
        goal_grid: np.ndarray,
        apriltag_family: str = "tag36h11",
        apriltag_nthreads: int = 2,
        apriltag_quad_decimate: float = 1.0,
        apriltag_quad_sigma: float = 1.0,
        apriltag_refine_edges: bool = True,
        apriltag_decode_sharpening: float = 0.25,
        enable_kalman: bool = True,
    ):
        self.static_model: AprilTagStaticReconstructionModel = (
            load_static_reconstruction_model(model_dir)
        )
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        self.bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
        self.bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
        self.resolution_xyz = np.asarray(resolution_xyz, dtype=np.int32).reshape(3)
        self.grid_hw = (int(self.resolution_xyz[1]), int(self.resolution_xyz[0]))  # (H, W) = (y, x)

        self.goal_grid = np.asarray(goal_grid, dtype=np.float32)
        if self.goal_grid.shape != self.grid_hw:
            raise ValueError(
                f"goal_grid shape {self.goal_grid.shape} != grid_hw {self.grid_hw}"
            )

        from pupil_apriltags import Detector

        self._detector = Detector(
            families=apriltag_family,
            nthreads=int(apriltag_nthreads),
            quad_decimate=float(apriltag_quad_decimate),
            quad_sigma=float(apriltag_quad_sigma),
            refine_edges=bool(apriltag_refine_edges),
            decode_sharpening=float(apriltag_decode_sharpening),
        )

        self._enable_kalman = bool(enable_kalman)
        self._pose_smoother = AprilTagPoseKalmanSmoother() if enable_kalman else None
        self._vertex_smoother = MeshVertexKalmanSmoother() if enable_kalman else None

    # ------------------------------------------------------------------
    # Main per-frame call
    # ------------------------------------------------------------------
    def step(
        self,
        color_rgb: np.ndarray,
        pusher_world_xyz: np.ndarray,
        *,
        timestamp_s: float,
    ) -> SpatialObservation:
        if color_rgb is None:
            return self._unavailable("Invalid camera frame")
        if color_rgb.ndim == 2:
            gray = color_rgb if color_rgb.dtype == np.uint8 else color_rgb.astype(np.uint8)
        elif color_rgb.ndim == 3 and color_rgb.shape[2] == 3:
            gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
        elif color_rgb.ndim == 3 and color_rgb.shape[2] == 1:
            gray = color_rgb[..., 0]
        else:
            return self._unavailable(f"Unsupported frame shape {color_rgb.shape}")
        try:
            raw_detections = self._detector.detect(gray)
        except Exception as exc:
            return self._unavailable(f"AprilTag detect failed: {exc}")
        detections = normalized_detections(raw_detections)
        if not detections:
            return self._unavailable("No tags detected")

        recon, status = reconstruct_object_mesh_from_detections(
            detections=detections,
            model=self.static_model,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
        )
        if recon is None:
            return self._unavailable(f"PnP failed: {status}")

        # Optional Kalman pose smoothing on camera_from_world; then re-derive
        # mesh in world frame via the smoothed camera_from_object composition.
        if self._enable_kalman:
            filtered_cfw = self._pose_smoother.update(
                recon.camera_from_world,
                timestamp_s=float(timestamp_s),
                reproj_error_px=float(recon.world_reproj_error_px),
                visible_tag_count=len(recon.visible_background_tag_ids),
            )
            t_world_from_object = filtered_cfw.inverse().compose(recon.camera_from_object)
            mesh_world = t_world_from_object.apply_points(self.static_model.mesh_vertices)
            mesh_world = self._vertex_smoother.update(
                mesh_world,
                timestamp_s=float(timestamp_s),
                reproj_error_px=float(recon.object_reproj_error_px),
            )
            tblock_translation = t_world_from_object.translation
            tblock_rotation = t_world_from_object.rotation
        else:
            mesh_world = recon.mesh_vertices_world
            tblock_translation = recon.T_world_from_object.translation
            tblock_rotation = recon.T_world_from_object.rotation

        sl = compute_spatial_language(
            mesh_vertices_world=mesh_world,
            mesh_faces=self.static_model.mesh_faces,
            pusher_point_world=pusher_world_xyz,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            resolution_xyz=self.resolution_xyz,
        )

        if not sl.available:
            # Even when the T-block can't be located we still ship the static
            # goal layer so the model sees a familiar baseline observation.
            return SpatialObservation(
                image=self.goal_grid.copy()[None],
                agent_pos=np.zeros(2, dtype=np.float32),
                available=False,
                status=sl.status,
                tblock_pose_world=self._serialize_pose(tblock_translation, tblock_rotation),
                raw_world_reproj_px=float(recon.world_reproj_error_px),
                raw_object_reproj_px=float(recon.object_reproj_error_px),
                visible_background_tags=recon.visible_background_tag_ids,
                visible_object_tags=recon.visible_object_tag_ids,
            )

        # Overlay the dense T-block 2D voxel set on top of the static goal
        # mask -- matches the tri-valued encoding used at training time.
        grid = rasterize_occupancy(sl.tblock_voxels_2d_full, self.goal_grid)

        agent_pos = np.zeros(2, dtype=np.float32)
        if len(sl.pusher_voxels_2d) > 0:
            agent_pos = np.asarray(sl.pusher_voxels_2d[0], dtype=np.float32)

        return SpatialObservation(
            image=grid[None],
            agent_pos=agent_pos,
            available=True,
            status=sl.status,
            tblock_pose_world=self._serialize_pose(tblock_translation, tblock_rotation),
            raw_world_reproj_px=float(recon.world_reproj_error_px),
            raw_object_reproj_px=float(recon.object_reproj_error_px),
            visible_background_tags=recon.visible_background_tag_ids,
            visible_object_tags=recon.visible_object_tag_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def voxel_xy_to_world(self, voxel_xy: np.ndarray, *, z_voxel: int) -> np.ndarray:
        """Inverse of compute_spatial_language for action decoding.

        Returns a (3,) world-frame point at the centre of voxel (vx, vy, z_voxel).
        """
        voxel_xy = np.asarray(voxel_xy, dtype=np.float64).reshape(2)
        extent = self.bbox_max - self.bbox_min
        voxel_size = extent / np.maximum(self.resolution_xyz.astype(np.float64), 1.0)
        center = np.array(
            [
                self.bbox_min[0] + (voxel_xy[0] + 0.5) * voxel_size[0],
                self.bbox_min[1] + (voxel_xy[1] + 0.5) * voxel_size[1],
                self.bbox_min[2] + (float(z_voxel) + 0.5) * voxel_size[2],
            ],
            dtype=np.float64,
        )
        return center

    @staticmethod
    def _serialize_pose(translation, rotation) -> dict:
        from scipy.spatial.transform import Rotation as _R

        xyzw = _R.from_matrix(rotation).as_quat()
        return {
            "translation_m": np.asarray(translation, dtype=np.float64).copy(),
            "wxyz": np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64),
        }

    def _unavailable(self, status: str) -> SpatialObservation:
        return SpatialObservation(
            image=self.goal_grid.copy()[None],
            agent_pos=np.zeros(2, dtype=np.float32),
            available=False,
            status=status,
            tblock_pose_world=None,
            raw_world_reproj_px=float("nan"),
            raw_object_reproj_px=float("nan"),
            visible_background_tags=tuple(),
            visible_object_tags=tuple(),
        )
