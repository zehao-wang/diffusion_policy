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
from typing import Optional

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

    image: np.ndarray            # (1, H, W) float32 in {0, 0.5, 1}
    agent_pos: np.ndarray        # (2,) float32 voxel coords [x, y]
    # Sparse T-bar voxel outline, (k, 2) int32 [x, y] sorted by (x, y). Mirrors
    # the `tblock_coords` JSON field used at training time by the tbar-coords
    # policy variant. Empty (k=0) for unavailable / failed frames.
    tblock_coords: np.ndarray
    # Fixed-slot AprilTag corner voxel xy, (S, 2) int32. One slot per
    # (tag_id, corner_idx) in the same ordering the recording used at
    # training time. Always shape (S, 2) — even when the object pose is
    # unavailable we re-project the last-known canonical layout (or zeros
    # at startup) so the policy sees a stable shape.
    tag_keypoints: np.ndarray
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
        object_tag_ids: Optional[list[int]] = None,
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

        # Pre-bake the canonical (T1-aligned, object-frame) corners of every
        # object tag, in the (tag_id, corner_idx) ordering the recording used
        # at training time (sorted ascending). Per frame we'll apply the
        # current `T_world_from_object` to these to get the live world xyz of
        # each slot, then project to voxel xy — mirroring the JSON field
        # `tblock_apriltag_points_world[i].coord_xy` shipped by the recording.
        #
        # `object_tag_ids` restricts the slot set to the tags that were
        # actually present at training time. Without this filter we'd default
        # to all calibrated object tags in the static model, which on this
        # rig is 9 tags = 36 slots, while the dataset only logged 3 tags = 12
        # slots — so the obs shape would mismatch the trained policy.
        configured_object_ids = tuple(
            int(t) for t in self.static_model.object_tag_ids
        )
        if object_tag_ids is None:
            allowed_ids = configured_object_ids
        else:
            allowed_ids = tuple(int(t) for t in object_tag_ids)
            unknown = [t for t in allowed_ids if t not in configured_object_ids]
            if unknown:
                raise ValueError(
                    f"object_tag_ids={list(allowed_ids)} contains tag(s) "
                    f"{unknown} not present in static model "
                    f"{configured_object_ids}"
                )
        self.tag_slot_keys: tuple[tuple[int, int], ...] = tuple(
            sorted(
                (int(tag_id), int(corner_idx))
                for tag_id in allowed_ids
                for corner_idx in range(
                    self.static_model.corner_points_by_tag[int(tag_id)].shape[0]
                )
            )
        )
        if self.tag_slot_keys:
            canonical = np.stack([
                self.static_model.corner_points_by_tag[tag_id][corner_idx]
                for tag_id, corner_idx in self.tag_slot_keys
            ]).astype(np.float64)             # (S, 3) in canonical object frame
        else:
            canonical = np.zeros((0, 3), dtype=np.float64)
        self._tag_canonical_world = canonical
        # Voxel-grid metadata for the projection step.
        extent = self.bbox_max - self.bbox_min
        self._voxel_size_xyz = extent / np.maximum(
            self.resolution_xyz.astype(np.float64), 1.0
        )
        # Forward-fill seed: zeros until the first frame produces a valid pose.
        self._last_tag_keypoints: np.ndarray = np.zeros(
            (len(self.tag_slot_keys), 2), dtype=np.int32
        )

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
        else:
            t_world_from_object = recon.T_world_from_object
            mesh_world = recon.mesh_vertices_world
        tblock_translation = t_world_from_object.translation
        tblock_rotation = t_world_from_object.rotation

        sl = compute_spatial_language(
            mesh_vertices_world=mesh_world,
            mesh_faces=self.static_model.mesh_faces,
            pusher_point_world=pusher_world_xyz,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            resolution_xyz=self.resolution_xyz,
        )

        # Reproject the canonical tag corners into the current world frame
        # using `T_world_from_object`, then voxelise. Done once regardless of
        # the sl.available outcome so the tag-keypoint policy still gets a
        # plausible per-frame slot layout (or the forward-filled last-known
        # one when the pose is missing).
        tag_keypoints_now = self._project_tag_keypoints(t_world_from_object)
        if tag_keypoints_now is not None:
            self._last_tag_keypoints = tag_keypoints_now
        tag_keypoints_obs = self._last_tag_keypoints.copy()

        if not sl.available:
            # Even when the T-block can't be located we still ship the static
            # goal layer so the model sees a familiar baseline observation.
            return SpatialObservation(
                image=self.goal_grid.copy()[None],
                agent_pos=np.zeros(2, dtype=np.float32),
                tblock_coords=_empty_tblock_coords(),
                tag_keypoints=tag_keypoints_obs,
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
            # Sparse outline matches the JSON `tblock_coords` field used by
            # the tbar-coords policy variant at training time.
            tblock_coords=_sorted_xy(sl.tblock_voxels_2d),
            tag_keypoints=tag_keypoints_obs,
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
            tblock_coords=_empty_tblock_coords(),
            tag_keypoints=self._last_tag_keypoints.copy(),
            available=False,
            status=status,
            tblock_pose_world=None,
            raw_world_reproj_px=float("nan"),
            raw_object_reproj_px=float("nan"),
            visible_background_tags=tuple(),
            visible_object_tags=tuple(),
        )

    def _project_tag_keypoints(self, t_world_from_object) -> Optional[np.ndarray]:
        """Apply the current `T_world_from_object` to the canonical tag corners
        and voxelise the xy of each. Returns (S, 2) int32 or None if no slots
        are configured / pose is missing."""
        if self._tag_canonical_world.shape[0] == 0 or t_world_from_object is None:
            return None
        world_pts = t_world_from_object.apply_points(self._tag_canonical_world)
        xy = world_pts[:, :2]
        voxel = np.floor(
            (xy - self.bbox_min[:2]) / self._voxel_size_xyz[:2]
        ).astype(np.int32)
        max_xy = self.resolution_xyz[:2].astype(np.int32) - 1
        np.clip(voxel, 0, max_xy, out=voxel)
        return voxel


def _empty_tblock_coords() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.int32)


def _sorted_xy(coords) -> np.ndarray:
    """Match the training-time convention in episode_parser.parse_episode:
    cast to int, sort by (x, y) so the downstream MLP/UNet sees a stable
    ordering regardless of perception's iteration order."""
    arr = np.asarray(coords, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return _empty_tblock_coords()
    arr = arr[:, :2]
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]
