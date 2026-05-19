"""3D scene for the inference viser GUI.

Owns the viser scene handles for the world frame, bbox volume, T-block
mesh, pusher tip, and action waypoints. `InferScene.update(snap)` is
called once per main-loop tick to refresh handle positions/visibility
from a `StepSnapshot`.

Why a separate class: viser scene setters require `np.ndarray` (their
`cast_vector` reads `.shape` directly without converting), so all the
vector args are funnelled through `np.asarray` here once. The rest of
the GUI can just hand over Python lists / tuples.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import viser

from ..infer_loop import InferLoopRunner, StepSnapshot


def _vec(values, dtype=np.float32) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


class InferScene:
    def __init__(self, server: viser.ViserServer, runner: InferLoopRunner):
        self._server = server
        self._runner = runner

        server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=0.05)
        server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.002)

        bbox_min = np.asarray(runner.cfg.bbox_min, dtype=np.float32)
        bbox_max = np.asarray(runner.cfg.bbox_max, dtype=np.float32)
        server.scene.add_box(
            "/bbox",
            dimensions=_vec(bbox_max - bbox_min),
            position=_vec((bbox_min + bbox_max) / 2.0),
            color=(60, 180, 255),
            opacity=0.08,
            wireframe=True,
        )

        self._pusher = server.scene.add_icosphere(
            "/pusher",
            radius=0.008,
            color=(255, 64, 64),
            position=_vec((0.0, 0.0, 0.0)),
            visible=False,
        )

        self._tblock = self._add_tblock_mesh()

        self._action_handles: list = []
        self._action_voxels_drawn: Optional[list] = None
        # If set, only the first N waypoints of each prediction are drawn.
        self.max_action_waypoints: Optional[int] = None

        # Background AprilTags: paired green/gray quads per tag, toggled by
        # detection state each tick. Drawn at static world-frame corners
        # so the operator can see *where* the reference QR codes sit and
        # which ones the camera is currently picking up.
        self._tag_on_handles: dict = {}
        self._tag_off_handles: dict = {}
        self._add_background_tags()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _add_tblock_mesh(self):
        try:
            import trimesh
        except Exception as exc:
            print(f"[infer-scene] trimesh unavailable; tblock mesh hidden: {exc}")
            return None
        try:
            static = self._runner.extractor.static_model
            mesh = trimesh.Trimesh(
                vertices=np.asarray(static.mesh_vertices, dtype=np.float64),
                faces=np.asarray(static.mesh_faces, dtype=np.int64),
                process=False,
            )
            mesh.visual.face_colors = np.tile(
                np.array([[60, 180, 255, 180]], dtype=np.uint8),
                (len(static.mesh_faces), 1),
            )
            return self._server.scene.add_mesh_trimesh(
                "/tblock", mesh, visible=False,
            )
        except Exception as exc:
            print(f"[infer-scene] failed to attach tblock mesh: {exc}")
            return None

    def _add_background_tags(self) -> None:
        static = self._runner.extractor.static_model
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        for tag_id in static.background_tag_ids:
            corners = static.corner_points_by_tag.get(tag_id)
            if corners is None:
                continue
            vertices = np.ascontiguousarray(np.asarray(corners, dtype=np.float32).reshape(4, 3))
            self._tag_on_handles[tag_id] = self._server.scene.add_mesh_simple(
                f"/tags/bg/{tag_id}/on",
                vertices=vertices, faces=faces,
                color=(60, 220, 90), opacity=0.75, side="double",
                visible=False,
            )
            self._tag_off_handles[tag_id] = self._server.scene.add_mesh_simple(
                f"/tags/bg/{tag_id}/off",
                vertices=vertices, faces=faces,
                color=(140, 140, 140), opacity=0.35, side="double",
                visible=True,
            )

    # ------------------------------------------------------------------
    # Per-tick refresh
    # ------------------------------------------------------------------
    def update(self, snap: StepSnapshot) -> None:
        self._update_pusher(snap)
        self._update_tblock(snap)
        self._update_action_waypoints(snap)
        self._update_tag_visibility(snap)

    def _update_tag_visibility(self, snap: StepSnapshot) -> None:
        visible = set(snap.visible_background_tag_ids or ())
        for tag_id, on_h in self._tag_on_handles.items():
            is_on = tag_id in visible
            on_h.visible = is_on
            self._tag_off_handles[tag_id].visible = not is_on

    def _update_pusher(self, snap: StepSnapshot) -> None:
        if snap.eef_world is None:
            return
        self._pusher.position = _vec(snap.eef_world)
        self._pusher.visible = True

    def _update_tblock(self, snap: StepSnapshot) -> None:
        if self._tblock is None:
            return
        pose = snap.tblock_pose_world
        if pose is None:
            self._tblock.visible = False
            return
        t = pose.get("translation_m")
        q = pose.get("wxyz")
        if t is None or q is None:
            self._tblock.visible = False
            return
        self._tblock.position = _vec(t)
        self._tblock.wxyz = _vec(q)
        self._tblock.visible = True

    def _update_action_waypoints(self, snap: StepSnapshot) -> None:
        """Sticky: only redraw when a new prediction arrives. Perception-only
        ticks have `action_voxels=None` and must NOT erase the last drawn
        prediction. Use `clear_action_waypoints()` for explicit reset."""
        if snap.action_voxels is None:
            return
        cap = self.max_action_waypoints
        visible_voxels = (
            list(snap.action_voxels[:cap]) if cap is not None
            else list(snap.action_voxels)
        )
        if visible_voxels == self._action_voxels_drawn:
            return  # same prediction + same cap as last tick; nothing to redo
        self.clear_action_waypoints()
        self._action_voxels_drawn = visible_voxels
        z_voxel = int(self._runner.cfg.action_z_voxel)
        for i, vox in enumerate(visible_voxels):
            world = self._runner.extractor.voxel_xy_to_world(
                np.asarray(vox, dtype=np.float64), z_voxel=z_voxel
            )
            h = self._server.scene.add_icosphere(
                f"/action/{i}",
                radius=0.006,
                color=(60, 220, 90),
                position=_vec(world),
            )
            self._action_handles.append(h)

    def clear_action_waypoints(self) -> None:
        for h in self._action_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._action_handles = []
        self._action_voxels_drawn = None
