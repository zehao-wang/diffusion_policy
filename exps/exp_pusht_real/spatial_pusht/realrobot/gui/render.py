"""Pure-function renderers for the inference viser GUI.

No state, no viser handles. Each function takes a snapshot (or a raw
frame) and returns either a numpy image or a markdown string. Lives
separately from `infer_app.py` so the App class can stay focused on
GUI wiring and the main loop.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..infer_loop import StepSnapshot
from ...policy_service.client import PolicyStatus


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------
def format_state(snap: StepSnapshot) -> str:
    eef = snap.eef_world
    eef_str = f"X={eef[0]:.3f} Y={eef[1]:.3f} Z={eef[2]:.3f}" if eef else "—"
    pusher = snap.pusher_voxel
    bg_tags = snap.visible_background_tag_ids
    obj_tags = snap.visible_object_tag_ids
    out = (
        f"**EEF (world m):** {eef_str}\n\n"
        f"**Pusher voxel:** {pusher if pusher else '—'}\n\n"
        f"**T-block voxels:** {snap.tblock_voxel_count}\n\n"
        f"**Tags seen:** bg={len(bg_tags)} {sorted(bg_tags) if bg_tags else ''} | "
        f"obj={len(obj_tags)} {sorted(obj_tags) if obj_tags else ''}\n\n"
        f"**Reproj px:** world={snap.world_reproj_px:.2f} / object={snap.object_reproj_px:.2f}\n\n"
        f"**Loop dt:** {snap.dt_s * 1000:.1f} ms"
    )
    if snap.action_voxels is not None:
        out += f"  |  **Policy:** {snap.policy_took_ms:.1f} ms"
    return out


def format_action(snap: StepSnapshot) -> str:
    if snap.action_voxels is None:
        return f"**Action voxels:** —  *(status: {snap.status})*"
    executed = "executed" if snap.executed else "computed only"
    lines = [f"({v[0]:.1f}, {v[1]:.1f})" for v in snap.action_voxels]
    return f"**Action voxels** ({executed}):\n\n" + " → ".join(lines)


def format_policy(status: PolicyStatus) -> str:
    return (
        f"**Policy:** {status.ckpt_path}\n\n"
        f"device={status.device} | "
        f"n_obs={status.n_obs_steps} | "
        f"n_act={status.n_action_steps}"
    )


# ---------------------------------------------------------------------------
# Image renderers
# ---------------------------------------------------------------------------
def occupancy_2d(
    snap: StepSnapshot,
    *,
    resolution_xyz: np.ndarray,
    cell_px: int,
) -> np.ndarray:
    """Render T-block voxels + pusher + action voxels as a tiled 2D image."""
    H, W = int(resolution_xyz[1]), int(resolution_xyz[0])
    canvas = np.full((H, W, 3), 20, dtype=np.uint8)

    if snap.image_2d is not None:
        occ = np.asarray(snap.image_2d, dtype=np.float32)
        if occ.ndim == 3:
            occ = occ[0]
        if occ.shape == (W, H):
            occ = occ.T
        canvas[np.clip(occ, 0.0, 1.0) > 0.0] = (255, 255, 255)

    def stamp(x: int, y: int, color: tuple[int, int, int], radius: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                xi, yi = x + dx, y + dy
                if 0 <= xi < W and 0 <= yi < H:
                    canvas[yi, xi] = color

    if snap.action_voxels is not None:
        for vox in snap.action_voxels:
            stamp(int(round(vox[0])), int(round(vox[1])), (0, 255, 0), radius=1)
    if snap.pusher_voxel is not None:
        stamp(int(snap.pusher_voxel[0]), int(snap.pusher_voxel[1]), (255, 64, 64), radius=1)

    return np.kron(canvas, np.ones((cell_px, cell_px, 1), dtype=np.uint8))


def camera_frame(frame: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Coerce a camera frame to (target_h, target_w, 3) uint8 RGB."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    tw, th = target_wh
    if (w, h) != (tw, th):
        arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
    return arr
