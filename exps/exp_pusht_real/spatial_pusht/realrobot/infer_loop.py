"""Real-robot inference coordinator for spatial_pusht_image.

The coordinator no longer owns the policy — the trained checkpoint lives
behind the HTTP service in `policy_service/`. This module owns:

    Blackfly RGB ──► AprilTag detect ──► T-block mesh in AprilTag-world ─┐
                                                                         ├─► extractor → obs
    pusht_service /robot/state (background poll) ──► tip world ──────────┘
                                                                          │
                                                                          ▼
                                                policy_service /predict (HTTP)
                                                                          │
                                          voxel → world → base → /robot/step ► real arm

`InferLoopRunner` takes its subsystems via the constructor — see
`build_subsystems` for the standard wiring. The CLI `run()` is a
headless front-end useful for `--dry-run` sanity checks; the viser GUI
lives in `gui/infer_app.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml
from omegaconf import OmegaConf

from ..data.occupancy_utils import load_goal_grid_from_json, pad_tbar_coords_frame
from .arm_client import RealRobotArmClient
from .arm_reader import ArmReader
from .perception.apriltag_reconstruction import camera_calibration_from_info
from .perception.pointgrey_calibration import (
    load_pointgrey_calibration,
    merge_pointgrey_camera_info,
)
from .perception.pointgrey_client import PointGreyCamera
from .perception.state_extractor import SpatialStateExtractor
from ..policy_service.client import PolicyClient, PolicyStatus


# ---------------------------------------------------------------------------
# Snapshot returned by step() / perceive_once()
# ---------------------------------------------------------------------------
@dataclass
class StepSnapshot:
    """One iteration result, suitable for GUI display."""

    step: int = 0
    available: bool = False
    status: str = ""
    pusher_voxel: Optional[list[int]] = None
    tblock_voxel_count: int = 0
    world_reproj_px: float = float("nan")
    object_reproj_px: float = float("nan")
    eef_world: Optional[list[float]] = None
    action_voxels: Optional[list[list[float]]] = None
    executed: bool = False
    dt_s: float = 0.0
    policy_took_ms: float = float("nan")
    tblock_pose_world: Optional[dict] = None
    visible_background_tag_ids: tuple = ()
    visible_object_tag_ids: tuple = ()
    image_2d: Optional[np.ndarray] = field(default=None, repr=False)
    color_preview: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
class InferLoopRunner:
    """Pulls observations together and dispatches policy + arm commands.

    Subsystems are constructor-injected; use `build_subsystems` for the
    default wiring from a yaml config. Any of `camera` / `arm_client` /
    `arm_reader` may be None to support `--no-camera` / `--no-arm` modes
    (perception or arm steps become no-ops with an explanatory status).
    """

    def __init__(
        self,
        *,
        cfg,
        camera: Optional[PointGreyCamera],
        arm_client: Optional[RealRobotArmClient],
        arm_reader: Optional[ArmReader],
        policy_client: PolicyClient,
        policy_status: PolicyStatus,
        extractor: SpatialStateExtractor,
    ):
        self.cfg = cfg
        self.cam = camera
        self.arm = arm_client
        self.arm_reader = arm_reader
        self.policy = policy_client
        self.policy_status = policy_status
        self.extractor = extractor

        self.n_obs_steps = int(policy_status.n_obs_steps)
        self.n_action_steps = int(policy_status.n_action_steps)
        # Three policy flavors ride on this loop:
        #   * "image" with image_shape[0]==3 (pusht_real_rgb): we mirror the
        #     server's training-time transform client-side and ship the
        #     (3, H, W) float32 tensor -- the raw 2448x2048 uint8 frame is
        #     ~500x bigger once JSON-encoded and dominates /predict latency.
        #   * "image" with image_shape[0]==1 OR "lowdim": send the
        #     perception-extracted occupancy grid (obs.image) as-is.
        #   * "tbar_coords": send a (T, K, 2) padded voxel coord set built
        #     from obs.tblock_coords; no image on the wire.
        # Two "coords"-style policies share the same wire shape (T, S, 2) but
        # pull per-frame slot data from different perception fields.
        kind = policy_status.policy_kind
        if kind == "tbar_coords":
            if policy_status.tbar_pad_n is None:
                raise ValueError(
                    "policy_kind=tbar_coords but status.tbar_pad_n is None")
            self._coords_kind: Optional[str] = "tbar_coords"
            self._coord_slot_count: Optional[int] = policy_status.tbar_pad_n
        elif kind == "tag_keypoints":
            if policy_status.n_tag_keypoints is None:
                raise ValueError(
                    "policy_kind=tag_keypoints but status.n_tag_keypoints is None")
            self._coords_kind = "tag_keypoints"
            self._coord_slot_count = policy_status.n_tag_keypoints
        else:
            self._coords_kind = None
            self._coord_slot_count = None
        self._preprocess_raw_camera = (
            kind == "image"
            and policy_status.image_shape is not None
            and policy_status.image_shape[0] == 3
        )
        # `_obs_hist` stores dicts whose obs entry key matches the wire format
        # the policy server expects. Constant per runner; fixed once here so
        # the stack at predict time picks the right field every iteration.
        self._wire_obs_key = "coords" if self._coords_kind is not None else "image"
        self._image_target_shape: Optional[tuple[int, int, int]] = (
            policy_status.image_shape if self._preprocess_raw_camera else None
        )
        self.target_dt = 1.0 / float(self.cfg.loop.rate_hz)
        self.arm_connected = False

        self._obs_hist: deque = deque(maxlen=self.n_obs_steps)
        self._step_count = 0

        print(
            f"[infer] policy n_obs_steps={self.n_obs_steps} "
            f"n_action_steps={self.n_action_steps} "
            f"(executing full chunk per prediction; matches diffusion-policy eval default) "
            f"at {policy_status.ckpt_path}"
        )

    # ------------------------------------------------------------------
    # Arm passthroughs (write side stays on the client; reader is read-only)
    # ------------------------------------------------------------------
    def connect_arm(self) -> dict:
        if self.arm is None:
            raise RuntimeError("Arm client disabled (--no-arm).")
        result = self.arm.connect(
            channel=str(self.cfg.pusht_service.channel),
            interface=str(self.cfg.pusht_service.interface),
        )
        self.arm_connected = True
        return result

    def disconnect_arm(self) -> dict:
        if self.arm is None:
            return {"status": "no-op (--no-arm)"}
        try:
            return self.arm.disconnect()
        finally:
            self.arm_connected = False

    def lock_arm(self, speed: int = 50) -> dict:
        if self.arm is None:
            raise RuntimeError("Arm client disabled (--no-arm).")
        return self.arm.lock_pose(speed=int(speed))

    def unlock_arm(self) -> dict:
        if self.arm is None:
            raise RuntimeError("Arm client disabled (--no-arm).")
        return self.arm.unlock_pose()

    def reset_history(self) -> None:
        self._obs_hist.clear()

    # ------------------------------------------------------------------
    # One iteration. Always perceives; calls the policy when `run_policy`
    # AND we have a fresh arm reading; sends actions when `execute` AND
    # the arm is connected. With `run_policy=False` this is just a
    # perception preview that doesn't mutate the obs-history window.
    # ------------------------------------------------------------------
    def tick(
        self,
        *,
        run_policy: bool,
        execute: bool,
        on_predicted: Optional[Callable[["StepSnapshot"], None]] = None,
        on_executing: Optional[Callable[["StepSnapshot"], None]] = None,
    ) -> StepSnapshot:
        snap = StepSnapshot(step=self._step_count)
        t0 = time.time()
        try:
            color = self._capture_color(snap)
            if color is None:
                return snap
            snap.color_preview = color

            pusher_world, arm_status = self._read_arm_sync()
            if arm_status:
                snap.status = arm_status

            obs = self.extractor.step(color, pusher_world, timestamp_s=time.time())
            self._fill_snap_from_obs(snap, obs, arm_pusher_world=pusher_world)

            # Policy path is gated on: caller asking for it, perception OK,
            # and a live arm reading (we don't want the obs history to track
            # the placeholder zero pusher when /robot/state is offline).
            # Reproj is *not* gated here: the Kalman smoother in the
            # extractor already drops high-reproj measurements internally
            # (max_reproj_error_px_for_update), matching minye's design.
            if not run_policy or not obs.available or arm_status:
                return snap

            if self._coords_kind == "tbar_coords":
                wire_obs = pad_tbar_coords_frame(
                    obs.tblock_coords, self._coord_slot_count
                )
            elif self._coords_kind == "tag_keypoints":
                # Perception always emits a fixed (S, 2) tag_keypoints array;
                # we just defend against an S mismatch (different ckpt vs the
                # one the perception was configured for).
                if obs.tag_keypoints.shape != (self._coord_slot_count, 2):
                    raise RuntimeError(
                        f"tag_keypoints shape {obs.tag_keypoints.shape} != "
                        f"expected ({self._coord_slot_count}, 2). The ckpt "
                        f"and the perception's static model disagree.")
                wire_obs = obs.tag_keypoints.astype(np.float32)
            elif self._preprocess_raw_camera:
                wire_obs = self._preprocess_camera_frame(color)
            else:
                wire_obs = obs.image
            self._obs_hist.append({self._wire_obs_key: wire_obs, "agent_pos": obs.agent_pos})
            if len(self._obs_hist) < self.n_obs_steps:
                snap.status = f"warmup ({len(self._obs_hist)}/{self.n_obs_steps})"
                self._step_count += 1
                return snap

            obs_window = np.stack([w[self._wire_obs_key] for w in self._obs_hist], axis=0)
            agent_window = np.stack([w["agent_pos"] for w in self._obs_hist], axis=0)
            try:
                if self._coords_kind is not None:
                    result = self.policy.predict(
                        agent_pos_window=agent_window, coords_window=obs_window
                    )
                else:
                    result = self.policy.predict(obs_window, agent_window)
            except Exception as exc:
                snap.status = f"policy service error: {type(exc).__name__}: {exc}"
                return snap

            # Execute the full predicted action chunk (n_action_steps) before
            # re-observing, matching the diffusion-policy paper/eval convention.
            chosen = result["action"]
            snap.action_voxels = chosen.tolist()
            snap.policy_took_ms = float(result["took_ms"])
            snap.status = "ok"

            # Let the caller render the predicted waypoints BEFORE the (blocking)
            # execute loop, so the operator sees the targets while the arm is
            # still moving toward them.
            if on_predicted is not None:
                try:
                    on_predicted(snap)
                except Exception as exc:
                    print(f"[infer] on_predicted callback failed: {exc}", flush=True)

            if execute and self.arm_connected and self.arm is not None:
                target_worlds = [
                    self.extractor.voxel_xy_to_world(
                        vox_xy, z_voxel=int(self.cfg.action_z_voxel)
                    )
                    for vox_xy in chosen
                ]
                print(
                    f"[infer] step={snap.step} executing {len(target_worlds)} "
                    f"waypoint(s) via /robot/step_chunk (smooth)"
                )
                self.arm.send_targets_world(
                    target_worlds,
                    speed=int(self.cfg.pusht_service.speed),
                    timesteps=int(self.cfg.pusht_service.timesteps),
                    dt=float(self.cfg.pusht_service.dt),
                )
                # Refresh pusher + T-block in viser ONCE after the full chunk.
                # Per-waypoint refresh is gone deliberately — it gated the
                # next /robot/step on perception, which made motion chunky.
                if on_executing is not None:
                    fresh_world, _ = self._read_arm_sync()
                    fresh_color = self._capture_color(snap)
                    if fresh_color is not None:
                        snap.color_preview = fresh_color
                        try:
                            obs_now = self.extractor.step(
                                fresh_color, fresh_world, timestamp_s=time.time()
                            )
                            self._fill_snap_from_obs(
                                snap, obs_now, arm_pusher_world=fresh_world
                            )
                        except Exception as exc:
                            print(
                                f"[infer] post-execute perception failed: {exc}",
                                flush=True,
                            )
                    else:
                        snap.eef_world = fresh_world.tolist()
                    try:
                        on_executing(snap)
                    except Exception as exc:
                        print(
                            f"[infer] on_executing callback failed: {exc}",
                            flush=True,
                        )
                snap.executed = True

            self._step_count += 1
            return snap
        finally:
            snap.dt_s = time.time() - t0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _capture_color(self, snap: StepSnapshot) -> Optional[np.ndarray]:
        if self.cam is None:
            snap.status = "camera disabled (--no-camera)"
            return None
        try:
            return self.cam.get_frames()[0]
        except Exception as exc:
            snap.status = f"camera get_frames failed: {type(exc).__name__}: {exc}"
            return None

    def _preprocess_camera_frame(self, color: np.ndarray) -> np.ndarray:
        """Client-side mirror of policy_runner._preprocess_raw_image_window.

        Applies gray -> center-crop square -> resize -> repeat to 3 channels,
        scaled to float32 [0, 1], returning (3, H_t, W_t). Doing this here
        keeps the JSON payload small (a 2048x2448 RGB frame is ~30x larger on
        the wire than the final 96x96 tensor and dominates /predict latency).
        """
        assert self._image_target_shape is not None
        C_t, H_t, W_t = self._image_target_shape
        if C_t != 3 or H_t != W_t:
            raise ValueError(
                f"client-side preprocess assumes 3-channel square target, "
                f"got image_shape={self._image_target_shape}"
            )
        if color.ndim == 3 and color.shape[-1] == 3:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        elif color.ndim == 2:
            gray = color
        else:
            raise ValueError(
                f"camera frame must be (H,W) or (H,W,3) uint8, got {color.shape}"
            )
        H, W = gray.shape
        side = min(H, W)
        top = (H - side) // 2
        left = (W - side) // 2
        cropped = gray[top:top + side, left:left + side]
        resized = cv2.resize(cropped, (W_t, H_t), interpolation=cv2.INTER_AREA)
        rgb = np.repeat(resized[..., None], 3, axis=-1)
        return np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0

    def _read_arm_sync(self) -> tuple[np.ndarray, str]:
        """Read arm pose via fresh HTTP (no cache); returns (pusher_world, status).

        Bypasses ArmReader's ~30Hz poll cache so the agent_pose handed to the
        policy reflects the arm's state *at the moment policy obs is built*,
        not whatever the background poller last saw (up to 1/poll_hz =
        ~33ms ago). Costs one ``/robot/state`` HTTP RTT (~10-50ms) per
        tick, which is acceptable at the loop's 10Hz cadence.

        ``ArmReader`` is still kept alive for the GUI's live-pose readout
        (eval_app polls it without blocking); only the policy obs path
        switches to sync.
        """
        if self.arm is None:
            return np.zeros(3, dtype=np.float64), "arm disabled (--no-arm)"
        try:
            reading = self.arm.read()
        except Exception as exc:
            return (
                np.zeros(3, dtype=np.float64),
                f"arm read: {type(exc).__name__}: {exc}",
            )
        return reading.pusher_world, ""

    def _fill_snap_from_obs(
        self,
        snap: StepSnapshot,
        obs,
        *,
        arm_pusher_world: np.ndarray,
    ) -> None:
        if not snap.status:
            snap.status = obs.status
        snap.eef_world = arm_pusher_world.tolist()
        snap.world_reproj_px = float(obs.raw_world_reproj_px)
        snap.object_reproj_px = float(obs.raw_object_reproj_px)
        snap.tblock_pose_world = obs.tblock_pose_world
        snap.visible_background_tag_ids = tuple(obs.visible_background_tags)
        snap.visible_object_tag_ids = tuple(obs.visible_object_tags)
        if obs.available:
            snap.available = True
            snap.pusher_voxel = obs.agent_pos.astype(int).tolist()
            snap.tblock_voxel_count = int(obs.image.sum())
            snap.image_2d = np.asarray(obs.image, dtype=np.float32)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self.arm_reader is not None:
            try:
                self.arm_reader.stop()
            except Exception:
                pass
        if self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
        if self.arm_connected and self.arm is not None:
            try:
                self.arm.disconnect()
            except Exception:
                pass
            self.arm_connected = False
        print("[infer] shutdown complete")


# ---------------------------------------------------------------------------
# Subsystem wiring
# ---------------------------------------------------------------------------
def load_cfg(cfg_path: str):
    raw = yaml.safe_load(Path(cfg_path).read_text())
    return OmegaConf.create(raw)


def build_subsystems(
    cfg,
    *,
    no_arm: bool = False,
    no_camera: bool = False,
) -> dict[str, Any]:
    """Construct camera / arm-client / arm-reader / policy-client / extractor
    from a config. Returned dict is splat-friendly into `InferLoopRunner`."""
    policy_url = str(cfg.policy_service.url)
    policy_client = PolicyClient(
        policy_url,
        request_timeout_s=float(cfg.policy_service.get("request_timeout_s", 30.0)),
    )
    print(f"[infer] waiting for policy service at {policy_url}")
    policy_client.wait_ready(timeout_s=60.0)
    policy_status = policy_client.status()
    print(f"[infer] policy ready: ckpt={policy_status.ckpt_path}")

    pointgrey_calib_path = Path(cfg.paths.pointgrey_calibration).expanduser().resolve()
    calib_payload = load_pointgrey_calibration(str(pointgrey_calib_path))
    camera_info = merge_pointgrey_camera_info({}, calib_payload)
    camera_matrix, dist_coeffs, calib_err = camera_calibration_from_info(camera_info)
    if calib_err is not None:
        raise RuntimeError(f"Bad pointgrey calibration: {calib_err}")

    if no_camera:
        camera = None
        print("[infer] camera disabled (--no-camera)")
    else:
        camera = PointGreyCamera(
            width=int(cfg.camera.width),
            height=int(cfg.camera.height),
            fps=int(cfg.camera.fps),
            socket_path=str(cfg.camera.socket_path),
            shm_prefix=str(cfg.camera.shm_prefix),
            service_python=cfg.camera.service_python if cfg.camera.auto_start else None,
            calibration_path=str(pointgrey_calib_path),
            device_serial=cfg.camera.device_serial,
        )
        camera.start()
        print("[infer] camera started")

    resolution_xyz = np.array(cfg.resolution_xyz, dtype=np.int32)
    grid_hw = (int(resolution_xyz[1]), int(resolution_xyz[0]))
    goal_grid = load_goal_grid_from_json(cfg.paths.goal_source_json, grid_hw)

    extractor = SpatialStateExtractor(
        model_dir=Path(cfg.paths.model_dir).expanduser().resolve(),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        bbox_min=np.array(cfg.bbox_min, dtype=np.float64),
        bbox_max=np.array(cfg.bbox_max, dtype=np.float64),
        resolution_xyz=resolution_xyz,
        goal_grid=goal_grid,
        apriltag_family=str(cfg.apriltag.family),
        apriltag_nthreads=int(cfg.apriltag.nthreads),
        apriltag_quad_decimate=float(cfg.apriltag.quad_decimate),
        apriltag_quad_sigma=float(cfg.apriltag.quad_sigma),
        apriltag_refine_edges=bool(cfg.apriltag.refine_edges),
        apriltag_decode_sharpening=float(cfg.apriltag.decode_sharpening),
        enable_kalman=bool(cfg.apriltag.enable_kalman),
        # tag_keypoints ckpts ship their training tag-subset; restricting the
        # extractor to that subset keeps the obs slot count aligned with the
        # trained policy.
        object_tag_ids=policy_status.tag_ids,
    )

    if no_arm:
        arm_client = None
        arm_reader = None
        print("[infer] arm disabled (--no-arm)")
    else:
        arm_client = RealRobotArmClient(
            service_url=str(cfg.pusht_service.url),
            world_transform_path=Path(cfg.paths.world_transform).expanduser().resolve(),
            tip_in_eef_m_override=_load_tip_offset(cfg.paths.get("tip_in_eef_override", None)),
        )
        arm_reader = ArmReader(
            arm_client,
            poll_hz=float(cfg.pusht_service.get("poll_hz", 30.0)),
        )
        arm_reader.start()

    return dict(
        cfg=cfg,
        camera=camera,
        arm_client=arm_client,
        arm_reader=arm_reader,
        policy_client=policy_client,
        policy_status=policy_status,
        extractor=extractor,
    )


# ---------------------------------------------------------------------------
# Headless CLI loop (mostly for --dry-run sanity checks)
# ---------------------------------------------------------------------------
def run(cfg_path: str, *, dry_run: bool = False):
    cfg = load_cfg(cfg_path)
    runner = InferLoopRunner(**build_subsystems(cfg))
    if not dry_run:
        print("[infer] connecting arm via pusht_service")
        runner.connect_arm()

    max_steps = int(runner.cfg.loop.max_steps)
    try:
        while max_steps < 0 or runner._step_count < max_steps:
            t0 = time.time()
            snap = runner.tick(run_policy=True, execute=not dry_run)
            if not snap.available:
                if snap.status:
                    print(f"[infer] step={snap.step} {snap.status}")
            elif snap.action_voxels is None:
                print(f"[infer] step={snap.step} {snap.status}")
            else:
                print(
                    f"[infer] step={snap.step} pusher_vox={snap.pusher_voxel} "
                    f"tblock_vox_n={snap.tblock_voxel_count} "
                    f"action_voxels={snap.action_voxels} "
                    f"policy={snap.policy_took_ms:.1f}ms"
                )
            _sleep_to(t0, runner.target_dt)
    except KeyboardInterrupt:
        print("\n[infer] interrupted")
    finally:
        runner.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sleep_to(t_start: float, target_dt: float) -> None:
    remaining = target_dt - (time.time() - t_start)
    if remaining > 0:
        time.sleep(remaining)


def _load_tip_offset(path):
    if path is None:
        return None
    p = Path(str(path)).expanduser()
    if not p.exists():
        print(f"[infer] tip offset override file not found ({p}); will use file baked into world_transform")
        return None
    payload = json.loads(p.read_text())
    arr = payload.get("tip_position_in_eef_m")
    if arr is None:
        print(f"[infer] {p} has no tip_position_in_eef_m field; ignoring override")
        return None
    return np.asarray(arr, dtype=np.float64)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless real-robot inference loop for spatial_pusht.")
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "realrobot.yaml"),
        help="Path to realrobot.yaml.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't connect to the arm; only run perception + policy and log actions.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(args.config, dry_run=args.dry_run)
