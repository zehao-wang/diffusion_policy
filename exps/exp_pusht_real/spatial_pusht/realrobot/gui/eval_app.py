"""Viser GUI for spatial_pusht_image real-robot EVALUATION.

Mirrors `infer_app.py` but runs the standard pusht evaluation protocol
instead of the free-running auto/step-once controls:

* Pressing **Start Trial** locks the button, resets the diffusion obs
  history, then auto-executes the policy from the current observation.
* The trial ends when either (a) coverage reaches the configured
  threshold (default 0.95) or (b) the step count hits the configured
  maximum (default 500). The Start Trial button re-enables so the
  operator can rearrange the T-block and start the next trial manually.
* Each trial dumps the same per-step combined frame as `infer_runs`
  plus a per-step `coverage_log.jsonl` and an end-of-trial
  `result.json`, under `data/realrobot/eval_runs/trial_{NNN}/`.

Coverage is computed from the tri-valued occupancy grid the extractor
already produces: background=0.0, goal=GOAL_VALUE (0.5),
T-block=TBLOCK_VALUE (1.0). When a T-block voxel lands on a goal cell
the grid value is overwritten to TBLOCK_VALUE, so coverage is the
fraction of original goal cells that the current T-block image marks
as TBLOCK_VALUE.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import viser
from scipy.ndimage import binary_dilation, binary_fill_holes

from ..infer_loop import InferLoopRunner, StepSnapshot, _sleep_to
from ...data.occupancy_utils import GOAL_VALUE, TBLOCK_VALUE
from . import render
from .scene import InferScene


PREVIEW_CELL_PX = 16
CAMERA_PREVIEW_MAX_W = 640
CAMERA_PREVIEW_MAX_H = 540
SAVE_ROOT = Path("data/realrobot/eval_runs")
DEFAULT_MAX_STEPS = 300  # matches diffusion_policy pusht_{image,lowdim}.yaml.
DEFAULT_COVERAGE_THRESHOLD = 0.95
# After each executed chunk we lift the EEF by REVEAL_LIFT_M to clear the
# pusher off the T-block tags, take one clean coverage reading, then lower
# back. The lift is broken into REVEAL_LIFT_N_STEPS sub-moves (each
# ~REVEAL_LIFT_M/N tall) so that:
#   (a) every sub-move is a single-segment chunk -- segment 1 is exempt
#       from the planner's z safety-clip, so each step can leave the slab
#       without truncating (vs. one big 5cm chunk whose segments 2..N
#       would be clipped);
#   (b) joint-space smoothness inside trajopt has very little room to
#       curve the Cartesian path on a 1cm move, so the EEF stays close
#       to a straight vertical line going up. On the way down we replay
#       the captured sub-trajectories in reverse order (each individually
#       reversed) so the descent retraces the ascent joint-by-joint.
REVEAL_LIFT_M = 0.05
REVEAL_LIFT_N_STEPS = 10
# Voxel dilation applied to the live T-block mask before intersecting
# with the goal mask, to compensate for rasterization mismatch between
# the goal mask (filled outline) and the live mask (orthographic 3D
# projection). 1 voxel at 128x128 over a ~0.55m bbox is ~4.3mm -- on
# the order of the rasterization noise floor, so perfectly-aligned poses
# reach ~1.0 coverage while a real 1+ voxel misalignment is still
# penalised. Set to 0 to recover the strict no-tolerance metric.
COVERAGE_TBLOCK_DILATION_VOX = 1


class EvalViserApp:
    def __init__(
        self,
        runner: InferLoopRunner,
        *,
        host: str = "0.0.0.0",
        port: int = 8013,
    ):
        self.runner = runner
        self.server = viser.ViserServer(
            host=host, port=port, label="spatial_pusht_image evaluation"
        )

        self._trial_running = threading.Event()
        self._arm_op_lock = threading.Lock()
        self._latest_snap: Optional[StepSnapshot] = None

        # Sticky display state — last prediction stays on the scene/2D
        # overlay until the next prediction lands (matches infer_app).
        self._sticky_action_voxels: Optional[list] = None
        self._sticky_action_text: Optional[str] = None

        # Per-trial state.
        self._trial_dir: Optional[Path] = None
        self._trial_step_count: int = 0
        self._trial_save_idx: int = 0
        self._trial_best_coverage: float = 0.0
        self._trial_last_coverage: float = 0.0
        self._trial_success: bool = False
        self._trial_log_path: Optional[Path] = None

        # Goal mask is static for the run; precompute once for coverage.
        # `goal_grid` stores only the outline cells at GOAL_VALUE (~88 cells
        # for the canonical T). The diffusion_policy pusht_env defines
        # coverage as polygon-intersection AREA over goal AREA, not over
        # the boundary, so we fill the interior here. binary_fill_holes on
        # the outline yields ~464 cells, matching the filled T-block size
        # from the same frame (~468) to within a voxel of rounding.
        goal_grid = np.asarray(self.runner.extractor.goal_grid, dtype=np.float32)
        goal_outline = np.isclose(goal_grid, GOAL_VALUE)
        self._goal_mask = binary_fill_holes(goal_outline).astype(bool)
        self._goal_cell_count = int(self._goal_mask.sum())

        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        next_idx = _next_trial_index(SAVE_ROOT)

        self._build_sidebar(default_trial_index=next_idx)
        self._scene = InferScene(self.server, self.runner)
        self._scene.max_action_waypoints = int(self._viz_action_n.value)

    # ------------------------------------------------------------------
    # Sidebar layout
    # ------------------------------------------------------------------
    def _build_sidebar(self, *, default_trial_index: int) -> None:
        srv = self.server

        with srv.gui.add_folder("Camera (Pointgrey)"):
            cam_w, cam_h = self._preview_size()
            self._camera_image = srv.gui.add_image(
                np.zeros((cam_h, cam_w, 3), dtype=np.uint8),
                label="Live frame",
            )
            self._camera_status_md = srv.gui.add_markdown("**Camera:** waiting…")

        with srv.gui.add_folder("Arm"):
            self._btn_connect = srv.gui.add_button("Connect")
            self._btn_disconnect = srv.gui.add_button("Disconnect")
            self._btn_lock = srv.gui.add_button("Lock Pose")
            self._btn_unlock = srv.gui.add_button("Unlock Pose…", color="red")
            self._arm_status_md = srv.gui.add_markdown("**Arm:** disconnected")
            disabled = self.runner.arm is None
            for btn in (self._btn_connect, self._btn_disconnect, self._btn_lock, self._btn_unlock):
                btn.disabled = disabled
        self._btn_connect.on_click(self._on_connect_click)
        self._btn_disconnect.on_click(self._on_disconnect_click)
        self._btn_lock.on_click(self._on_lock_click)
        self._btn_unlock.on_click(self._on_unlock_click)

        with srv.gui.add_folder("Evaluation"):
            self._trial_index = srv.gui.add_number(
                "Trial index", initial_value=int(default_trial_index),
                min=0, step=1,
            )
            self._max_steps = srv.gui.add_number(
                "Max steps", initial_value=int(DEFAULT_MAX_STEPS),
                min=1, step=1,
            )
            self._cov_threshold = srv.gui.add_number(
                "Coverage threshold", initial_value=float(DEFAULT_COVERAGE_THRESHOLD),
                min=0.0, max=1.0, step=0.01,
            )
            self._auto_lift = srv.gui.add_checkbox(
                "Auto-lift to reveal", initial_value=True,
            )
            self._btn_start_trial = srv.gui.add_button("Start Trial")
            self._btn_end_trial = srv.gui.add_button(
                "End Trial (abort)", color="red"
            )
            self._btn_end_trial.disabled = True
            srv.gui.add_markdown(render.format_policy(self.runner.policy_status))
            self._eval_status_md = srv.gui.add_markdown("**Eval:** idle")
        self._btn_start_trial.on_click(self._on_start_trial_click)
        self._btn_end_trial.on_click(self._on_end_trial_click)

        with srv.gui.add_folder("State"):
            self._state_md = srv.gui.add_markdown("**State:** waiting for first tick")

        with srv.gui.add_folder("Action"):
            self._action_md = srv.gui.add_markdown("**Action voxels:** —")
            n_act = int(self.runner.n_action_steps)
            self._viz_action_n = srv.gui.add_slider(
                "Show first N waypoints",
                min=1, max=max(n_act, 1), step=1, initial_value=n_act,
            )

            @self._viz_action_n.on_update
            def _(_evt) -> None:
                self._scene.max_action_waypoints = int(self._viz_action_n.value)

        with srv.gui.add_folder("Occupancy (XY)"):
            res_xyz = np.asarray(self.runner.cfg.resolution_xyz, dtype=np.int32)
            blank = np.zeros(
                (int(res_xyz[1]) * PREVIEW_CELL_PX, int(res_xyz[0]) * PREVIEW_CELL_PX, 3),
                dtype=np.uint8,
            )
            self._image_handle = srv.gui.add_image(
                blank, label="Voxel 2D (tblock + pusher + action)"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _set_status(handle, label: str, text: str) -> None:
        handle.content = f"**{label}:** {text}"

    def _preview_size(self) -> tuple[int, int]:
        w = int(self.runner.cfg.camera.width)
        h = int(self.runner.cfg.camera.height)
        scale = min(
            1.0, CAMERA_PREVIEW_MAX_W / max(w, 1), CAMERA_PREVIEW_MAX_H / max(h, 1)
        )
        return max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    # ------------------------------------------------------------------
    # Arm callbacks (verbatim from infer_app)
    # ------------------------------------------------------------------
    def _on_connect_click(self, _event) -> None:
        threading.Thread(target=self._arm_op, args=("connect",), daemon=True).start()

    def _on_disconnect_click(self, _event) -> None:
        threading.Thread(target=self._arm_op, args=("disconnect",), daemon=True).start()

    def _on_lock_click(self, _event) -> None:
        threading.Thread(target=self._arm_op, args=("lock",), daemon=True).start()

    def _on_unlock_click(self, event) -> None:
        client = event.client
        with client.gui.add_modal("Confirm Unlock Pose") as modal:
            client.gui.add_markdown(
                "**WARNING — torque will be cut.**\n\n"
                "* Motors will be DISABLED.\n"
                "* The arm WILL DROP under gravity.\n"
                "* Anything mounted on the EEF may collide with the table.\n"
                "* Support the arm physically BEFORE confirming.\n\n"
                "Call **Connect** or **Lock Pose** again afterwards to re-enable motors."
            )
            confirm_btn = client.gui.add_button("Yes, unlock now", color="red")
            cancel_btn = client.gui.add_button("Cancel")

            def _confirm(_evt):
                modal.close()
                threading.Thread(target=self._arm_op, args=("unlock",), daemon=True).start()

            def _cancel(_evt):
                modal.close()

            confirm_btn.on_click(_confirm)
            cancel_btn.on_click(_cancel)

    def _arm_op(self, op: str) -> None:
        def status(text: str) -> None:
            self._set_status(self._arm_status_md, "Arm", text)

        with self._arm_op_lock:
            try:
                if op == "connect":
                    status("connecting…")
                    self.runner.connect_arm()
                    status("connected")
                elif op == "disconnect":
                    status("disconnecting…")
                    self.runner.disconnect_arm()
                    status("disconnected")
                elif op == "lock":
                    status("locking…")
                    self.runner.lock_arm(speed=50)
                    status("locked (hold latched)")
                elif op == "unlock":
                    status("unlocking…")
                    self.runner.unlock_arm()
                    status("UNLOCKED — motors off (re-connect to re-enable)")
            except Exception as exc:
                status(f"FAILED ({op}): {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Trial control
    # ------------------------------------------------------------------
    def _on_start_trial_click(self, _event) -> None:
        if self._trial_running.is_set():
            return
        if self.runner.arm is None or not self.runner.arm_connected:
            self._set_status(self._eval_status_md, "Eval",
                             "start refused: arm not connected")
            return
        trial_idx = int(self._trial_index.value)
        trial_dir = SAVE_ROOT / f"trial_{trial_idx:03d}"
        if trial_dir.exists():
            self._set_status(
                self._eval_status_md, "Eval",
                f"start refused: {trial_dir.name} already exists; bump trial index",
            )
            return
        trial_dir.mkdir(parents=True, exist_ok=False)

        self._trial_dir = trial_dir
        self._trial_step_count = 0
        self._trial_save_idx = 0
        self._trial_best_coverage = 0.0
        self._trial_last_coverage = 0.0
        self._trial_success = False
        self._trial_log_path = trial_dir / "coverage_log.jsonl"
        self._trial_log_path.write_text("")

        self.runner.reset_history()
        self._sticky_action_voxels = None
        self._sticky_action_text = None
        self._scene.clear_action_waypoints()
        self._btn_start_trial.disabled = True
        self._btn_end_trial.disabled = False
        self._trial_running.set()
        print(f"[eval-viser] trial {trial_idx} started; dump dir: {trial_dir.resolve()}")
        self._set_status(self._eval_status_md, "Eval", f"trial {trial_idx} running")

    def _on_end_trial_click(self, _event) -> None:
        if not self._trial_running.is_set():
            return
        # Operator-requested abort: mark as not-success and let the normal
        # finish flow write result.json + flip buttons back.
        print(
            f"[eval-viser] trial {int(self._trial_index.value)} ended by user "
            f"(executed_steps={self._trial_step_count}, "
            f"best_cov={self._trial_best_coverage:.3f})"
        )
        self._trial_success = False
        self._finish_trial(reason="ended by user")

    def _finish_trial(self, *, reason: str) -> None:
        if not self._trial_running.is_set():
            return
        self._trial_running.clear()
        trial_idx = int(self._trial_index.value)
        if self._trial_dir is not None:
            threshold = float(self._cov_threshold.value)
            # Matches diffusion_policy env_runners' `sim_max_reward`:
            #   max over the episode of clip(coverage / success_threshold, 0, 1).
            # `_trial_best_coverage` is already the per-trial max coverage,
            # so the clipped/normalized score is just one min() away.
            normalized_score = float(
                min(self._trial_best_coverage / threshold, 1.0)
                if threshold > 0 else 0.0
            )
            summary = {
                "trial_index": trial_idx,
                "reason": reason,
                "success": bool(self._trial_success),
                "steps": int(self._trial_step_count),
                "frames_saved": int(self._trial_save_idx),
                "last_coverage": float(self._trial_last_coverage),
                "best_coverage": float(self._trial_best_coverage),
                "normalized_score": normalized_score,
                "coverage_threshold": threshold,
                "max_steps": int(self._max_steps.value),
            }
            (self._trial_dir / "result.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(
                f"[eval-viser] trial {trial_idx} finished ({reason}): "
                f"success={summary['success']} steps={summary['steps']} "
                f"best_cov={summary['best_coverage']:.3f} "
                f"last_cov={summary['last_coverage']:.3f} "
                f"score={normalized_score:.3f}"
            )
        try:
            self._trial_index.value = trial_idx + 1
        except Exception:
            pass
        self._btn_start_trial.disabled = False
        self._btn_end_trial.disabled = True
        self._set_status(
            self._eval_status_md, "Eval",
            f"trial {trial_idx} done ({reason}); "
            f"best_cov={self._trial_best_coverage:.3f}",
        )
        self._trial_dir = None
        self._trial_log_path = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _main_loop(self) -> None:
        target_dt = float(self.runner.target_dt)
        while True:
            t0 = time.time()
            running = self._trial_running.is_set()
            execute = (
                running
                and self.runner.arm is not None
                and self.runner.arm_connected
            )
            try:
                snap = self.runner.tick(
                    run_policy=running,
                    execute=execute,
                    on_predicted=self._on_predicted,
                    on_executing=self._on_executing,
                )
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                print(f"[eval-viser] tick error: {err_msg}", flush=True)
                traceback.print_exc()
                self._set_status(
                    self._eval_status_md, "Eval",
                    f"loop error: {err_msg}",
                )
                if running:
                    self._finish_trial(reason=f"error: {err_msg}")
                time.sleep(0.5)
                continue

            self._latest_snap = snap
            coverage = self._compute_coverage(snap) if running else None
            self._render_snapshot(snap, running=running, coverage=coverage)

            if running:
                # max_steps counts ACTIONS actually sent to the arm, not
                # main-loop ticks. Each successful chunk contributes
                # len(snap.action_voxels) waypoints (typically n_action_steps).
                # Warmup / perception-only / failed ticks contribute 0.
                just_executed = (
                    snap.executed and snap.action_voxels is not None
                )
                n_executed_waypoints = (
                    len(snap.action_voxels) if just_executed else 0
                )

                # Coverage gating runs ONLY after a fresh reveal so the
                # decision is based on a clean (pusher-lifted) view of the
                # T-block AprilTags, not the in-flight overlap.
                is_reveal = False
                if just_executed:
                    if bool(self._auto_lift.value):
                        reveal_cov, reveal_snap = self._reveal_and_measure_coverage()
                        if reveal_snap is not None and reveal_cov is not None:
                            snap = reveal_snap
                            coverage = reveal_cov
                            is_reveal = True
                            self._render_snapshot(
                                snap, running=True, coverage=coverage
                            )
                    self._trial_step_count += n_executed_waypoints

                cov_val = float(coverage) if coverage is not None else 0.0
                # Bump-before-save in the non-reveal path is already implicit
                # above (we incremented before saving). Save reflects clean
                # reveal snap on reveal rows, regular snap otherwise.
                self._save_trial_frame(
                    snap, coverage=cov_val, is_reveal=is_reveal,
                )
                self._trial_last_coverage = cov_val
                if cov_val > self._trial_best_coverage:
                    self._trial_best_coverage = cov_val

                threshold = float(self._cov_threshold.value)
                max_steps = int(self._max_steps.value)
                # Auto-lift is an auxiliary view-cleaner: when on, the reveal
                # snap replaces `snap`/`cov_val` above so we measure on the
                # un-occluded image. Either way, coverage >= threshold ends
                # the trial. max_steps still applies in both modes.
                if snap.available and cov_val >= threshold:
                    self._trial_success = True
                    self._finish_trial(
                        reason=f"coverage {cov_val:.3f} >= {threshold:.2f}"
                    )
                elif self._trial_step_count >= max_steps:
                    self._finish_trial(reason=f"max_steps ({max_steps}) reached")
            _sleep_to(t0, target_dt)

    # ------------------------------------------------------------------
    # Snapshot → GUI
    # ------------------------------------------------------------------
    def _on_predicted(self, snap: StepSnapshot) -> None:
        self._render_snapshot(snap, running=True, coverage=self._compute_coverage(snap))

    def _on_executing(self, snap: StepSnapshot) -> None:
        self._render_snapshot(snap, running=True, coverage=self._compute_coverage(snap))

    def _reveal_and_measure_coverage(
        self,
    ) -> tuple[Optional[float], Optional[StepSnapshot]]:
        """Post-chunk reveal: lift the EEF by REVEAL_LIFT_M in
        REVEAL_LIFT_N_STEPS short sub-moves, take one perception-only
        tick for a tag-clear coverage reading, then lower the EEF back
        by replaying the captured sub-trajectories in reverse order
        (each reversed in time).

        Splitting the lift keeps each sub-move as a single-segment
        planner call (segment 1 is exempt from the z safety-clip) and
        leaves trajopt's joint-space smoothness almost no room to bend
        the Cartesian path away from vertical. Per-sub-move trajectories
        are captured from each ``/robot/step_chunk`` response and
        replayed verbatim on the way down so the descent retraces the
        ascent at the JointCtrl-frame level.

        Returns ``(coverage, lifted_snap)``; either side may be ``None``
        if the arm/reader is unavailable or perception failed.
        """
        if (
            self.runner.arm is None
            or not self.runner.arm_connected
            or self.runner.arm_reader is None
        ):
            return None, None
        reading, _, _ = self.runner.arm_reader.get_reading()
        if reading is None:
            return None, None

        original_world = np.asarray(reading.pusher_world, dtype=np.float64).copy()
        step_dz = REVEAL_LIFT_M / float(max(REVEAL_LIFT_N_STEPS, 1))

        speed = int(self.runner.cfg.pusht_service.speed)
        timesteps = int(self.runner.cfg.pusht_service.timesteps)
        dt = float(self.runner.cfg.pusht_service.dt)

        print(
            f"[eval-viser] reveal: lifting EEF +{REVEAL_LIFT_M:.3f}m in "
            f"{REVEAL_LIFT_N_STEPS} sub-moves of {step_dz*1000:.1f}mm each "
            f"(z {original_world[2]:.3f} -> "
            f"{original_world[2] + REVEAL_LIFT_M:.3f})"
        )

        # Each entry is the captured joint-config trajectory for one
        # 1cm sub-move, in execution order. We retain them all so the
        # return leg can replay reversed(sub_trajs[i]) for i from last
        # to first.
        sub_trajectories: list[list] = []
        lift_aborted = False
        for i in range(1, REVEAL_LIFT_N_STEPS + 1):
            sub_target = original_world.copy()
            sub_target[2] = original_world[2] + step_dz * i
            try:
                resp = self.runner.arm.send_targets_world(
                    [sub_target], speed=speed, timesteps=timesteps, dt=dt,
                )
            except Exception as exc:
                print(
                    f"[eval-viser] reveal-lift sub-move {i}/"
                    f"{REVEAL_LIFT_N_STEPS} FAILED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                lift_aborted = True
                break
            traj = resp.get("trajectory") if isinstance(resp, dict) else None
            if not traj:
                print(
                    f"[eval-viser] reveal-lift sub-move {i}/"
                    f"{REVEAL_LIFT_N_STEPS}: response had no 'trajectory' "
                    "field; cannot replay exactly. Aborting lift to avoid "
                    "lopsided up/down paths.",
                    flush=True,
                )
                lift_aborted = True
                break
            sub_trajectories.append(traj)
            print(
                f"[eval-viser] reveal-lift sub-move {i}/"
                f"{REVEAL_LIFT_N_STEPS} OK; captured {len(traj)} configs "
                f"(target z={sub_target[2]:.3f})"
            )

        clean_snap: Optional[StepSnapshot] = None
        coverage: Optional[float] = None
        if not lift_aborted:
            try:
                clean_snap = self.runner.tick(run_policy=False, execute=False)
                coverage = self._compute_coverage(clean_snap)
            except Exception as exc:
                print(
                    f"[eval-viser] reveal perception failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        # Return leg: replay sub-trajectories last-to-first, each reversed.
        # Always attempt to return -- if some sub-moves succeeded before the
        # abort, we still need to bring the arm back down to where we
        # started, so we replay whatever sub_trajectories we did collect.
        replay_ok_count = 0
        for i, traj in enumerate(reversed(sub_trajectories)):
            reversed_traj = list(reversed(traj))
            try:
                self.runner.arm.replay_joint_trajectory(
                    reversed_traj, speed=speed, dt=dt,
                )
                replay_ok_count += 1
            except Exception as exc:
                print(
                    f"[eval-viser] reveal-return replay FAILED at sub-move "
                    f"{i + 1}/{len(sub_trajectories)}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                break
        print(
            f"[eval-viser] reveal-return: replayed {replay_ok_count}/"
            f"{len(sub_trajectories)} sub-trajectories (reversed)"
        )

        # If the replay didn't get the arm back all the way, fall back to a
        # planner-driven return to the original world pose so we don't
        # strand the arm aloft mid-trial.
        if replay_ok_count < len(sub_trajectories):
            print(
                "[eval-viser] reveal-return: invoking planner fallback to "
                "drive EEF back to the original pose",
                flush=True,
            )
            try:
                self.runner.arm.send_targets_world(
                    [original_world], speed=speed, timesteps=timesteps, dt=dt,
                )
            except Exception as exc:
                print(
                    f"[eval-viser] reveal-return (planner fallback) failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        return coverage, clean_snap

    def _compute_coverage(self, snap: StepSnapshot) -> Optional[float]:
        if snap.image_2d is None or self._goal_cell_count == 0:
            return None
        occ = np.asarray(snap.image_2d, dtype=np.float32)
        if occ.ndim == 3:
            occ = occ[0]
        if occ.shape != self._goal_mask.shape:
            return None
        tblock = np.isclose(occ, TBLOCK_VALUE)
        if COVERAGE_TBLOCK_DILATION_VOX > 0:
            tblock = binary_dilation(
                tblock, iterations=int(COVERAGE_TBLOCK_DILATION_VOX)
            )
        intersection = int(np.count_nonzero(tblock & self._goal_mask))
        return intersection / float(self._goal_cell_count)

    def _render_snapshot(
        self,
        snap: StepSnapshot,
        *,
        running: bool,
        coverage: Optional[float],
    ) -> None:
        if snap.action_voxels is not None:
            self._sticky_action_voxels = snap.action_voxels
            self._sticky_action_text = render.format_action(snap)

        display_snap = snap
        if snap.action_voxels is None and self._sticky_action_voxels is not None:
            display_snap = replace(snap, action_voxels=self._sticky_action_voxels)

        if snap.color_preview is not None:
            preview = render.camera_frame(snap.color_preview, self._preview_size())
            self._camera_image.image = preview
            h, w = snap.color_preview.shape[:2]
            self._set_status(
                self._camera_status_md, "Camera",
                f"streaming ({w}x{h}) | dt={snap.dt_s * 1000:.1f}ms",
            )
        elif self.runner.cam is None:
            self._set_status(self._camera_status_md, "Camera", "disabled (--no-camera)")

        self._state_md.content = render.format_state(snap)
        self._action_md.content = (
            self._sticky_action_text
            if self._sticky_action_text is not None
            else render.format_action(snap)
        )
        self._image_handle.image = render.occupancy_2d(
            display_snap,
            resolution_xyz=np.asarray(self.runner.cfg.resolution_xyz, dtype=np.int32),
            cell_px=PREVIEW_CELL_PX,
        )
        self._scene.update(display_snap)

        if running:
            cov_str = f"{coverage:.3f}" if coverage is not None else "—"
            best_str = f"{self._trial_best_coverage:.3f}"
            steps = self._trial_step_count
            max_s = int(self._max_steps.value)
            self._set_status(
                self._eval_status_md, "Eval",
                f"trial {int(self._trial_index.value)} running | step={steps}/{max_s} | "
                f"cov={cov_str} (best={best_str}) | {snap.status}",
            )

    # ------------------------------------------------------------------
    # Per-step dump (mirrors infer_app._save_auto_frame plus coverage log)
    # ------------------------------------------------------------------
    def _save_trial_frame(
        self,
        snap: StepSnapshot,
        *,
        coverage: float,
        is_reveal: bool = False,
    ) -> None:
        if self._trial_dir is None or snap.color_preview is None:
            return

        display_snap = snap
        if snap.action_voxels is None and self._sticky_action_voxels is not None:
            display_snap = replace(snap, action_voxels=self._sticky_action_voxels)

        frame = render.camera_frame(snap.color_preview, self._preview_size())
        occ = render.occupancy_2d(
            display_snap,
            resolution_xyz=np.asarray(self.runner.cfg.resolution_xyz, dtype=np.int32),
            cell_px=PREVIEW_CELL_PX,
        )
        fh = frame.shape[0]
        oh, ow = occ.shape[:2]
        if oh != fh:
            new_ow = max(1, int(round(ow * fh / oh)))
            occ = cv2.resize(occ, (new_ow, fh), interpolation=cv2.INTER_NEAREST)
        combined = np.concatenate([frame, occ], axis=1)
        out_path = self._trial_dir / f"step_{self._trial_save_idx:06d}.png"
        try:
            cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            print(f"[eval-viser] failed to save {out_path}: {exc}", flush=True)
            return

        if self._trial_log_path is not None:
            policy_ms = snap.policy_took_ms
            policy_ms_field = (
                None if (policy_ms is None or math.isnan(policy_ms)) else float(policy_ms)
            )
            entry = {
                "save_idx": int(self._trial_save_idx),
                # snap.step is the runner's monotonic policy-tick counter
                # (incremented per inference attempt, not per executed
                # waypoint), so it's the per-trial inference round.
                "inference_round": int(snap.step),
                # Cumulative count of action waypoints actually sent to the
                # arm so far in this trial -- matches the trial-stop budget
                # (max_steps) and result.json["steps"].
                "executed_steps": int(self._trial_step_count),
                "coverage": float(coverage),
                # True when this row was captured after a post-chunk reveal
                # (EEF lifted REVEAL_LIFT_M off the table) -- only these
                # rows are used for the success-stop decision.
                "is_reveal": bool(is_reveal),
                "available": bool(snap.available),
                "status": snap.status,
                "pusher_voxel": snap.pusher_voxel,
                "tblock_voxel_count": int(snap.tblock_voxel_count),
                "executed": bool(snap.executed),
                "policy_took_ms": policy_ms_field,
            }
            with self._trial_log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

        self._trial_save_idx += 1

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        print(
            f"[eval-viser] GUI at "
            f"http://{self.server.get_host()}:{self.server.get_port()}"
        )
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[eval-viser] interrupted")
        finally:
            self._trial_running.clear()
            self.runner.shutdown()


def _next_trial_index(save_root: Path) -> int:
    """Smallest non-negative N such that `trial_{N:03d}` doesn't exist under save_root."""
    if not save_root.exists():
        return 0
    pat = re.compile(r"^trial_(\d+)$")
    existing = set()
    for entry in save_root.iterdir():
        if not entry.is_dir():
            continue
        m = pat.match(entry.name)
        if m:
            existing.add(int(m.group(1)))
    n = 0
    while n in existing:
        n += 1
    return n
