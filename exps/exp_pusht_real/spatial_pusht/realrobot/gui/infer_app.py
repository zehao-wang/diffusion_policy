"""Viser GUI for the spatial_pusht_image real-robot inference coordinator.

A single main loop drives perception + (optional) policy + GUI in the
main thread (matching `/tmp/robodata_minye/gui/viser_collector.py`'s
structure). The only worker threads are short-lived ones for arm HTTP
commands so the GUI stays responsive while the service does its thing.

3D scene handling lives in `scene.InferScene`; pure markdown/image
renderers live in `render.py`. This file owns the sidebar layout, the
button callbacks, and the main loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import viser

from ..infer_loop import InferLoopRunner, StepSnapshot, _sleep_to
from . import render
from .scene import InferScene


PREVIEW_CELL_PX = 16
CAMERA_PREVIEW_MAX_W = 640
CAMERA_PREVIEW_MAX_H = 540


class InferViserApp:
    def __init__(
        self,
        runner: InferLoopRunner,
        *,
        host: str = "0.0.0.0",
        port: int = 8013,
    ):
        self.runner = runner
        self.server = viser.ViserServer(
            host=host, port=port, label="spatial_pusht_image inference"
        )

        self._auto_running = threading.Event()
        # Step Once auto-loops up to `n_obs_steps + 1` ticks so warmup
        # doesn't silently swallow the click. Decremented in the main loop.
        self._step_once_remaining = 0
        self._arm_op_lock = threading.Lock()
        self._latest_snap: Optional[StepSnapshot] = None

        # Sticky display state — the latest prediction stays on the GUI
        # (3D scene, 2D occupancy overlay, Action panel) until either a
        # new prediction arrives or the user clicks Reset Obs History.
        self._sticky_action_voxels: Optional[list] = None
        self._sticky_action_text: Optional[str] = None
        self._last_execute_reason: str = "OFF: checkbox unchecked"

        self._build_sidebar()
        self._scene = InferScene(self.server, self.runner)
        self._scene.max_action_waypoints = int(self._viz_action_n.value)

    # ------------------------------------------------------------------
    # Sidebar layout
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
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

        with srv.gui.add_folder("Inference"):
            self._chk_execute = srv.gui.add_checkbox(
                "Execute (send /robot/step)", initial_value=False
            )
            self._btn_start = srv.gui.add_button("Start Auto")
            self._btn_stop = srv.gui.add_button("Stop")
            self._btn_step_once = srv.gui.add_button("Step Once")
            self._btn_reset_hist = srv.gui.add_button("Reset Obs History")
            srv.gui.add_markdown(render.format_policy(self.runner.policy_status))
            self._infer_status_md = srv.gui.add_markdown("**Inference:** idle")
        self._btn_start.on_click(self._on_start_click)
        self._btn_stop.on_click(self._on_stop_click)
        self._btn_step_once.on_click(self._on_step_once_click)
        self._btn_reset_hist.on_click(self._on_reset_hist_click)

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
                # Scene is created after _build_sidebar(); safe at callback time.
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
    # Arm callbacks (each runs in its own daemon thread)
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
    # Inference callbacks
    # ------------------------------------------------------------------
    def _on_start_click(self, _event) -> None:
        if not self.runner.arm_connected and self.runner.arm is not None:
            self._set_status(self._infer_status_md, "Inference", "start refused: arm not connected")
            return
        self._auto_running.set()
        self._set_status(self._infer_status_md, "Inference", "auto running")

    def _on_stop_click(self, _event) -> None:
        self._auto_running.clear()
        self._step_once_remaining = 0
        self._set_status(self._infer_status_md, "Inference", "stopped")

    def _on_step_once_click(self, _event) -> None:
        if self._auto_running.is_set():
            return
        self._step_once_remaining = max(
            self._step_once_remaining, self.runner.n_obs_steps + 1
        )

    def _on_reset_hist_click(self, _event) -> None:
        self.runner.reset_history()
        self._sticky_action_voxels = None
        self._sticky_action_text = None
        self._scene.clear_action_waypoints()
        self._set_status(self._infer_status_md, "Inference", "obs history cleared")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _resolve_execute(self) -> tuple[bool, str]:
        """Return (execute_flag, reason) so the GUI can show *why* exec is on/off."""
        checkbox = bool(self._chk_execute.value)
        if not checkbox:
            return False, "OFF: checkbox unchecked"
        if self.runner.arm is None:
            return False, "OFF: --no-arm"
        if not self.runner.arm_connected:
            return False, "OFF: arm not connected"
        return True, "ON"

    def _main_loop(self) -> None:
        target_dt = float(self.runner.target_dt)
        while True:
            t0 = time.time()

            stepping = self._step_once_remaining > 0
            run_policy = self._auto_running.is_set() or stepping
            execute, execute_reason = self._resolve_execute()
            self._last_execute_reason = execute_reason

            try:
                snap = self.runner.tick(
                    run_policy=run_policy,
                    execute=execute,
                    on_predicted=self._on_predicted,
                    on_executing=self._on_executing,
                )
            except Exception as exc:
                self._step_once_remaining = 0
                self._set_status(
                    self._infer_status_md, "Inference",
                    f"loop error: {type(exc).__name__}: {exc}",
                )
                time.sleep(0.5)
                continue

            if stepping:
                self._step_once_remaining -= 1
                if snap.action_voxels is not None:
                    self._step_once_remaining = 0

            self._latest_snap = snap
            self._render_snapshot(snap, ran_policy=run_policy)
            _sleep_to(t0, target_dt)

    # ------------------------------------------------------------------
    # Snapshot → GUI
    # ------------------------------------------------------------------
    def _on_predicted(self, snap: StepSnapshot) -> None:
        """Runner hook: render the just-predicted waypoints BEFORE execute."""
        self._render_snapshot(snap, ran_policy=True)

    def _on_executing(self, snap: StepSnapshot) -> None:
        """Runner hook: after each waypoint, with fresh perception + arm pose."""
        self._render_snapshot(snap, ran_policy=True)

    def _render_snapshot(self, snap: StepSnapshot, *, ran_policy: bool) -> None:
        # Refresh sticky display state whenever a new prediction arrives.
        if snap.action_voxels is not None:
            self._sticky_action_voxels = snap.action_voxels
            self._sticky_action_text = render.format_action(snap)

        # `display_snap` injects the sticky action so the scene + 2D
        # overlay keep showing the last prediction across perception-only
        # ticks. The State / status lines stay tied to the raw snap.
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

        exec_tag = f"[exec {self._last_execute_reason}]"
        if self._auto_running.is_set():
            hz = 1.0 / max(snap.dt_s, 1e-6)
            extra = (
                f" | policy={snap.policy_took_ms:.0f}ms"
                if snap.action_voxels is not None
                else ""
            )
            self._set_status(
                self._infer_status_md, "Inference",
                f"auto running {exec_tag} | step={snap.step} | {hz:.1f} Hz{extra} | {snap.status}",
            )
        elif ran_policy:
            self._set_status(
                self._infer_status_md, "Inference",
                f"step once {exec_tag}: step={snap.step} {snap.status}",
            )
        else:
            self._set_status(
                self._infer_status_md, "Inference",
                f"idle {exec_tag}",
            )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        print(
            f"[infer-viser] GUI at "
            f"http://{self.server.get_host()}:{self.server.get_port()}"
        )
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[infer-viser] interrupted")
        finally:
            self._auto_running.clear()
            self.runner.shutdown()
