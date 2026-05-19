#!/usr/bin/env python3
"""World frame calibration via viser GUI.

Point the arm at 4 corners of a physical rectangle to define a world
coordinate frame. The transform is saved to a JSON file for use in
data collection and arm control.

Usage:
    # With physical arm
    python calibrate_world.py

    # Demo mode (no hardware)
    python calibrate_world.py --demo
"""

import argparse
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from utils.urdf_loader import (
    load_piper_urdf,
    can_qpos_to_urdf_cfg_with_gripper,
    fingertip_center_from_urdf_cfg,
)
from utils.world_frame import compute_world_frame, save_world_config


POINT_LABELS = {
    0: "P1 (Origin)",
    1: "P2 (+X)",
    2: "P3 (Opposite)",
    3: "P4 (+Y)",
}


def main():
    parser = argparse.ArgumentParser(description="World frame calibration")
    parser.add_argument("--output", type=str, default="./data/world_config.json",
                        help="Output config file path")
    parser.add_argument("--can-interface", type=str, default="gs_usb",
                        choices=["gs_usb", "socketcan"],
                        help="CAN bus interface type")
    parser.add_argument("--can-channel", type=str, default="can0",
                        help="CAN channel (for socketcan mode)")
    parser.add_argument("--bitrate", type=int, default=1_000_000,
                        help="CAN bus bitrate")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: use simulated points instead of arm")
    args = parser.parse_args()

    # Connect to arm
    arm_reader = None
    if not args.demo:
        from robot.arm_reader import ArmReader
        arm_reader = ArmReader(
            can_interface=args.can_interface,
            can_channel=args.can_channel,
            bitrate=args.bitrate,
        )
        arm_reader.start()
        print("[Calibrate] Arm reader started")
    else:
        print("[Calibrate] Demo mode — using simulated points")

    # Start viser
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    urdf = load_piper_urdf()
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # State
    recorded_points = [None, None, None, None]  # 4 xyz positions in meters
    point_spheres = [None, None, None, None]  # viser handles

    # --- Sidebar GUI ---
    with server.gui.add_folder("Calibration"):
        status_labels = []
        record_btns = []
        for i in range(4):
            label = server.gui.add_markdown(f"**{POINT_LABELS[i]}:** Not recorded")
            status_labels.append(label)
            btn = server.gui.add_button(f"Record {POINT_LABELS[i]}")
            record_btns.append(btn)

        save_btn = server.gui.add_button("Save Calibration")
        save_btn.disabled = True
        cal_status = server.gui.add_markdown("**Status:** Record 4 corner points")

    def _get_eef_position() -> np.ndarray:
        """Get current EEF position in meters from arm or simulated."""
        if args.demo:
            # Demo: return some reasonable positions for testing
            demo_points = [
                np.array([0.2, -0.1, 0.05]),
                np.array([0.4, -0.1, 0.05]),
                np.array([0.4, 0.1, 0.05]),
                np.array([0.2, 0.1, 0.05]),
            ]
            n_recorded = sum(1 for p in recorded_points if p is not None)
            return demo_points[min(n_recorded, 3)]
        else:
            state = arm_reader.get_state()
            cfg = can_qpos_to_urdf_cfg_with_gripper(state.qpos, state.gripper)
            return fingertip_center_from_urdf_cfg(urdf, cfg)

    def _make_record_handler(idx):
        def handler(_event):
            pos = _get_eef_position()
            recorded_points[idx] = pos.copy()
            status_labels[idx].content = (
                f"**{POINT_LABELS[idx]}:** [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m"
            )
            # Show sphere in 3D scene
            colors = [(255, 0, 0), (0, 200, 0), (200, 200, 0), (0, 100, 255)]
            r, g, b = colors[idx]
            if point_spheres[idx] is not None:
                point_spheres[idx].remove()
            point_spheres[idx] = server.scene.add_icosphere(
                f"/cal_point_{idx}",
                radius=0.008,
                color=(r, g, b),
                position=tuple(pos),
            )
            # Check if all 4 recorded
            if all(p is not None for p in recorded_points):
                save_btn.disabled = False
                cal_status.content = "**Status:** All points recorded. Press Save."
            else:
                n = sum(1 for p in recorded_points if p is not None)
                cal_status.content = f"**Status:** {n}/4 points recorded"
            print(f"[Calibrate] Recorded {POINT_LABELS[idx]}: {pos}")
        return handler

    for i in range(4):
        record_btns[i].on_click(_make_record_handler(i))

    def _on_save(_event):
        if not all(p is not None for p in recorded_points):
            cal_status.content = "**Status:** ERROR: Not all points recorded"
            return
        try:
            result = compute_world_frame(*recorded_points)
            for w in result["warnings"]:
                print(f"[Calibrate] WARNING: {w}")

            config = {
                "points_base_m": {
                    "p1": recorded_points[0],
                    "p2": recorded_points[1],
                    "p3": recorded_points[2],
                    "p4": recorded_points[3],
                },
                "T_base_from_world": result["T_base_from_world"],
                "T_world_from_base": result["T_world_from_base"],
            }
            save_world_config(config, args.output)

            warn_text = ""
            if result["warnings"]:
                warn_text = " (with warnings — see terminal)"
            cal_status.content = f"**Status:** Saved to {args.output}{warn_text}"

            # Visualize the frame
            from utils.world_frame import add_world_frame_visual, load_world_config
            loaded = load_world_config(args.output)
            if loaded:
                add_world_frame_visual(server, loaded)

        except ValueError as e:
            cal_status.content = f"**Status:** ERROR: {e}"
            print(f"[Calibrate] ERROR: {e}")

    save_btn.on_click(_on_save)

    print(f"[Calibrate] Server started at http://localhost:{args.port}")
    print("Move the arm to each corner and click Record.\n")

    try:
        while True:
            # Update arm visualization
            if arm_reader is not None:
                state = arm_reader.get_state()
            else:
                from robot.arm_reader import ArmState
                state = ArmState()
            cfg = can_qpos_to_urdf_cfg_with_gripper(state.qpos, state.gripper)
            urdf_vis.update_cfg(cfg)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[Calibrate] Shutting down...")
    finally:
        if arm_reader is not None:
            arm_reader.stop()


if __name__ == "__main__":
    main()
