"""Direct PySpin → perception smoke test (no capture service involved).

Grabs one frame from the first Blackfly enumerated by PySpin and runs the
full detect → double-PnP → mesh → voxel chain. Prints PnP reprojection
errors, visible tag IDs, T-block pose, and an agent_pos sanity check.

Run from the diffusion_policy repo root with the system Python (where
PySpin + numpy<2 are installed):

    cd /home/zwa0839/Documents/Projects/robodata_Agilex/packages/diffusion_policy
    /usr/bin/python3 -m exps.exp_pusht_real.spatial_pusht.realrobot.tools.test_perception_direct

Pass a custom pusher world point with ``--pusher x y z`` (metres in the
AprilTag-world frame). Defaults to the workspace centre (0.20, 0.18, 0.03).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Importing PySpin must happen before numpy 2.x sneaks in from elsewhere.
import PySpin

# Repo root: 4 levels up from this file's directory.
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from exps.exp_pusht_real.spatial_pusht.realrobot.perception.apriltag_reconstruction import (  # noqa: E402
    camera_calibration_from_info,
)
from exps.exp_pusht_real.spatial_pusht.realrobot.perception.pointgrey_calibration import (  # noqa: E402
    load_pointgrey_calibration,
    merge_pointgrey_camera_info,
)
from exps.exp_pusht_real.spatial_pusht.realrobot.perception.state_extractor import (  # noqa: E402
    SpatialStateExtractor,
)
from exps.exp_pusht_real.spatial_pusht.data.occupancy_utils import (  # noqa: E402
    load_goal_grid_from_json,
)


_DEFAULT_CALIB = _REPO_ROOT / "data/realrobot/pointgrey_calibration.json"
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data/realrobot/model"
_DEFAULT_GOAL_JSON = (
    _REPO_ROOT
    / "data/spatial_episode_2026051/spatial_episode_20260513_181635_580.json"
)


def _grab_single_frame(serial: str | None, timeout_ms: int) -> np.ndarray:
    sys_inst = PySpin.System.GetInstance()
    cams = sys_inst.GetCameras()
    if cams.GetSize() == 0:
        cams.Clear()
        sys_inst.ReleaseInstance()
        raise RuntimeError("No FLIR/PointGrey cameras enumerated.")

    chosen_index = -1
    for i in range(cams.GetSize()):
        candidate = cams.GetByIndex(i)
        try:
            if serial is None:
                chosen_index = i
                break
            n = PySpin.CStringPtr(candidate.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
            if PySpin.IsReadable(n) and n.GetValue() == serial:
                chosen_index = i
                break
        finally:
            del candidate
    if chosen_index < 0:
        cams.Clear()
        sys_inst.ReleaseInstance()
        raise RuntimeError(f"No camera with serial {serial!r}")
    cam = cams.GetByIndex(chosen_index)

    cam.Init()
    nm = cam.GetNodeMap()
    acq = PySpin.CEnumerationPtr(nm.GetNode("AcquisitionMode"))
    acq.SetIntValue(acq.GetEntryByName("Continuous").GetValue())
    cam.BeginAcquisition()
    try:
        image_result = cam.GetNextImage(timeout_ms)
        frame = np.array(image_result.GetNDArray(), copy=True)
        image_result.Release()
    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cams.Clear()
        sys_inst.ReleaseInstance()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default=str(_DEFAULT_CALIB))
    parser.add_argument("--model-dir", default=str(_DEFAULT_MODEL_DIR))
    parser.add_argument("--device-serial", default=None,
                        help="Blackfly serial (omitted = first camera).")
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--pusher", nargs=3, type=float,
                        default=[0.20, 0.18, 0.03],
                        metavar=("X", "Y", "Z"),
                        help="Pusher world point in metres (AprilTag-world frame).")
    parser.add_argument("--no-kalman", action="store_true")
    parser.add_argument("--goal-json", default=str(_DEFAULT_GOAL_JSON),
                        help="Episode JSON to source the static goal_coords from.")
    args = parser.parse_args()

    calib = load_pointgrey_calibration(args.calibration)
    if calib is None:
        print(f"[ERR] cannot load calibration: {args.calibration}")
        return 1
    info = merge_pointgrey_camera_info(
        {"resolution": {
            "width": int(calib["resolution"]["width"]),
            "height": int(calib["resolution"]["height"]),
        }},
        calib,
    )
    cm, dc, err = camera_calibration_from_info(info)
    if err:
        print(f"[ERR] {err}")
        return 1

    goal_grid = load_goal_grid_from_json(args.goal_json, grid_hw=(128, 128))
    extractor = SpatialStateExtractor(
        model_dir=args.model_dir,
        camera_matrix=cm,
        dist_coeffs=dc,
        bbox_min=np.array([-0.05, -0.10, 0.0]),
        bbox_max=np.array([0.45, 0.45, 0.10]),
        resolution_xyz=np.array([128, 128, 12]),
        goal_grid=goal_grid,
        enable_kalman=not args.no_kalman,
    )

    print(f"[grab] reading 1 frame (serial={args.device_serial or 'any'})")
    t0 = time.time()
    frame = _grab_single_frame(args.device_serial, args.timeout_ms)
    grab_ms = (time.time() - t0) * 1000
    print(f"[grab] {grab_ms:.1f} ms, shape={frame.shape} dtype={frame.dtype} "
          f"min={int(frame.min())} max={int(frame.max())} mean={float(frame.mean()):.1f}")

    pusher_world = np.asarray(args.pusher, dtype=np.float64)
    t1 = time.time()
    obs = extractor.step(frame, pusher_world, timestamp_s=time.time())
    proc_ms = (time.time() - t1) * 1000
    print(f"[proc] {proc_ms:.1f} ms")
    print(f"       available={obs.available}")
    print(f"       status   ={obs.status}")
    print(f"       reproj px world={obs.raw_world_reproj_px:.2f}  "
          f"object={obs.raw_object_reproj_px:.2f}")
    print(f"       visible bg tags    ={obs.visible_background_tags}")
    print(f"       visible object tags={obs.visible_object_tags}")
    if obs.tblock_pose_world is not None:
        t = obs.tblock_pose_world["translation_m"]
        q = obs.tblock_pose_world["wxyz"]
        print(f"       tblock translation_m=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")
        print(f"       tblock wxyz         =({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})")
    print(f"       T-block occupied 2D cells={int(obs.image.sum())}")
    print(f"       agent_pos voxel={obs.agent_pos.tolist()}  (from pusher {pusher_world.tolist()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
