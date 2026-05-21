"""HTTP client for `pusht_service` + world-frame coordinate helpers.

We talk to the service over raw `urllib` rather than going through
``pusht_client.PushTClient`` so the inference loop can run in Python 3.9
environments (the upstream ``pusht_client`` uses 3.10+ ``X | None`` runtime
syntax in defaults). The endpoint contract is identical to what the
``PushTClient`` would call -- see ``pusht_service/src/server.py``.

Adds two perception-side conveniences on top of the raw HTTP:
  * `get_pusher_world()` -- composes the live link6 pose with a calibrated
    stick-tip-in-EEF offset and transforms into the AprilTag-world frame.
  * `send_target_world(target_world_xyz)` -- inverts that transform and
    calls `/robot/step` with the corresponding delta-EEF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error, request

import numpy as np

from .perception.world_frame import (
    load_world_config,
    point_base_to_world,
    point_world_to_base,
)


def load_world_transform(path: "str | Path") -> dict:
    """Load a base<->world transform from either of minye's two schemas.

    Returns a dict with `T_world_from_base`, `T_base_from_world`, and
    optionally `tip_position_in_eef_m` (only when the file is the
    `arm_world_from_apriltag_points_v1` flavour).

    Recognised inputs:
      * `world_config.json` (4-corner rectangle, `utils/world_frame.py`)
      * `arm_world_calibration.json` (joint solve, `arm_world_from_apriltag_points_v1`)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"World transform file not found: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("type") == "arm_world_from_apriltag_points_v1":
        out = {
            "T_world_from_base": np.asarray(raw["T_world_from_base"], dtype=np.float64),
            "T_base_from_world": np.asarray(raw["T_base_from_world"], dtype=np.float64),
        }
        if "tip_position_in_eef_m" in raw:
            out["tip_position_in_eef_m"] = np.asarray(raw["tip_position_in_eef_m"], dtype=np.float64)
        return out

    # Fallback: 4-corner schema from world_frame.save_world_config.
    cfg = load_world_config(p)
    if cfg is None:
        raise ValueError(f"Unrecognised world transform schema in {p}")
    return {
        "T_world_from_base": np.asarray(cfg["T_world_from_base"], dtype=np.float64),
        "T_base_from_world": np.asarray(cfg["T_base_from_world"], dtype=np.float64),
    }


@dataclass
class ArmReading:
    qpos_rad: np.ndarray              # joint positions
    eef_position_base: np.ndarray     # (3,) link6 origin in base
    eef_wxyz_base: np.ndarray         # (4,) wxyz quaternion
    pusher_world: np.ndarray          # (3,) stick tip in world


class RealRobotArmClient:
    """Combines HTTP plumbing to `pusht_service` with world-frame transforms."""

    def __init__(
        self,
        *,
        service_url: str = "http://localhost:8012",
        world_transform_path: "str | Path",
        tip_in_eef_m_override: Optional[np.ndarray] = None,
        request_timeout_s: float = 10.0,
    ):
        self._service_url = service_url.rstrip("/")
        self._timeout = float(request_timeout_s)

        cfg = load_world_transform(world_transform_path)
        self.T_world_from_base = cfg["T_world_from_base"]
        self.T_base_from_world = cfg["T_base_from_world"]

        # Prefer an explicit override (e.g. from a separate calibration), else
        # use the tip baked into the joint-solve file, else fall back to zero
        # (link6 origin treated as the tip, almost certainly wrong on hardware).
        if tip_in_eef_m_override is not None:
            self.tip_in_eef_m = np.asarray(tip_in_eef_m_override, dtype=np.float64).reshape(3)
        elif "tip_position_in_eef_m" in cfg:
            self.tip_in_eef_m = cfg["tip_position_in_eef_m"]
        else:
            self.tip_in_eef_m = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _post(self, path: str, payload: Optional[dict] = None, *, timeout: Optional[float] = None) -> dict:
        return self._request("POST", path, payload, timeout=timeout)

    def _get(self, path: str, *, timeout: Optional[float] = None) -> dict:
        return self._request("GET", path, None, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict],
        *,
        timeout: Optional[float],
    ) -> dict:
        url = f"{self._service_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=timeout if timeout is not None else self._timeout) as resp:
                return json.loads(resp.read())
        except error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"{method} {path} -> {exc.code}: {body}")
        except error.URLError as exc:
            raise ConnectionError(f"Cannot reach pusht_service at {url}: {exc}")

    # ------------------------------------------------------------------
    # Connection lifecycle (server-owned)
    # ------------------------------------------------------------------
    def connect(self, *, channel: str = "can0", interface: str = "socketcan") -> dict:
        return self._post(
            "/arm/connect",
            {"channel": channel, "interface": interface},
            timeout=30.0,
        )

    def disconnect(self) -> dict:
        return self._post("/arm/disconnect", {}, timeout=10.0)

    def lock_pose(self, speed: int = 50) -> dict:
        return self._post("/arm/lock_pose", {"speed": int(speed)}, timeout=10.0)

    def unlock_pose(self) -> dict:
        """Disable motor torque. **The arm will drop under gravity.**"""
        return self._post("/arm/unlock_pose", {}, timeout=10.0)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def read(self) -> ArmReading:
        state = self._get("/robot/state")
        qpos = np.asarray(state["qpos"], dtype=np.float64)
        eef_xyz_base = np.asarray(state["xyz"], dtype=np.float64).reshape(3)
        eef_wxyz_base = np.asarray(state["wxyz"], dtype=np.float64).reshape(4)

        R_base_from_eef = _wxyz_to_matrix(eef_wxyz_base)
        tip_base = eef_xyz_base + R_base_from_eef @ self.tip_in_eef_m
        pusher_world = point_base_to_world(tip_base, self.T_world_from_base)
        return ArmReading(
            qpos_rad=qpos,
            eef_position_base=eef_xyz_base,
            eef_wxyz_base=eef_wxyz_base,
            pusher_world=pusher_world,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def send_target_world(
        self,
        target_world_xyz: np.ndarray,
        *,
        speed: int = 5,
        timesteps: int = 15,
        dt: float = 0.05,
        final_tol: int = 100,
    ) -> dict:
        """Translate a world-frame waypoint to a delta-EEF /robot/step call.

        Procedure:
            1. Read live EEF pose (base frame) and convert link6 origin -> tip.
            2. Map desired tip_world -> tip_base.
            3. Subtract the stick offset (using live orientation) to get the
               corresponding link6 base target.
            4. Send the delta to /robot/step (server reads live EEF again and
               adds the delta, so drift between steps 1 and 4 is absorbed).
        """
        target_tip_base = point_world_to_base(
            np.asarray(target_world_xyz, dtype=np.float64).reshape(3),
            self.T_base_from_world,
        )
        state = self._get("/robot/state")
        eef_xyz_base = np.asarray(state["xyz"], dtype=np.float64).reshape(3)
        eef_wxyz_base = np.asarray(state["wxyz"], dtype=np.float64).reshape(4)
        R_base_from_eef = _wxyz_to_matrix(eef_wxyz_base)
        target_eef_base = target_tip_base - R_base_from_eef @ self.tip_in_eef_m
        delta = target_eef_base - eef_xyz_base
        # pusht_service /robot/step expects `delta: [dx, dy, dz]` for the 3D
        # planner (dim from planner.waypoint_dim). See pusht_service/src/server.py
        # _handle_robot_step.
        return self._post(
            "/robot/step",
            {
                "delta": [float(delta[0]), float(delta[1]), float(delta[2])],
                "speed": int(speed),
                "timesteps": int(timesteps),
                "dt": float(dt),
                "final_tol": int(final_tol),
            },
            timeout=60.0,
        )

    def send_targets_world(
        self,
        target_worlds_xyz: "list[np.ndarray]",
        *,
        speed: int = 5,
        timesteps: int = 15,
        dt: float = 0.05,
        final_tol: int = 100,
        final_settle_s: float = 0.3,
        trim_intermediate_stops: Optional[bool] = None,
        timeout: float = 120.0,
    ) -> dict:
        """Chunked version of `send_target_world`: all targets in one call.

        Sends every target as an independent delta (each measured from the
        live EEF read at call time, same as `/robot/step`). The server plans
        one continuous trajectory through all targets and streams it to the
        arm without stopping between waypoints — much smoother than calling
        `send_target_world` in a Python loop.

        Note: each delta uses the live EEF orientation read once at call
        time. If orientation drifts during execution the later targets
        accumulate that drift; for short pusht chunks this is negligible.
        """
        if len(target_worlds_xyz) == 0:
            return {"status": "empty", "num_targets": 0}

        state = self._get("/robot/state")
        eef_xyz_base = np.asarray(state["xyz"], dtype=np.float64).reshape(3)
        eef_wxyz_base = np.asarray(state["wxyz"], dtype=np.float64).reshape(4)
        R_base_from_eef = _wxyz_to_matrix(eef_wxyz_base)
        tip_offset_base = R_base_from_eef @ self.tip_in_eef_m

        deltas = []
        for t_world in target_worlds_xyz:
            target_tip_base = point_world_to_base(
                np.asarray(t_world, dtype=np.float64).reshape(3),
                self.T_base_from_world,
            )
            target_eef_base = target_tip_base - tip_offset_base
            d = target_eef_base - eef_xyz_base
            deltas.append([float(d[0]), float(d[1]), float(d[2])])

        payload = {
            "deltas": deltas,
            "speed": int(speed),
            "timesteps": int(timesteps),
            "dt": float(dt),
            "final_tol": int(final_tol),
            "final_settle_s": float(final_settle_s),
        }
        # Only include the override if the caller set it explicitly; otherwise
        # let the server fall back to the pusht_service config default.
        if trim_intermediate_stops is not None:
            payload["trim_intermediate_stops"] = bool(trim_intermediate_stops)
        return self._post("/robot/step_chunk", payload, timeout=timeout)

    def replay_joint_trajectory(
        self,
        trajectory: "list[list[float]] | list[np.ndarray] | np.ndarray",
        *,
        speed: int = 5,
        dt: float = 0.05,
        final_tol: int = 100,
        final_settle_s: float = 0.3,
        final_timeout: float = 5.0,
        timeout: float = 120.0,
    ) -> dict:
        """Stream a caller-supplied joint trajectory through the arm verbatim.

        Bypasses the planner entirely -- the trajectory is streamed config-by-
        config at the given dt. Intended for replaying (typically reversing) a
        trajectory previously returned by ``send_targets_world`` so that
        lift/reveal/return motions follow the exact same joint path in both
        directions rather than re-planning each leg independently.
        """
        rows: list[list[float]] = []
        for cfg in trajectory:
            arr = np.asarray(cfg, dtype=np.float64).reshape(-1)
            rows.append([float(v) for v in arr.tolist()])
        if not rows:
            return {"status": "empty", "num_configs": 0}
        payload = {
            "trajectory": rows,
            "speed": int(speed),
            "dt": float(dt),
            "final_tol": int(final_tol),
            "final_settle_s": float(final_settle_s),
            "final_timeout": float(final_timeout),
        }
        return self._post("/robot/execute_trajectory", payload, timeout=timeout)


def _wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
