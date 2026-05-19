"""World frame calibration and coordinate transforms.

Defines a custom world coordinate frame from 4 physical corner points,
and provides transforms between base frame and world frame.
"""

from __future__ import annotations


import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def compute_world_frame(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> dict:
    """Compute a world coordinate frame from 4 rectangle corner points.

    Convention:
        P1 = origin, P2 = +X direction, P3 = opposite corner, P4 = +Y direction.

    All points should be in the robot base frame, in meters.

    Returns:
        dict with keys:
            T_base_from_world: (4,4) transform world->base
            T_world_from_base: (4,4) transform base->world
            x_axis, y_axis, z_axis: (3,) unit vectors in base frame
            origin: (3,) world origin in base frame
            warnings: list of validation warning strings
    """
    p1, p2, p3, p4 = [np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4)]
    warnings = []

    # X axis: P1 -> P2
    x_vec = p2 - p1
    x_len = np.linalg.norm(x_vec)
    if x_len < 1e-6:
        raise ValueError("P1 and P2 are too close — cannot define X axis")
    x_axis = x_vec / x_len

    # Y direction hint: P1 -> P4
    y_vec = p4 - p1
    y_len = np.linalg.norm(y_vec)
    if y_len < 1e-6:
        raise ValueError("P1 and P4 are too close — cannot define Y axis")

    # Z axis: cross(X, Y_hint)  (right-hand rule)
    z_vec = np.cross(x_vec, y_vec)
    z_len = np.linalg.norm(z_vec)
    if z_len < 1e-6:
        raise ValueError("Points are collinear — cannot define a plane")
    z_axis = z_vec / z_len

    # Orthogonal Y axis
    y_axis = np.cross(z_axis, x_axis)

    # Validation: check that edges are roughly orthogonal
    dot_xy = abs(np.dot(x_vec, y_vec) / (x_len * y_len))
    if dot_xy > 0.3:
        warnings.append(
            f"Edges P1->P2 and P1->P4 are not very orthogonal (cos={dot_xy:.2f})"
        )

    # Validation: check that P3 is roughly at the opposite corner
    p3_expected = p1 + x_vec + y_vec
    p3_err = np.linalg.norm(p3 - p3_expected)
    diag_len = np.linalg.norm(p3_expected - p1)
    if diag_len > 1e-6 and p3_err / diag_len > 0.2:
        warnings.append(
            f"P3 deviates from expected opposite corner by {p3_err*1000:.1f}mm "
            f"({p3_err/diag_len*100:.0f}% of diagonal)"
        )

    # Build 4x4 transforms
    # T_base_from_world: columns are [x_axis, y_axis, z_axis, origin]
    R = np.column_stack([x_axis, y_axis, z_axis])
    T_base_from_world = np.eye(4)
    T_base_from_world[:3, :3] = R
    T_base_from_world[:3, 3] = p1  # origin

    T_world_from_base = np.eye(4)
    T_world_from_base[:3, :3] = R.T
    T_world_from_base[:3, 3] = -R.T @ p1

    return {
        "T_base_from_world": T_base_from_world,
        "T_world_from_base": T_world_from_base,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "origin": p1.copy(),
        "warnings": warnings,
    }


def save_world_config(config: dict, path: str | Path) -> None:
    """Save world frame config to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "points_base_m": {
            k: config["points_base_m"][k].tolist()
            if isinstance(config["points_base_m"][k], np.ndarray)
            else config["points_base_m"][k]
            for k in ("p1", "p2", "p3", "p4")
        },
        "T_base_from_world": np.asarray(config["T_base_from_world"]).tolist(),
        "T_world_from_base": np.asarray(config["T_world_from_base"]).tolist(),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[WorldFrame] Saved config -> {path}")


def load_world_config(path: str | Path) -> dict | None:
    """Load world frame config from a JSON file.

    Returns:
        dict with numpy arrays, or None if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    return {
        "timestamp": data["timestamp"],
        "points_base_m": {
            k: np.array(v, dtype=np.float64)
            for k, v in data["points_base_m"].items()
        },
        "T_base_from_world": np.array(data["T_base_from_world"], dtype=np.float64),
        "T_world_from_base": np.array(data["T_world_from_base"], dtype=np.float64),
    }


def point_base_to_world(p_base: np.ndarray, T_world_from_base: np.ndarray) -> np.ndarray:
    """Transform a 3D point from base frame to world frame."""
    p = np.asarray(p_base, dtype=np.float64)
    p_h = np.array([p[0], p[1], p[2], 1.0])
    return (T_world_from_base @ p_h)[:3]


def point_world_to_base(p_world: np.ndarray, T_base_from_world: np.ndarray) -> np.ndarray:
    """Transform a 3D point from world frame to base frame."""
    p = np.asarray(p_world, dtype=np.float64)
    p_h = np.array([p[0], p[1], p[2], 1.0])
    return (T_base_from_world @ p_h)[:3]


def pose_base_to_world(
    pos_base: np.ndarray,
    wxyz_base: np.ndarray,
    T_world_from_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a full pose (position + orientation) from base to world frame.

    Args:
        pos_base: (3,) position in base frame.
        wxyz_base: (4,) quaternion [w,x,y,z] in base frame.
        T_world_from_base: (4,4) transform.

    Returns:
        (pos_world, wxyz_world) — position and quaternion in world frame.
    """
    from scipy.spatial.transform import Rotation

    pos_world = point_base_to_world(pos_base, T_world_from_base)

    R_world_from_base = T_world_from_base[:3, :3]
    wxyz = np.asarray(wxyz_base, dtype=np.float64)
    r_base = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    r_world = Rotation.from_matrix(R_world_from_base) * r_base
    xyzw = r_world.as_quat()
    wxyz_world = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)

    return pos_world, wxyz_world


def pose_world_to_base(
    pos_world: np.ndarray,
    wxyz_world: np.ndarray,
    T_base_from_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a full pose (position + orientation) from world to base frame.

    Args:
        pos_world: (3,) position in world frame.
        wxyz_world: (4,) quaternion [w,x,y,z] in world frame.
        T_base_from_world: (4,4) transform.

    Returns:
        (pos_base, wxyz_base) — position and quaternion in base frame.
    """
    from scipy.spatial.transform import Rotation

    pos_base = point_world_to_base(pos_world, T_base_from_world)

    R_base_from_world = T_base_from_world[:3, :3]
    wxyz = np.asarray(wxyz_world, dtype=np.float64)
    r_world = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    r_base = Rotation.from_matrix(R_base_from_world) * r_world
    xyzw = r_base.as_quat()
    wxyz_base = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)

    return pos_base, wxyz_base


def add_world_frame_visual(server, world_config: dict, *, show_axes: bool = True) -> None:
    """Add calibration rectangle visualization to a viser scene.

    Draws blue border lines and a semi-transparent blue filled rectangle
    using the 4 calibration points (in base frame coordinates).

    Args:
        server: viser.ViserServer instance.
        world_config: dict from load_world_config().
    """
    import trimesh

    pts = world_config["points_base_m"]
    p1, p2, p3, p4 = pts["p1"], pts["p2"], pts["p3"], pts["p4"]

    # Border lines (blue)
    corners = [p1, p2, p3, p4, p1]
    for i in range(4):
        a, b = corners[i], corners[i + 1]
        points = np.array([a, b], dtype=np.float32).reshape(1, 2, 3)
        server.scene.add_line_segments(
            f"/world_frame/border_{i}",
            points=points,
            colors=np.array([66, 133, 244], dtype=np.uint8),
            line_width=3.0,
        )

    # Filled rectangle mesh (blue, 20% opacity)
    vertices = np.array([p1, p2, p3, p4], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = [66, 133, 244, 51]  # ~20% opacity
    server.scene.add_mesh_trimesh("/world_frame/surface", mesh)

    if not show_axes:
        return

    # Origin axes (small, 5cm)
    T = world_config["T_base_from_world"]
    if isinstance(T, list):
        T = np.array(T, dtype=np.float64)
    origin = T[:3, 3]
    x_end = origin + T[:3, 0] * 0.05
    y_end = origin + T[:3, 1] * 0.05
    z_end = origin + T[:3, 2] * 0.05

    for label, end, color in [
        ("x", x_end, [255, 0, 0]),
        ("y", y_end, [0, 255, 0]),
        ("z", z_end, [0, 0, 255]),
    ]:
        pts_line = np.array([origin, end], dtype=np.float32).reshape(1, 2, 3)
        server.scene.add_line_segments(
            f"/world_frame/axis_{label}",
            points=pts_line,
            colors=np.array(color, dtype=np.uint8),
            line_width=2.0,
        )
