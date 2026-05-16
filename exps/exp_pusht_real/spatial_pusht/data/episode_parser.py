"""Parse a spatial_episode_v1 JSON into aligned per-frame arrays.

The viewer (episode_viewer.py) reveals that dense, RGB-aligned state lives in
`movements[*].frames[*]`, NOT in the sparse `spatial_history`. Each `frame`
carries:
    step_index, timestamp,
    target_coord = [x, y]                          # 2D voxel action
    move.target_coord = [x, y, z]                  # 3D voxel action (optional)
    move.target_world_m = [x, y, z]                # metric 3D action (optional)
    spatial.pusher_coords = [[x, y]]               # current pusher voxel
    spatial.tblock_coords / tblock_coords_full     # T-block occupied voxels
    spatial.available, spatial.status              # perception health
"""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .occupancy_utils import rasterize_occupancy


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_episode(
    json_path: Path,
    grid_hw: Tuple[int, int] = (128, 128),
    use_full_occupancy: bool = True,
    forward_fill_unavailable: bool = True,
) -> dict:
    """Read one episode JSON and return aligned numpy arrays.

    Returns dict with keys:
        occupancy:     (T, H, W) float32 in {0, 1}
        agent_pos:     (T, 2)    float32, [x, y] voxel coords of pusher
        action:        (T, 2)    float32, [x, y] voxel coords of target
        available:     (T,)      bool, True if perception was healthy that frame
    All T values are aligned 1:1 with the mp4 frames.
    """
    with json_path.open("r", encoding="utf-8") as f:
        ep = json.load(f)

    movements = ep.get("movements", [])
    if not movements:
        raise ValueError(f"No movements found in {json_path}")

    occupancy_key = "tblock_coords_full" if use_full_occupancy else "tblock_coords"

    occ_list: List[np.ndarray] = []
    agent_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    avail_list: List[bool] = []

    last_occ = np.zeros(grid_hw, dtype=np.float32)
    last_agent = np.zeros(2, dtype=np.float32)

    for m in movements:
        for fr in m.get("frames", []):
            sp = fr.get("spatial") or {}
            available = bool(sp.get("available", False))

            tcoords = sp.get(occupancy_key)
            if tcoords is None:
                tcoords = sp.get("tblock_coords")
            occ = rasterize_occupancy(tcoords or [], grid_hw)

            pcoords = sp.get("pusher_coords") or []
            if pcoords:
                agent = np.asarray(pcoords[0][:2], dtype=np.float32)
            else:
                pc = sp.get("pusher_coord")
                agent = np.asarray(pc[:2], dtype=np.float32) if pc else last_agent.copy()

            target = fr.get("target_coord")
            if target is None:
                target = _safe_get(fr, "move", "target_coord", default=[0, 0])
            action = np.asarray(target[:2], dtype=np.float32)

            if (not available) and forward_fill_unavailable:
                occ = last_occ.copy()
                # keep current agent/action as-is; only perception is patched

            occ_list.append(occ)
            agent_list.append(agent)
            action_list.append(action)
            avail_list.append(available)

            last_occ = occ
            last_agent = agent

    if not occ_list:
        raise ValueError(f"No frames extracted from {json_path}")

    return {
        "occupancy": np.stack(occ_list, axis=0).astype(np.float32),
        "agent_pos": np.stack(agent_list, axis=0).astype(np.float32),
        "action": np.stack(action_list, axis=0).astype(np.float32),
        "available": np.asarray(avail_list, dtype=np.bool_),
    }
