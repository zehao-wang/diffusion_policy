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

The logged target_coord at frame t is the target the arm has ALREADY reached
by that frame, so target_coord[t] is essentially equal to pusher_coords[t] and
makes a useless training label. Official PushT semantics expect action[t] to
be the command issued AFTER observing state[t] -- i.e. the next state. We
therefore default to action_source="next_agent" and drop the final frame of
each episode (it has no valid next state). Length becomes T-1.
"""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .occupancy_utils import goal_grid_from_movements, rasterize_occupancy


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
    action_source: str = "next_agent",
) -> dict:
    """Read one episode JSON and return aligned numpy arrays.

    Returns dict with keys (N = T for current_target, N = T-1 otherwise):
        occupancy:     (N, H, W) float32 in {0, 1}
        agent_pos:     (N, 2)    float32, [x, y] voxel coords of pusher
        action:        (N, 2)    float32, [x, y] voxel coords of target
        available:     (N,)      bool, True if perception was healthy that frame

    For next_agent / next_target the final frame is dropped because it has no
    valid next-frame label.
    """
    valid_action_sources = {"current_target", "next_target", "next_agent"}
    if action_source not in valid_action_sources:
        raise ValueError(
            f"action_source={action_source!r} must be one of "
            f"{sorted(valid_action_sources)}"
        )

    with json_path.open("r", encoding="utf-8") as f:
        ep = json.load(f)

    movements = ep.get("movements", [])
    if not movements:
        raise ValueError(f"No movements found in {json_path}")

    occupancy_key = "tblock_coords_full" if use_full_occupancy else "tblock_coords"

    # Goal mask is a per-dataset constant (verified byte-identical across all
    # recorded episodes), so we rasterize it once from the first available
    # frame and reuse it for every frame in this episode.
    goal_grid = goal_grid_from_movements(movements, grid_hw,
                                         source_repr=str(json_path))

    occ_list: List[np.ndarray] = []
    agent_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    avail_list: List[bool] = []

    # Initialise the forward-fill buffer to the goal-only state so a leading
    # `available=False` frame still ships the static goal layer.
    last_occ = goal_grid.copy()
    last_agent = np.zeros(2, dtype=np.float32)

    for m in movements:
        for fr in m.get("frames", []):
            sp = fr.get("spatial") or {}
            available = bool(sp.get("available", False))

            tcoords = sp.get(occupancy_key)
            if tcoords is None:
                tcoords = sp.get("tblock_coords")
            occ = rasterize_occupancy(tcoords or [], goal_grid)

            pcoords = sp.get("pusher_coords") or []
            if pcoords:
                agent = np.asarray(pcoords[0][:2], dtype=np.float32)
            else:
                pc = sp.get("pusher_coord")
                agent = np.asarray(pc[:2], dtype=np.float32) if pc else last_agent.copy()

            target = fr.get("target_coord")
            if target is None:
                target = _safe_get(fr, "move", "target_coord", default=[0, 0])
            target = np.asarray(target[:2], dtype=np.float32)

            if (not available) and forward_fill_unavailable:
                occ = last_occ.copy()
                # keep current agent/action as-is; only perception is patched

            occ_list.append(occ)
            agent_list.append(agent)
            target_list.append(target)
            avail_list.append(available)

            last_occ = occ
            last_agent = agent

    if not occ_list:
        raise ValueError(f"No frames extracted from {json_path}")

    targets = np.stack(target_list, axis=0).astype(np.float32)
    agents = np.stack(agent_list, axis=0).astype(np.float32)
    occupancy = np.stack(occ_list, axis=0).astype(np.float32)
    available = np.asarray(avail_list, dtype=np.bool_)

    if action_source == "current_target":
        actions = targets
    elif action_source == "next_target":
        actions = targets[1:]
        agents = agents[:-1]
        occupancy = occupancy[:-1]
        available = available[:-1]
    elif action_source == "next_agent":
        actions = agents[1:]
        agents = agents[:-1]
        occupancy = occupancy[:-1]
        available = available[:-1]
    else:
        raise AssertionError(action_source)

    return {
        "occupancy": occupancy,
        "agent_pos": agents,
        "action": actions,
        "available": available,
    }
