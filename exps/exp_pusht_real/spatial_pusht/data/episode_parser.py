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
from typing import List, Optional, Tuple

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
    tbar_coord_key: str = "tblock_coords",
) -> dict:
    """Read one episode JSON and return aligned numpy arrays.

    Returns dict with keys (N = T for current_target, N = T-1 otherwise):
        occupancy:        (N, H, W)        float32 in {0, 0.5, 1}
        agent_pos:        (N, 2)           float32, [x, y] voxel coords of pusher
        action:           (N, 2)           float32, [x, y] voxel coords of target
        available:        (N,)             bool, True if perception was healthy
        tblock_coords_raw: list[np.ndarray] length N, each (k_t, 2) int16 in voxel
                          coords, sorted by (x, y). Variable per frame (padding is
                          deferred to the builder which knows the dataset-wide N).
        tag_keypoints:    (N, S, 2)        int16 fixed-slot tag-corner voxel xy,
                          slots ordered by ascending (tag_id, corner_idx). S is
                          determined from the first frame that ships
                          `tblock_apriltag_points_world` and is asserted constant
                          for the remainder of the episode. None if no frame
                          carries the field (legacy recordings).
        tag_slot_keys:    list[(tag_id, corner_idx)] length S, or None if
                          tag_keypoints is None. Identifies which slot in
                          tag_keypoints corresponds to which physical corner.

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
    tbar_coord_list: List[np.ndarray] = []
    tag_kp_list: List[np.ndarray] = []  # one (S, 2) entry per frame (filled as we go)
    tag_slot_keys: Optional[List[Tuple[int, int]]] = None  # locked on first frame that has tags

    # Initialise the forward-fill buffer to the goal-only state so a leading
    # `available=False` frame still ships the static goal layer.
    last_occ = goal_grid.copy()
    last_agent = np.zeros(2, dtype=np.float32)
    last_tbar = np.zeros((0, 2), dtype=np.int16)
    last_tag_kp: Optional[np.ndarray] = None  # set once we know S

    for m in movements:
        for fr in m.get("frames", []):
            sp = fr.get("spatial") or {}
            available = bool(sp.get("available", False))

            tcoords = sp.get(occupancy_key)
            if tcoords is None:
                tcoords = sp.get("tblock_coords")
            occ = rasterize_occupancy(tcoords or [], goal_grid)

            # Per-frame variable-length T-bar voxel set for the coord-only
            # policy variant. Always pulled from `tbar_coord_key` (sparse outline
            # by default), independent of which key feeds the occupancy raster,
            # and sorted by (x, y) so a downstream MLP/UNet sees a stable order.
            tbar_raw = sp.get(tbar_coord_key) or []
            if tbar_raw:
                tbar = np.asarray(tbar_raw, dtype=np.int16)[:, :2]
                order = np.lexsort((tbar[:, 1], tbar[:, 0]))
                tbar = tbar[order]
            else:
                tbar = np.zeros((0, 2), dtype=np.int16)

            # Per-frame fixed-slot tag-corner voxel positions for the
            # tag-keypoint policy variant. The first frame that ships tags
            # locks the slot ordering (sorted by (tag_id, corner_idx)); every
            # subsequent frame is re-ordered to that ordering or, if a slot
            # is missing this frame, falls back to forward-fill.
            tag_kp_records = sp.get("tblock_apriltag_points_world") or []
            if tag_slot_keys is None and tag_kp_records:
                tag_slot_keys = sorted({
                    (int(r["tag_id"]), int(r["corner_idx"]))
                    for r in tag_kp_records
                })
                last_tag_kp = np.zeros((len(tag_slot_keys), 2), dtype=np.int16)
            if tag_slot_keys is not None:
                tag_kp = _extract_tag_keypoints(
                    tag_kp_records, tag_slot_keys, last_tag_kp,
                )
            else:
                tag_kp = None  # episode has no tags at all (logged later)

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
                tbar = last_tbar.copy()
                if last_tag_kp is not None:
                    tag_kp = last_tag_kp.copy()
                # keep current agent/action as-is; only perception is patched

            occ_list.append(occ)
            agent_list.append(agent)
            target_list.append(target)
            avail_list.append(available)
            tbar_coord_list.append(tbar)
            tag_kp_list.append(tag_kp)  # may be None until first tag-bearing frame

            last_occ = occ
            last_agent = agent
            last_tbar = tbar
            if tag_kp is not None:
                last_tag_kp = tag_kp

    if not occ_list:
        raise ValueError(f"No frames extracted from {json_path}")

    targets = np.stack(target_list, axis=0).astype(np.float32)
    agents = np.stack(agent_list, axis=0).astype(np.float32)
    occupancy = np.stack(occ_list, axis=0).astype(np.float32)
    available = np.asarray(avail_list, dtype=np.bool_)

    if action_source == "current_target":
        actions = targets
    else:
        # next_target / next_agent: action[t] is the t+1 view of the chosen
        # field, so drop the trailing frame from every aligned array.
        next_field = targets if action_source == "next_target" else agents
        actions = next_field[1:]
        agents = agents[:-1]
        occupancy = occupancy[:-1]
        available = available[:-1]
        tbar_coord_list = tbar_coord_list[:-1]
        tag_kp_list = tag_kp_list[:-1]

    # Stack per-frame tag keypoints into (N, S, 2). Any leading None entries
    # (frames before the first one that shipped tags) are back-filled with
    # the first known slot positions so the array shape is uniform.
    tag_keypoints_arr: Optional[np.ndarray] = None
    if tag_slot_keys is not None:
        seed = next((kp for kp in tag_kp_list if kp is not None), None)
        if seed is None:
            tag_keypoints_arr = None
        else:
            S = seed.shape[0]
            tag_keypoints_arr = np.empty((len(tag_kp_list), S, 2), dtype=np.int16)
            running = seed.copy()
            for i, kp in enumerate(tag_kp_list):
                if kp is not None:
                    running = kp
                tag_keypoints_arr[i] = running

    return {
        "occupancy": occupancy,
        "agent_pos": agents,
        "action": actions,
        "available": available,
        "tblock_coords_raw": tbar_coord_list,
        "tag_keypoints": tag_keypoints_arr,
        "tag_slot_keys": tag_slot_keys,
    }


def _extract_tag_keypoints(records, tag_slot_keys, last_known):
    """Pull (tag_id, corner_idx, coord_xy) records into a (S, 2) int16 array
    in the order given by `tag_slot_keys`. Missing slots fall back to the
    same slot in `last_known` (forward-fill at the slot level)."""
    indexed = {
        (int(r["tag_id"]), int(r["corner_idx"])):
            np.asarray(r["coord_xy"][:2], dtype=np.int16)
        for r in records
    }
    out = last_known.copy() if last_known is not None else \
        np.zeros((len(tag_slot_keys), 2), dtype=np.int16)
    for slot, key in enumerate(tag_slot_keys):
        if key in indexed:
            out[slot] = indexed[key]
    return out
