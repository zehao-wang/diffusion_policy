"""Rasterize the tri-valued occupancy grid used by the diffusion policy.

Coordinate convention (matches episode_viewer.py rendering at lines 245-246):
    coord = [x, y]  where x is the ROW index and y is the COLUMN index.
    grid[x, y] = value marks the cell occupied.

Value encoding (kept within [0, 1] so the downstream image-range normalizer
(*2 - 1) maps it to [-1, 1] cleanly):
    background = 0.0
    goal cell  = GOAL_VALUE     (0.5)  -- static, painted once per dataset
    T-block    = TBLOCK_VALUE   (1.0)  -- per-frame, drawn on top so overlap
                                          keeps the dynamic T geometry intact

The goal mask is constant across all frames in this dataset (verified
byte-identical across all 38 recorded episodes); it is therefore precomputed
once via `rasterize_goal_mask` and reused as the base grid for every frame.
"""
import json
from pathlib import Path

import numpy as np

GOAL_VALUE = 0.5
TBLOCK_VALUE = 1.0

# Sentinel value used for empty slots in the padded T-bar coord field
# `tblock_coords (T, K, 2)`. Lives just outside [0, grid-1] so the policy's
# LinearNormalizer maps it cleanly to one end of [-1, 1] and the network can
# learn to ignore it.
TBAR_PAD = -1.0


def pad_tbar_coords_frame(coords, n_pad):
    """Pad/sub-sample one frame's T-bar voxel set into a fixed (n_pad, 2) array.

    Matches the convention used by `replay_buffer_builder._pad_tbar_coords`
    over the whole training set: shorter frames get TBAR_PAD-filled slots,
    longer frames are uniformly sub-sampled (`np.linspace`-spaced indices).
    Shared by the builder and the inference loop so train/eval stay aligned.
    """
    out = np.full((n_pad, 2), TBAR_PAD, dtype=np.float32)
    if coords is None:
        return out
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return out
    coords = coords[:, :2]
    k = coords.shape[0]
    if k == 0:
        return out
    if k > n_pad:
        idx = np.linspace(0, k - 1, n_pad).round().astype(np.int64)
        coords = coords[idx]
        k = n_pad
    out[:k] = coords
    return out


def _paint(grid, coords, value):
    """In-place write `value` at the valid integer cells of `coords`."""
    if coords is None or len(coords) == 0:
        return
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return
    h, w = grid.shape
    xs = arr[:, 0]
    ys = arr[:, 1]
    valid = (xs >= 0) & (xs < h) & (ys >= 0) & (ys < w)
    grid[xs[valid], ys[valid]] = value


def rasterize_goal_mask(goal_coords, grid_hw):
    """Precompute the static goal layer. Call once per dataset / deployment."""
    h, w = grid_hw
    grid = np.zeros((h, w), dtype=np.float32)
    _paint(grid, goal_coords, GOAL_VALUE)
    return grid


def rasterize_occupancy(coords, goal_grid):
    """Overlay the current T-block on top of the precomputed goal grid.

    `goal_grid` is a (H, W) float32 mask returned by `rasterize_goal_mask`.
    T-block cells overwrite goal cells on overlap.
    """
    grid = goal_grid.copy()
    _paint(grid, coords, TBLOCK_VALUE)
    return grid


def goal_grid_from_movements(movements, grid_hw, source_repr=""):
    """Find the first non-empty `goal_coords` inside a spatial_episode_v1
    `movements` list and rasterize it. Raises if nothing is found.

    `source_repr` is only used to make the error message useful when called
    from a path-aware wrapper.
    """
    for m in movements:
        for fr in m.get("frames", []):
            gc = (fr.get("spatial") or {}).get("goal_coords")
            if gc:
                return rasterize_goal_mask(gc, grid_hw)
    suffix = f" in {source_repr}" if source_repr else ""
    raise ValueError(f"No goal_coords found{suffix}")


def load_goal_grid_from_json(json_path, grid_hw):
    """Load a spatial_episode_v1 JSON and rasterize its static goal layer."""
    path = Path(json_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        ep = json.load(f)
    return goal_grid_from_movements(ep.get("movements", []), grid_hw,
                                    source_repr=str(path))
