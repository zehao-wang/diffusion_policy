"""Rasterize T-block occupancy coords into a binary grid.

Coordinate convention (matches episode_viewer.py rendering at lines 245-246):
    coord = [x, y]  where x is the ROW index and y is the COLUMN index.
    grid[x, y] = 1 marks the cell occupied.
"""
import numpy as np


def rasterize_occupancy(coords, grid_hw):
    """coords: iterable of [x, y] voxel indices. Returns float32 (H, W) binary grid."""
    h, w = grid_hw
    grid = np.zeros((h, w), dtype=np.float32)
    if not coords:
        return grid
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return grid
    xs = arr[:, 0]
    ys = arr[:, 1]
    valid = (xs >= 0) & (xs < h) & (ys >= 0) & (ys < w)
    grid[xs[valid], ys[valid]] = 1.0
    return grid
