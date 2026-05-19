"""Convert a folder of spatial_episode_*.json files into a zarr ReplayBuffer.

Stored data arrays per step:
    occupancy: (N, H, W) float32, T-block binary occupancy grid
    agent_pos: (N, 2)    float32, pusher voxel [x, y]
    action:    (N, 2)    float32, command target voxel [x, y]
Plus meta/episode_ends.
"""
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import zarr
from numcodecs import Blosc

from diffusion_policy.common.replay_buffer import ReplayBuffer

from .episode_parser import parse_episode


def build_replay_buffer(
    json_dir: Path,
    output_zarr: Path,
    grid_hw: Tuple[int, int] = (128, 128),
    use_full_occupancy: bool = True,
    forward_fill_unavailable: bool = True,
    action_source: str = "next_agent",
    glob_pattern: str = "spatial_episode_*.json",
    episodes: Optional[Sequence[str]] = None,
) -> ReplayBuffer:
    json_dir = Path(json_dir)
    json_paths = sorted(json_dir.glob(glob_pattern))
    if episodes is not None:
        wanted = set(episodes)
        json_paths = [p for p in json_paths if p.stem in wanted or p.name in wanted]
    if not json_paths:
        raise FileNotFoundError(f"No episodes under {json_dir} match {glob_pattern}")

    output_zarr = Path(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(output_zarr))
    root = zarr.group(store=store, overwrite=True)
    rb = ReplayBuffer.create_empty_zarr(root=root)

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    chunks = {
        "occupancy": (32, grid_hw[0], grid_hw[1]),
        "agent_pos": (1024, 2),
        "action": (1024, 2),
    }

    total_frames = 0
    for jp in json_paths:
        data = parse_episode(
            jp,
            grid_hw=grid_hw,
            use_full_occupancy=use_full_occupancy,
            forward_fill_unavailable=forward_fill_unavailable,
            action_source=action_source,
        )
        ep_data = {
            "occupancy": data["occupancy"],
            "agent_pos": data["agent_pos"],
            "action": data["action"],
        }
        rb.add_episode(ep_data, chunks=chunks, compressors=compressor)
        total_frames += data["occupancy"].shape[0]
        print(f"  + {jp.name}: {data['occupancy'].shape[0]} frames "
              f"(avail={int(data['available'].sum())}/{len(data['available'])})")

    print(f"Done. {len(json_paths)} episodes, {total_frames} frames -> {output_zarr}")
    return rb
