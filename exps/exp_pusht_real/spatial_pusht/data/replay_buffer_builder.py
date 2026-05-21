"""Convert a folder of spatial_episode_*.json files into a zarr ReplayBuffer.

Stored data arrays per step:
    occupancy:      (N, H, W) float32, T-block + goal tri-valued occupancy grid
    agent_pos:      (N, 2)    float32, pusher voxel [x, y]
    action:         (N, 2)    float32, command target voxel [x, y]
    tblock_coords:  (N, K, 2) float32, padded T-bar voxel coord set per frame.
                              K = dataset-wide max-voxel-count (auto-detected),
                              shorter frames are padded with sentinel TBAR_PAD
                              (= -1) so the model can learn to ignore them.
    tag_keypoints:  (N, S, 2) float32, fixed-slot tag-corner voxel xy per frame.
                              S = number of (tag_id, corner_idx) slots; ordering
                              is determined from the first parsed episode and
                              asserted consistent across the dataset. Field is
                              omitted if no episode logs apriltag data.
Plus meta/episode_ends.
"""
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import zarr
from numcodecs import Blosc

from diffusion_policy.common.replay_buffer import ReplayBuffer

from .episode_parser import parse_episode
from .occupancy_utils import TBAR_PAD, pad_tbar_coords_frame


def _pad_tbar_coords(coord_list: List[np.ndarray], n_pad: int) -> np.ndarray:
    """Stack a variable-length list of (k_t, 2) voxel sets into (T, n_pad, 2).

    Each frame is independently padded/sub-sampled via `pad_tbar_coords_frame`,
    which is shared with the inference loop so train/eval treat outliers the
    same way.
    """
    return np.stack(
        [pad_tbar_coords_frame(c, n_pad) for c in coord_list],
        axis=0,
    )


def build_replay_buffer(
    json_dir: Path,
    output_zarr: Path,
    grid_hw: Tuple[int, int] = (128, 128),
    use_full_occupancy: bool = True,
    forward_fill_unavailable: bool = True,
    action_source: str = "next_agent",
    glob_pattern: str = "spatial_episode_*.json",
    episodes: Optional[Sequence[str]] = None,
    tbar_coord_key: str = "tblock_coords",
    tbar_pad_n: Optional[int] = None,
) -> ReplayBuffer:
    json_dir = Path(json_dir)
    json_paths = sorted(json_dir.glob(glob_pattern))
    if episodes is not None:
        wanted = set(episodes)
        json_paths = [p for p in json_paths if p.stem in wanted or p.name in wanted]
    if not json_paths:
        raise FileNotFoundError(f"No episodes under {json_dir} match {glob_pattern}")

    # Pass 1: parse every episode and remember the variable-length coord lists
    # so we can size the padded array to the dataset-wide max.
    parsed = []
    for jp in json_paths:
        data = parse_episode(
            jp,
            grid_hw=grid_hw,
            use_full_occupancy=use_full_occupancy,
            forward_fill_unavailable=forward_fill_unavailable,
            action_source=action_source,
            tbar_coord_key=tbar_coord_key,
        )
        parsed.append((jp, data))

    observed_max = max(
        (c.shape[0] for _, data in parsed for c in data["tblock_coords_raw"]),
        default=0,
    )
    if tbar_pad_n is None:
        n_pad = max(observed_max, 1)
    else:
        n_pad = int(tbar_pad_n)
        if n_pad < observed_max:
            print(
                f"[warn] tbar_pad_n={n_pad} < observed max {observed_max}; "
                f"longer frames will be uniformly sub-sampled.")
    print(f"tblock_coords padding length K = {n_pad} "
          f"(observed max = {observed_max}, key = {tbar_coord_key!r})")

    # tag-keypoint slot ordering: lock to the first episode that has tags,
    # then assert every later episode reports the same ordering. Skipped if
    # no episode logs `tblock_apriltag_points_world`.
    tag_slot_keys: Optional[List[Tuple[int, int]]] = None
    for jp, data in parsed:
        keys = data.get("tag_slot_keys")
        if not keys:
            continue
        if tag_slot_keys is None:
            tag_slot_keys = list(keys)
            print(f"tag_keypoints S = {len(tag_slot_keys)} slots, ordering = "
                  f"{tag_slot_keys}")
            continue
        if list(keys) != tag_slot_keys:
            raise ValueError(
                f"tag slot ordering mismatch in {jp.name}: "
                f"expected {tag_slot_keys}, got {list(keys)}"
            )
    write_tag_keypoints = tag_slot_keys is not None

    # Pass 2: write zarr with the now-known K.
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
        "tblock_coords": (256, n_pad, 2),
    }
    if write_tag_keypoints:
        chunks["tag_keypoints"] = (1024, len(tag_slot_keys), 2)

    total_frames = 0
    for jp, data in parsed:
        ep_data = {
            "occupancy": data["occupancy"],
            "agent_pos": data["agent_pos"],
            "action": data["action"],
            "tblock_coords": _pad_tbar_coords(data["tblock_coords_raw"], n_pad),
        }
        if write_tag_keypoints:
            tk = data.get("tag_keypoints")
            if tk is None:
                raise ValueError(
                    f"{jp.name}: dataset is supposed to have tag_keypoints "
                    f"(other episodes do), but this one returned None.")
            ep_data["tag_keypoints"] = tk.astype(np.float32)
        rb.add_episode(ep_data, chunks=chunks, compressors=compressor)
        n = data["occupancy"].shape[0]
        total_frames += n
        print(f"  + {jp.name}: {n} frames "
              f"(avail={int(data['available'].sum())}/{len(data['available'])})")

    print(f"Done. {len(json_paths)} episodes, {total_frames} frames -> {output_zarr}")
    return rb
