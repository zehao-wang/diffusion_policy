"""LeRobot → Diffusion Policy dataset adapter.

Reads a LeRobot v3 dataset (parquet + AV1 video files) and exposes it as a
``BaseImageDataset`` compatible with the official Diffusion Policy training loop.

Expected dataset layout (produced by record_piper.py / lerobot 0.5.0):

    {dataset_root}/
      meta/info.json
      data/chunk-000/file-000.parquet   # all episodes in one file
      videos/
        observation.images.realsense/chunk-000/file-000.mp4  (ep 0)
        observation.images.realsense/chunk-000/file-001.mp4  (ep 1) ...
        observation.images.zed2i/chunk-000/file-000.mp4
        ...

shape_meta expected by the task config (must match what this loader produces):

    obs:
      realsense:     shape: [3, H, W]   type: rgb
      zed2i:         shape: [3, H, W]   type: rgb
      robot_state:   shape: [7]         type: low_dim
    action:          shape: [7]
"""

from typing import Dict, Optional
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.pytorch_util import dict_apply


# ---------------------------------------------------------------------------
# Video decoding
# ---------------------------------------------------------------------------

def _decode_video_av(video_path: str, out_hw: tuple) -> np.ndarray:
    """Decode all frames of an AV1 (or any ffmpeg-supported) video.

    Args:
        video_path: path to the .mp4 file.
        out_hw: (height, width) to resize each frame to.

    Returns:
        (T, H, W, 3) uint8 numpy array in RGB order.
    """
    out_h, out_w = out_hw
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="rgb24")          # (H, W, 3) uint8
            if img.shape[:2] != (out_h, out_w):
                img = cv2.resize(img, (out_w, out_h),
                                 interpolation=cv2.INTER_AREA)
            frames.append(img)
    return np.stack(frames, axis=0)                          # (T, H, W, 3)


# ---------------------------------------------------------------------------
# ReplayBuffer builder
# ---------------------------------------------------------------------------

def _build_replay_buffer(
    dataset_root: Path,
    image_keys: list,          # e.g. ['realsense', 'zed2i']
    image_size_hw: tuple,      # (H, W) to decode video at
    store,
) -> ReplayBuffer:
    """Read lerobot dataset and write it into a zarr ReplayBuffer.

    The parquet file stores all episodes in a single file, discriminated by
    `episode_index`.  Video files are named file-{episode_index:03d}.mp4.
    """
    # Map lerobot image key → official DP short name
    # e.g. 'observation.images.realsense' → 'realsense'
    lerobot_image_keys = {k: f"observation.images.{k}" for k in image_keys}

    # Read meta
    meta = json.loads((dataset_root / "meta" / "info.json").read_text())
    num_episodes: int = meta["total_episodes"]
    chunks_size: int = meta.get("chunks_size", 1000)

    # Load all parquet files (one chunk per file)
    parquet_dfs = []
    chunk_idx = 0
    while True:
        p = dataset_root / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"
        if not p.exists():
            break
        parquet_dfs.append(pd.read_parquet(p))
        chunk_idx += 1
    df = pd.concat(parquet_dfs, ignore_index=True)

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)

    for ep_idx in range(num_episodes):
        ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        T = len(ep_df)
        if T == 0:
            print(f"[lerobot_dataset] WARNING: episode {ep_idx} has 0 frames, skipping.")
            continue

        # Low-dim data
        states  = np.stack(ep_df["observation.state"].values).astype(np.float32)
        actions = np.stack(ep_df["action"].values).astype(np.float32)

        episode_data = {
            "robot_state": states,   # (T, 7)
            "action":      actions,  # (T, 7)
        }

        # Video data — locate file: episode_index corresponds to file-{ep_idx:03d}.mp4
        chunk_for_ep = ep_idx // chunks_size
        file_idx     = ep_idx % chunks_size
        for short_key, lerobot_key in lerobot_image_keys.items():
            video_path = (
                dataset_root
                / "videos"
                / lerobot_key
                / f"chunk-{chunk_for_ep:03d}"
                / f"file-{file_idx:03d}.mp4"
            )
            if not video_path.exists():
                raise FileNotFoundError(
                    f"Video not found: {video_path}\n"
                    "Make sure the dataset was recorded with both cameras."
                )
            frames = _decode_video_av(str(video_path), image_size_hw)  # (T, H, W, 3)
            if len(frames) != T:
                # Truncate to the shorter of the two (minor sync issue)
                n = min(len(frames), T)
                frames  = frames[:n]
                episode_data["robot_state"] = episode_data["robot_state"][:n]
                episode_data["action"]       = episode_data["action"][:n]
                T = n
            episode_data[short_key] = frames

        replay_buffer.add_episode(episode_data, compressors="disk")
        print(f"[lerobot_dataset] Episode {ep_idx:03d}: {T} frames loaded.")

    return replay_buffer


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LeRobotImageDataset(BaseImageDataset):
    """Official Diffusion Policy-compatible dataset for lerobot v3 recordings.

    Args:
        shape_meta: Hydra shape_meta dict (obs keys + action shape).
        dataset_path: Path to the lerobot dataset root (contains meta/, data/, videos/).
        horizon: Sequence length returned per sample.
        pad_before: Frames to pad before each episode start.
        pad_after: Frames to pad after each episode end.
        n_obs_steps: How many observation steps to return (None = all).
        use_cache: Cache the decoded zarr buffer to disk for faster restarts.
        seed: RNG seed for train/val split.
        val_ratio: Fraction of episodes held out for validation (0 = no val).
        max_train_episodes: Cap on training episodes (None = all).
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        use_cache: bool = True,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ):
        dataset_root = Path(os.path.expanduser(dataset_path))
        assert dataset_root.is_dir(), f"Dataset path not found: {dataset_root}"

        # Parse shape_meta
        rgb_keys, lowdim_keys = [], []
        image_size_hw = None
        for key, meta in shape_meta["obs"].items():
            if meta["type"] == "rgb":
                rgb_keys.append(key)
                if image_size_hw is None:
                    # shape is [C, H, W]
                    image_size_hw = (meta["shape"][1], meta["shape"][2])
            elif meta["type"] == "low_dim":
                lowdim_keys.append(key)

        assert image_size_hw is not None, "No rgb key found in shape_meta.obs"

        # ---- Load or build zarr cache ----
        replay_buffer: ReplayBuffer
        if use_cache:
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode()).hexdigest()
            cache_zarr_path = str(dataset_root / f"{shape_meta_hash}.zarr.zip")
            cache_lock_path = cache_zarr_path + ".lock"
            print(f"[lerobot_dataset] Acquiring cache lock: {cache_zarr_path}")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    print("[lerobot_dataset] Cache miss — decoding videos (first run, takes a few minutes)...")
                    try:
                        replay_buffer = _build_replay_buffer(
                            dataset_root, rgb_keys, image_size_hw,
                            store=zarr.MemoryStore()
                        )
                        print("[lerobot_dataset] Saving cache to disk...")
                        with zarr.ZipStore(cache_zarr_path) as zs:
                            replay_buffer.save_to_store(zs)
                        print("[lerobot_dataset] Cache saved.")
                    except Exception:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise
                else:
                    print("[lerobot_dataset] Loading cached ReplayBuffer from disk...")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zs:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zs, store=zarr.MemoryStore()
                        )
                    print("[lerobot_dataset] Cache loaded.")
        else:
            replay_buffer = _build_replay_buffer(
                dataset_root, rgb_keys, image_size_hw,
                store=zarr.MemoryStore()
            )

        # ---- Train / val split ----
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        if max_train_episodes is not None:
            n = min(max_train_episodes, train_mask.sum())
            idxs = np.where(train_mask)[0][:n]
            train_mask = np.zeros_like(train_mask)
            train_mask[idxs] = True

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.replay_buffer  = replay_buffer
        self.sampler        = sampler
        self.rgb_keys       = rgb_keys
        self.lowdim_keys    = lowdim_keys
        self.val_mask       = val_mask
        self.horizon        = horizon
        self.n_obs_steps    = n_obs_steps
        self.pad_before     = pad_before
        self.pad_after      = pad_after

    # ------------------------------------------------------------------
    def get_validation_dataset(self) -> "LeRobotImageDataset":
        val = copy.copy(self)
        val.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val.val_mask = ~self.val_mask
        return val

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)   # None → take all

        obs_dict: dict = {}
        for key in self.rgb_keys:
            # (T, H, W, C) uint8 → (T, C, H, W) float32 in [0, 1]
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)

        return {
            "obs":    dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
