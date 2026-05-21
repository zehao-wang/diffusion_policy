"""Three diffusion-policy datasets over the same occupancy zarr.

The split mirrors how the upstream repo separates `pusht_image` vs `pusht_lowdim`
-- only the state representation changes; the rest of the training logic should
stay as close to those two canonical baselines as possible.

Variant A  (image, mirrors PushTImageDataset / pusht_image.yaml):
    BaseImageDataset subclass.
    Returns: obs = {image: (T, 1, 128, 128), agent_pos: (T, 2)},
             action = (T, 2).
    Trained with the hybrid image workspace (CNN sub-encoder + low_dim concat).

Variant B  (flat lowdim, mirrors PushTLowdimDataset / pusht_lowdim.yaml):
    BaseLowdimDataset subclass.
    Returns: obs = (T, 128*128 + 2) [occupancy.flatten() || agent_pos],
             action = (T, 2).
    Trained with the lowdim workspace (raw obs vector -> U-Net global cond).

Variant C  (T-bar coords lowdim, no occupancy raster):
    BaseLowdimDataset subclass.
    Returns: obs = (T, K*2 + 2) [tblock_coords.flatten() || agent_pos],
             action = (T, 2).
    Uses the dataset-wide-padded `tblock_coords` zarr field (sentinel = -1) so
    every frame has a fixed K voxel slots. Trained with the same U-Net lowdim
    workspace as variant B, but with a far smaller obs vector.
"""
import copy
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class _SpatialReplayBufferLoader:
    """Shared zarr loading + episode masking + sequence sampling.

    Pulled out as a mixin so the two dataset variants can inherit from their
    respective canonical base class (BaseImageDataset / BaseLowdimDataset)
    without duplicating the boilerplate.
    """

    def _init_from_zarr(
        self,
        zarr_path: str,
        horizon: int,
        pad_before: int,
        pad_after: int,
        seed: int,
        val_ratio: float,
        max_train_episodes,
        stride: int = 1,
        keys=("occupancy", "agent_pos", "action"),
    ):
        assert stride >= 1, f"stride must be >= 1, got {stride}"
        self._zarr_keys = tuple(keys)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=list(keys)
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # Window-internal stride: each horizon-step is `stride` raw frames apart.
        # Sampler pulls `raw_len` consecutive frames, __getitem__ picks every
        # stride-th. pad_* are given in horizon-time and scaled to raw frames.
        self.stride = int(stride)
        raw_len = (horizon - 1) * self.stride + 1
        raw_pad_before = pad_before * self.stride
        raw_pad_after = pad_after * self.stride

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=raw_len,
            pad_before=raw_pad_before,
            pad_after=raw_pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._raw_len = raw_len
        self._raw_pad_before = raw_pad_before
        self._raw_pad_after = raw_pad_after

    def _val_split(self):
        """Shallow copy with the sampler restricted to the held-out episodes."""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self._raw_len,
            pad_before=self._raw_pad_before,
            pad_after=self._raw_pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def _stride_sample(self, sample: dict) -> dict:
        """Pick every stride-th frame of each array in `sample`."""
        if self.stride == 1:
            return sample
        return {k: v[::self.stride] for k, v in sample.items()}

    def get_validation_dataset(self):
        return self._val_split()

    def __len__(self):
        return len(self.sampler)

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Every variant ultimately follows the same pipeline: pull a raw window
        # from the sampler, stride-subsample it, and hand it off to a variant-
        # specific `_sample_to_data`. Subclasses just implement that one method.
        sample = self._stride_sample(self.sampler.sample_sequence(idx))
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)

    def _fit_lowdim_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Fit one LinearNormalizer across the full (obs, action) the way
        PushTLowdimDataset does. Lowdim variants delegate `get_normalizer` here;
        image variants override `get_normalizer` directly."""
        all_data = self._sample_to_data({
            k: self.replay_buffer[k][:] for k in self._zarr_keys
        })
        normalizer = LinearNormalizer()
        normalizer.fit(data=all_data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer


# --------------------------------------------------------------------------- #
#                Variant A: occupancy as a single-channel image               #
# --------------------------------------------------------------------------- #
class SpatialPushTOccupancyImageDataset(_SpatialReplayBufferLoader, BaseImageDataset):
    """Mirrors PushTImageDataset; occupancy replaces the RGB image.

    obs = {
        image: (T, 1, 128, 128),   # binary T-block occupancy, 1-channel
        agent_pos: (T, 2),         # pusher voxel coords
    }
    action = (T, 2)
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes=None,
        stride: int = 1,
    ):
        super().__init__()
        self._init_from_zarr(zarr_path, horizon, pad_before, pad_after,
                             seed, val_ratio, max_train_episodes,
                             stride=stride)

    def _sample_to_data(self, sample):
        occ = sample["occupancy"].astype(np.float32)[:, None, :, :]  # (T, 1, H, W)
        return {
            "obs": {
                "image": occ,
                "agent_pos": sample["agent_pos"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        # Mirrors PushTImageDataset.get_normalizer: low_dim keys + identity for image.
        data = {
            "action": self.replay_buffer["action"][:],
            "agent_pos": self.replay_buffer["agent_pos"][:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer


# --------------------------------------------------------------------------- #
#               Variant B: flat occupancy + agent_pos as one obs              #
# --------------------------------------------------------------------------- #
class SpatialPushTOccupancyFlatDataset(_SpatialReplayBufferLoader, BaseLowdimDataset):
    """Mirrors PushTLowdimDataset; obs vector = [occupancy.flatten() || agent_pos].

    PushTLowdimDataset returns obs of shape (T, n_kp*2 + 2) by concatenating
    flattened keypoints with agent_pos. We do the same with the occupancy grid.

    obs    = (T, 128*128 + 2)
    action = (T, 2)
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes=None,
        occupancy_pool: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self._init_from_zarr(zarr_path, horizon, pad_before, pad_after,
                             seed, val_ratio, max_train_episodes,
                             stride=stride)
        # Optional non-overlapping avg-pool over the occupancy grid before
        # flattening. pool=1 reproduces the original (T, 128*128+2) obs.
        assert occupancy_pool >= 1
        H, W = self.replay_buffer["occupancy"].shape[-2:]
        assert H % occupancy_pool == 0 and W % occupancy_pool == 0, (
            f"occupancy_pool={occupancy_pool} must divide grid ({H}, {W})")
        self.occupancy_pool = occupancy_pool

    def _pool_occupancy(self, occ: np.ndarray) -> np.ndarray:
        p = self.occupancy_pool
        if p == 1:
            return occ
        T_, H, W = occ.shape
        return occ.reshape(T_, H // p, p, W // p, p).mean(axis=(2, 4))

    def _sample_to_data(self, sample):
        occ = self._pool_occupancy(sample["occupancy"].astype(np.float32))
        T_ = occ.shape[0]
        obs = np.concatenate([
            occ.reshape(T_, -1),                       # (T, (H/p)*(W/p))
            sample["agent_pos"].astype(np.float32),    # (T, 2)
        ], axis=-1)
        return {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }

    # Mirrors PushTLowdimDataset: one LinearNormalizer over the full (obs, action).
    # Binary occupancy columns get min=0/max=1 -> identity scaling; agent_pos and
    # action get scaled to [-1, 1] like upstream pusht_lowdim.
    get_normalizer = _SpatialReplayBufferLoader._fit_lowdim_normalizer


# --------------------------------------------------------------------------- #
#       Variant C: padded T-bar voxel coords + agent_pos as one obs vec       #
# --------------------------------------------------------------------------- #
class SpatialPushTTBarCoordsDataset(_SpatialReplayBufferLoader, BaseLowdimDataset):
    """Lowdim policy variant whose state is the T-bar voxel set itself.

    The builder pads every frame's T-block voxel coords to a fixed K (the
    dataset-wide observed max), with sentinel -1 for empty slots. We flatten
    that and concatenate the pusher position to form a small dense obs vector,
    skipping the rasterized occupancy entirely.

    obs    = (T, K*2 + 2)   # [tblock_coords.flatten() || agent_pos]
    action = (T, 2)
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes=None,
        stride: int = 1,
    ):
        super().__init__()
        self._init_from_zarr(zarr_path, horizon, pad_before, pad_after,
                             seed, val_ratio, max_train_episodes,
                             stride=stride,
                             keys=("tblock_coords", "agent_pos", "action"))
        self.tbar_pad_n = int(self.replay_buffer["tblock_coords"].shape[1])

    def _sample_to_data(self, sample):
        coords = sample["tblock_coords"].astype(np.float32)  # (T, K, 2)
        T_ = coords.shape[0]
        obs = np.concatenate([
            coords.reshape(T_, -1),                       # (T, K*2)
            sample["agent_pos"].astype(np.float32),       # (T, 2)
        ], axis=-1)
        return {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }

    # Same pattern as variant B. The -1 sentinel sits just outside [0, grid-1],
    # so under "limits" mode it lands exactly at the -1 end of the normalized
    # range and stays distinguishable from real voxels.
    get_normalizer = _SpatialReplayBufferLoader._fit_lowdim_normalizer


# --------------------------------------------------------------------------- #
#       Variant D: fixed-slot AprilTag keypoints + agent_pos as one obs       #
# --------------------------------------------------------------------------- #
class SpatialPushTTagKeypointsDataset(_SpatialReplayBufferLoader, BaseLowdimDataset):
    """Lowdim policy variant whose state is the per-frame voxel xy of the
    object-tag corners.

    Mirrors the upstream `PushTLowdimDataset` 9-keypoint design exactly: every
    frame ships the same fixed S keypoints (one per (tag_id, corner_idx) slot,
    ordering set at build time), so the U-Net global-cond MLP sees a stable
    layout — no padding, no sentinel.

    obs    = (T, S*2 + 2)   # [tag_keypoints.flatten() || agent_pos]
    action = (T, 2)
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes=None,
        stride: int = 1,
    ):
        super().__init__()
        self._init_from_zarr(zarr_path, horizon, pad_before, pad_after,
                             seed, val_ratio, max_train_episodes,
                             stride=stride,
                             keys=("tag_keypoints", "agent_pos", "action"))
        self.n_tag_keypoints = int(self.replay_buffer["tag_keypoints"].shape[1])

    def _sample_to_data(self, sample):
        kp = sample["tag_keypoints"].astype(np.float32)  # (T, S, 2)
        T_ = kp.shape[0]
        obs = np.concatenate([
            kp.reshape(T_, -1),                          # (T, S*2)
            sample["agent_pos"].astype(np.float32),      # (T, 2)
        ], axis=-1)
        return {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }

    get_normalizer = _SpatialReplayBufferLoader._fit_lowdim_normalizer
