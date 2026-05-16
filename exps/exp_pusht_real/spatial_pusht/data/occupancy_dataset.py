"""Two diffusion-policy dataset variants over the same occupancy zarr.

Both variants follow the multi-key ImageDataset contract:
    obs = { <occupancy_key>: (T, ...), agent_pos: (T, 2) }
    action = (T, action_dim)

Variant A (image):
    occupancy_key = 'image', shape (T, 1, 128, 128).
    Goes through a CNN sub-encoder via MultiImageObsEncoder.

Variant B (flat):
    occupancy_key = 'occupancy_flat', shape (T, 128*128).
    Treated as low_dim, concatenated with agent_pos in the encoder.

agent_pos is ALWAYS a separate input (robot state). action horizon length = horizon.
"""
import copy
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class _SpatialPushTOccupancyBase(BaseImageDataset):
    """Shared zarr loading + sampling. Subclasses define how to shape `occupancy`."""

    obs_occupancy_key: str = "image"  # overridden by subclasses

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int = None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["occupancy", "agent_pos", "action"]
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def _shape_occupancy(self, occ_thw: np.ndarray) -> np.ndarray:
        """Subclass hook: occ_thw is (T, H, W) float32 binary."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        occ = sample["occupancy"].astype(np.float32)
        occ_shaped = self._shape_occupancy(occ)

        data = {
            "obs": {
                self.obs_occupancy_key: occ_shaped,
                "agent_pos": sample["agent_pos"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)


class SpatialPushTOccupancyImageDataset(_SpatialPushTOccupancyBase):
    """Variant A: occupancy as (1, 128, 128) single-channel image -> CNN encoder."""

    obs_occupancy_key = "image"

    def _shape_occupancy(self, occ_thw: np.ndarray) -> np.ndarray:
        # (T, H, W) -> (T, 1, H, W)
        return occ_thw[:, None, :, :]

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "action": self.replay_buffer["action"][:],
            "agent_pos": self.replay_buffer["agent_pos"][:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # occupancy is already in [0, 1]; identity normalizer in that range
        normalizer["image"] = get_image_range_normalizer()
        return normalizer


class SpatialPushTOccupancyFlatDataset(_SpatialPushTOccupancyBase):
    """Variant B: occupancy as flat (H*W,) low_dim vector -> concatenated MLP."""

    obs_occupancy_key = "occupancy_flat"

    def _shape_occupancy(self, occ_thw: np.ndarray) -> np.ndarray:
        # (T, H, W) -> (T, H*W)
        t = occ_thw.shape[0]
        return occ_thw.reshape(t, -1)

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "action": self.replay_buffer["action"][:],
            "agent_pos": self.replay_buffer["agent_pos"][:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # Occupancy is already binary in [0, 1]; use identity (avoids divide-by-zero
        # on all-zero columns which would happen with mode='limits').
        normalizer["occupancy_flat"] = get_image_range_normalizer()
        return normalizer
