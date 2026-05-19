"""Sanity-check that both dataset variants emit correctly-shaped batches.

Usage (from repo root):
    python -m exps.exp_pusht_real.spatial_pusht.scripts.test_dataset \
        --zarr exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from exps.exp_pusht_real.spatial_pusht.data import (
    SpatialPushTOccupancyFlatDataset,
    SpatialPushTOccupancyImageDataset,
)


def _summarize_batch(name, batch):
    print(f"[{name}]")
    print(f"  action: {tuple(batch['action'].shape)} dtype={batch['action'].dtype}")
    obs = batch["obs"]
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"  obs.{k}: {tuple(v.shape)} dtype={v.dtype} "
                  f"min={v.min().item():.3f} max={v.max().item():.3f}")
    else:
        print(f"  obs: {tuple(obs.shape)} dtype={obs.dtype} "
              f"min={obs.min().item():.3f} max={obs.max().item():.3f}")


def _summarize_alignment(ds):
    agent = ds.replay_buffer["agent_pos"][:]
    action = ds.replay_buffer["action"][:]
    agent_action_dist = torch.linalg.norm(
        torch.from_numpy(action - agent), dim=-1
    ).float()
    action_delta = torch.linalg.norm(
        torch.from_numpy(action[1:] - action[:-1]), dim=-1
    ).float()
    print(
        f"  |action-agent| mean={agent_action_dist.mean().item():.3f} "
        f"p50={agent_action_dist.median().item():.3f} "
        f"p95={agent_action_dist.quantile(0.95).item():.3f}"
    )
    print(
        f"  |delta action| mean={action_delta.mean().item():.3f} "
        f"p50={action_delta.median().item():.3f} "
        f"p95={action_delta.quantile(0.95).item():.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    for name, cls in [
        ("variant A: occupancy image", SpatialPushTOccupancyImageDataset),
        ("variant B: occupancy flat", SpatialPushTOccupancyFlatDataset),
    ]:
        ds = cls(zarr_path=str(args.zarr), horizon=args.horizon,
                 pad_before=1, pad_after=7, val_ratio=0.05)
        print(f"{name}: len={len(ds)} (train), val_len={len(ds.get_validation_dataset())}")
        _summarize_alignment(ds)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        _summarize_batch(name, batch)

        # also verify normalizer roundtrip
        normalizer = ds.get_normalizer()
        obs = batch["obs"]
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k in normalizer.params_dict:
                    err = (normalizer[k].unnormalize(normalizer[k].normalize(v))
                           - v).abs().max().item()
                    print(f"  normalizer[{k}] roundtrip max-err={err:.2e}")
        else:
            err = (normalizer["obs"].unnormalize(normalizer["obs"].normalize(obs))
                   - obs).abs().max().item()
            print(f"  normalizer[obs] roundtrip max-err={err:.2e}")
        a = batch["action"]
        err = (normalizer["action"].unnormalize(normalizer["action"].normalize(a))
               - a).abs().max().item()
        print(f"  normalizer[action] roundtrip max-err={err:.2e}")
        print()

    print("OK")


if __name__ == "__main__":
    main()
