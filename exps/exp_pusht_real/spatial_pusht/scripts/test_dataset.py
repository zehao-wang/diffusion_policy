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
    for k, v in batch["obs"].items():
        print(f"  obs.{k}: {tuple(v.shape)} dtype={v.dtype} "
              f"min={v.min().item():.3f} max={v.max().item():.3f}")


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
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        _summarize_batch(name, batch)

        # also verify the normalizer roundtrip on the keys we care about
        normalizer = ds.get_normalizer()
        for k, v in batch["obs"].items():
            if k in normalizer.params_dict:
                nv = normalizer[k].normalize(v)
                rv = normalizer[k].unnormalize(nv)
                err = (rv - v).abs().max().item()
                print(f"  normalizer[{k}] roundtrip max-err={err:.2e}")
        a = batch["action"]
        na = normalizer["action"].normalize(a)
        ra = normalizer["action"].unnormalize(na)
        print(f"  normalizer[action] roundtrip max-err={(ra-a).abs().max().item():.2e}")
        print()

    print("OK")


if __name__ == "__main__":
    main()
