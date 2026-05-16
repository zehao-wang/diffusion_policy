"""Render a few (obs, action) samples from each dataset variant for sanity-check.

Output goes under the repo-wide outputs root used by the workspace
(`data/outputs/<date>/<time>_inspect_io/`) so it sits next to real training dumps.

Usage (from repo root):
    python -m exps.exp_pusht_real.spatial_pusht.scripts.visualize_io \
        --zarr exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr \
        --n_samples 6
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from exps.exp_pusht_real.spatial_pusht.data import (
    SpatialPushTOccupancyFlatDataset,
    SpatialPushTOccupancyImageDataset,
)


# Coord convention (from episode_viewer.py:245-246):
#   data stores [x, y] = [row, col]; image axis 0 (vertical) is x, axis 1 is y.
def _agent_xy(agent_pos):
    return float(agent_pos[1]), float(agent_pos[0])  # plt expects (col, row)


def _render_sample(occ_thw, agent_pos_t, action_t, n_obs_steps, sample_idx, out_path):
    T, H, W = occ_thw.shape
    cols = min(T, 8)
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows + 0.6),
                             squeeze=False)
    # full action trajectory in (col, row) order for overlay
    traj_cols = action_t[:, 1]
    traj_rows = action_t[:, 0]

    for t in range(T):
        ax = axes[t // cols][t % cols]
        ax.imshow(occ_thw[t], cmap="Greys", origin="upper", vmin=0, vmax=1,
                  interpolation="nearest", zorder=0)
        # trajectory layer (drawn first; partially transparent so markers pop)
        ax.plot(traj_cols, traj_rows, color="red", linewidth=1.1, alpha=0.45,
                zorder=2)
        ax.scatter(traj_cols, traj_rows, s=14, c="red", alpha=0.45, zorder=3)
        # marker layer (drawn last so it sits on top of trajectory & occupancy)
        ax.scatter(traj_cols[t], traj_rows[t], s=110, marker="*",
                   edgecolors="black", facecolors="red", linewidths=0.8,
                   zorder=4)
        ax.scatter(*_agent_xy(agent_pos_t[t]), s=70, marker="o",
                   edgecolors="black", facecolors="tab:blue", linewidths=0.8,
                   zorder=5)
        is_obs = t < n_obs_steps
        ax.set_title(f"t={t} {'[obs]' if is_obs else '[act]'}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    # blank any unused panels
    for t in range(T, rows * cols):
        axes[t // cols][t % cols].axis("off")

    fig.suptitle(
        f"sample #{sample_idx}  blue=pusher  red*=action_t  red·line=action chunk",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _dump_text(sample_idx, agent_pos_t, action_t, occ_thw, out_path):
    occ_sums = occ_thw.reshape(occ_thw.shape[0], -1).sum(axis=1).astype(int)
    with out_path.open("w") as f:
        f.write(f"sample #{sample_idx}\n")
        f.write(f"horizon T = {occ_thw.shape[0]}, grid = {occ_thw.shape[1]}x{occ_thw.shape[2]}\n")
        f.write("\nt | agent_pos [x,y] | action_target [x,y] | occupancy non-zero cells\n")
        f.write("-" * 70 + "\n")
        for t in range(occ_thw.shape[0]):
            ax_, ay_ = agent_pos_t[t]
            tx_, ty_ = action_t[t]
            f.write(f"{t:2d} | [{ax_:6.2f},{ay_:6.2f}] | [{tx_:6.2f},{ty_:6.2f}] | {occ_sums[t]:4d}\n")


def _occ_from_batch(name, sample):
    """Return (T, H, W) float occupancy from whichever variant's obs dict."""
    if "image" in sample["obs"]:
        occ = sample["obs"]["image"].numpy()  # (T, 1, H, W)
        return occ[:, 0]
    flat = sample["obs"]["occupancy_flat"].numpy()  # (T, H*W)
    side = int(round(flat.shape[-1] ** 0.5))
    return flat.reshape(flat.shape[0], side, side)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_obs_steps", type=int, default=2)
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outputs_root", type=Path,
                   default=REPO_ROOT / "data" / "outputs")
    return p.parse_args()


def main():
    args = parse_args()
    now = dt.datetime.now()
    out_dir = args.outputs_root / now.strftime("%Y.%m.%d") / (
        now.strftime("%H.%M.%S") + "_inspect_io")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}")

    rng = np.random.default_rng(args.seed)

    for tag, cls in [
        ("image", SpatialPushTOccupancyImageDataset),
        ("flat", SpatialPushTOccupancyFlatDataset),
    ]:
        ds = cls(
            zarr_path=str(args.zarr),
            horizon=args.horizon,
            pad_before=args.n_obs_steps - 1,
            pad_after=args.horizon - args.n_obs_steps,
            val_ratio=0.0,
        )
        idxs = rng.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)
        variant_dir = out_dir / tag
        variant_dir.mkdir(exist_ok=True)
        print(f"  variant {tag}: dataset len={len(ds)}, picking {len(idxs)} samples")
        for i, idx in enumerate(idxs):
            sample = ds[int(idx)]
            occ = _occ_from_batch(tag, sample)
            agent = sample["obs"]["agent_pos"].numpy()
            action = sample["action"].numpy()
            _render_sample(occ, agent, action, args.n_obs_steps, int(idx),
                           variant_dir / f"sample_{i:02d}_idx{int(idx)}.png")
            _dump_text(int(idx), agent, action, occ,
                       variant_dir / f"sample_{i:02d}_idx{int(idx)}.txt")
        print(f"    -> {variant_dir}")

    print("done.")


if __name__ == "__main__":
    main()
