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
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import torch

from exps.exp_pusht_real.spatial_pusht.data import (
    SpatialPushTOccupancyFlatDataset,
    SpatialPushTOccupancyImageDataset,
)
from exps.exp_pusht_real.spatial_pusht.data.occupancy_utils import (
    GOAL_VALUE, TBLOCK_VALUE,
)


# Coord convention (from episode_viewer.py:245-246):
#   data stores [x, y] = [row, col]; image axis 0 (vertical) is x, axis 1 is y.
def _agent_xy(agent_pos):
    return float(agent_pos[1]), float(agent_pos[0])  # plt expects (col, row)


# Tri-valued colormap: background = white, goal = light green, T-block = dark red.
# Boundaries chosen to bucket exact stored values (0.0, GOAL_VALUE, TBLOCK_VALUE).
_OCC_CMAP = ListedColormap(["#ffffff", "#9bd49b", "#b21f1f"])
_OCC_NORM = BoundaryNorm(
    [-0.5, 0.25, (GOAL_VALUE + TBLOCK_VALUE) / 2.0, 1.5],
    _OCC_CMAP.N,
)


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
        ax.imshow(occ_thw[t], cmap=_OCC_CMAP, norm=_OCC_NORM, origin="upper",
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
        f"sample #{sample_idx}  "
        f"green=goal({GOAL_VALUE})  dark-red=T-block({TBLOCK_VALUE})  "
        f"blue=pusher  red*=action_t  red·line=action chunk",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _dump_text(sample_idx, agent_pos_t, action_t, occ_thw, out_path):
    # Break down cell counts per (background / goal / T-block) so we can sanity
    # check that BOTH layers survived the dataset pipeline.
    flat = occ_thw.reshape(occ_thw.shape[0], -1)
    goal_cnt = np.isclose(flat, GOAL_VALUE).sum(axis=1).astype(int)
    tblk_cnt = np.isclose(flat, TBLOCK_VALUE).sum(axis=1).astype(int)
    uniq_per_t = [np.unique(flat[t]).round(3).tolist() for t in range(flat.shape[0])]
    with out_path.open("w") as f:
        f.write(f"sample #{sample_idx}\n")
        f.write(f"horizon T = {occ_thw.shape[0]}, grid = {occ_thw.shape[1]}x{occ_thw.shape[2]}\n")
        f.write(f"goal value = {GOAL_VALUE}  t-block value = {TBLOCK_VALUE}\n")
        f.write("\n t | agent_pos [x,y] | action_target [x,y] | n_goal | n_tblk | unique vals\n")
        f.write("-" * 90 + "\n")
        for t in range(occ_thw.shape[0]):
            ax_, ay_ = agent_pos_t[t]
            tx_, ty_ = action_t[t]
            f.write(f"{t:2d} | [{ax_:6.2f},{ay_:6.2f}] | [{tx_:6.2f},{ty_:6.2f}] "
                    f"| {goal_cnt[t]:5d} | {tblk_cnt[t]:5d} | {uniq_per_t[t]}\n")


def _occ_from_batch(name, sample):
    """Return (T, H, W) float occupancy and (T, 2) agent_pos for a sample.

    Variant A returns obs as a dict {image, agent_pos}; variant B returns obs
    as a single concatenated tensor [occupancy.flatten() || agent_pos].
    """
    obs = sample["obs"]
    if isinstance(obs, dict):
        occ = obs["image"].numpy()              # (T, 1, H, W)
        return occ[:, 0], obs["agent_pos"].numpy()
    flat = obs.numpy()                          # (T, H*W + 2)
    agent = flat[:, -2:]
    side = int(round((flat.shape[-1] - 2) ** 0.5))
    occ = flat[:, :-2].reshape(flat.shape[0], side, side)
    return occ, agent


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
            occ, agent = _occ_from_batch(tag, sample)
            action = sample["action"].numpy()
            _render_sample(occ, agent, action, args.n_obs_steps, int(idx),
                           variant_dir / f"sample_{i:02d}_idx{int(idx)}.png")
            _dump_text(int(idx), agent, action, occ,
                       variant_dir / f"sample_{i:02d}_idx{int(idx)}.txt")
        print(f"    -> {variant_dir}")

    print("done.")


if __name__ == "__main__":
    main()
