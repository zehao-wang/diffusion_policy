"""Build the zarr replay buffer from a directory of spatial_episode_*.json files.

Usage (from repo root):
    python -m exps.exp_pusht_real.spatial_pusht.scripts.build_replay_buffer \
        --json_dir data/spatial_episode_2026051 \
        --output exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exps.exp_pusht_real.spatial_pusht.data.replay_buffer_builder import build_replay_buffer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json_dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--grid", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    p.add_argument("--sparse", action="store_true",
                   help="Use tblock_coords (sparse) instead of tblock_coords_full (dense).")
    p.add_argument("--no_ffill", action="store_true",
                   help="Do not forward-fill occupancy on frames with available=False.")
    return p.parse_args()


def main():
    args = parse_args()
    build_replay_buffer(
        json_dir=args.json_dir,
        output_zarr=args.output,
        grid_hw=tuple(args.grid),
        use_full_occupancy=not args.sparse,
        forward_fill_unavailable=not args.no_ffill,
    )


if __name__ == "__main__":
    main()
