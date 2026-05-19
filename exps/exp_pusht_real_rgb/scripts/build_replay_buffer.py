"""Build the pusht_image-compatible zarr from mp4+json episodes.

Usage (from repo root):
    python -m exps.exp_pusht_real_rgb.scripts.build_replay_buffer \
        --data_dir data/spatial_episode_2026051 \
        --output exps/exp_pusht_real_rgb/data/pusht_real_rgb.zarr
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exps.exp_pusht_real_rgb.data.replay_buffer_builder import build_replay_buffer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True, type=Path,
                   help="Directory containing spatial_episode_*.json and matching .mp4 files.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--image_size", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    build_replay_buffer(
        data_dir=args.data_dir,
        output_zarr=args.output,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
