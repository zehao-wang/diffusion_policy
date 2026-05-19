#!/usr/bin/env bash
# Train the pusht_real_rgb hybrid (image+agent_pos) diffusion policy.
#
# Build zarr first (only once):
#   python -m exps.exp_pusht_real_rgb.scripts.build_replay_buffer \
#       --data_dir data/spatial_episode_2026051 \
#       --output  exps/exp_pusht_real_rgb/data/pusht_real_rgb.zarr
#
# Extra hydra overrides are forwarded:
#   ./train.sh horizon=8 dataloader.batch_size=32
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR=exps/exp_pusht_real_rgb/config
ZARR=exps/exp_pusht_real_rgb/data/pusht_real_rgb.zarr
LOG_DIR=exps/exp_pusht_real_rgb/data/runs
TS=$(date +%Y%m%d_%H%M%S)
GPU="${GPU:-0}"

[ -d "$ZARR" ] || { echo "missing zarr: $ZARR (run build_replay_buffer.py first)"; exit 1; }
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/rgb_${TS}.log"
conda run -n robodiff --no-capture-output python train.py \
    --config-dir="$CONFIG_DIR" \
    --config-name=train_pusht_real_rgb_workspace \
    training.device="cuda:${GPU}" \
    "$@" 2>&1 | tee "$LOG"
