#!/usr/bin/env bash
# Optional experiment: shrink the flat-obs cond input by avg-pooling the 128x128
# occupancy grid before flatten. Pure dataset-side change; no model edits.
#
# Usage:
#   ./train_flat_pooled.sh                       # default POOL=4 -> obs_dim 1026
#   POOL=8 ./train_flat_pooled.sh                # POOL=8 -> obs_dim 258
#   POOL=4 ./train_flat_pooled.sh training.num_epochs=300 optimizer.weight_decay=1e-3
#
# POOL must divide 128 (1, 2, 4, 8, 16, 32, ...).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR=exps/exp_pusht_real/spatial_pusht/config
ZARR=exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr
LOG_DIR=exps/exp_pusht_real/spatial_pusht/data/runs

POOL=${POOL:-4}
GPU=${GPU:-1}
TS=$(date +%Y%m%d_%H%M%S)
GROUP="spatial_pusht_flat_pool${POOL}_${TS}"

[ -d "$ZARR" ] || { echo "missing zarr: $ZARR (run build_replay_buffer.py first)"; exit 1; }
[ $((128 % POOL)) -eq 0 ] || { echo "POOL=$POOL must divide 128"; exit 1; }
mkdir -p "$LOG_DIR"

GRID=$((128 / POOL))
OBS_DIM=$((GRID * GRID + 2))
KP_DIM=$((GRID * GRID))

LOG="$LOG_DIR/flat_pool${POOL}_${TS}.log"
echo "POOL=$POOL  grid=${GRID}x${GRID}  obs_dim=$OBS_DIM  gpu=cuda:${GPU}"
echo "log: $LOG"
echo "wandb group: $GROUP"

conda run -n robodiff --no-capture-output python train.py \
    --config-dir="$CONFIG_DIR" --config-name=train_spatial_pusht_flat_workspace \
    training.device="cuda:${GPU}" \
    logging.group="$GROUP" \
    task.dataset.occupancy_pool="$POOL" \
    task.obs_dim="$OBS_DIM" \
    task.keypoint_dim="$KP_DIM" \
    "$@" 2>&1 | tee "$LOG"
