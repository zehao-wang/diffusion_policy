#!/usr/bin/env bash
# Launch both occupancy-only diffusion-policy variants in parallel.
#   variant A (image, 1x128x128)  -> cuda:0
#   variant B (flat,  16384)      -> cuda:1
#
# Extra hydra overrides are forwarded to both runs, e.g.
#   ./train_both.sh horizon=8 dataloader.batch_size=32
#
# Run a single variant in the foreground (live logs in terminal):
#   conda run -n robodiff --no-capture-output python train.py \
#       --config-dir=exps/exp_pusht_real/spatial_pusht/config \
#       --config-name=train_spatial_pusht_image_workspace training.device=cuda:0
#   conda run -n robodiff --no-capture-output python train.py \
#       --config-dir=exps/exp_pusht_real/spatial_pusht/config \
#       --config-name=train_spatial_pusht_flat_workspace  training.device=cuda:1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR=exps/exp_pusht_real/spatial_pusht/config
ZARR=exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr
LOG_DIR=exps/exp_pusht_real/spatial_pusht/data/runs
TS=$(date +%Y%m%d_%H%M%S)
GROUP="spatial_pusht_${TS}"

[ -d "$ZARR" ] || { echo "missing zarr: $ZARR (run build_replay_buffer.py first)"; exit 1; }
mkdir -p "$LOG_DIR"

launch() {
    local tag=$1 cfg=$2 gpu=$3
    local log="$LOG_DIR/${tag}_${TS}.log"
    conda run -n robodiff --no-capture-output python train.py \
        --config-dir="$CONFIG_DIR" --config-name="$cfg" \
        training.device="cuda:${gpu}" \
        logging.group="$GROUP" \
        "$@" >"$log" 2>&1 &
    echo "$! $log"
}

shift_args=("${@:1}")

read IMAGE_PID IMAGE_LOG < <(launch image train_spatial_pusht_image_workspace 0 "${shift_args[@]}")
read FLAT_PID  FLAT_LOG  < <(launch flat  train_spatial_pusht_flat_workspace  1 "${shift_args[@]}")

printf "image -> cuda:0  pid=%s  log=%s\n" "$IMAGE_PID" "$IMAGE_LOG"
printf "flat  -> cuda:1  pid=%s  log=%s\n" "$FLAT_PID"  "$FLAT_LOG"
printf "wandb group: %s\n" "$GROUP"
printf "tail logs:    tail -F %s %s\n" "$IMAGE_LOG" "$FLAT_LOG"

trap 'kill $IMAGE_PID $FLAT_PID 2>/dev/null || true' INT TERM
wait $IMAGE_PID $FLAT_PID
