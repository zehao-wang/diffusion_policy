# Spatial PushT — Occupancy-only Diffusion Policy

Two variants of "occupancy-as-state" diffusion policy trained on the real-robot
PushT recordings under `data/spatial_episode_*/`. RGB frames are **not** used —
the only environment observation is a 2-D **tri-valued** occupancy grid on
the discretized workspace (default 128×128) encoding both the goal mask and the
T-block:

```
background = 0.0
goal cell  = 0.5
T-block    = 1.0   (drawn last; overwrites goal on overlap so the dynamic
                    T geometry is never destroyed by the static goal)
```

`agent_pos` (the pusher's current voxel coords) is fed as a separate "robot
state" input, and the policy outputs an action chunk of length `horizon`.

```
obs (per timestep):
    occupancy   -- T-block binary grid (shape depends on variant)
    agent_pos   -- pusher voxel coords        [2]
action chunk:
    target_coord per step                     [horizon, 2]
```

## Two variants

| Variant | `occupancy` key | shape       | encoder path                |
|---------|-----------------|-------------|-----------------------------|
| A (image) | `image`         | `[1,128,128]` | CNN sub-encoder (rgb path) |
| B (flat)  | `occupancy_flat`| `[16384]`     | low_dim concat → MLP        |

Both share the same zarr replay buffer; the dataset class decides shape on the fly.

## Layout

```
spatial_pusht/
├── data/
│   ├── occupancy_utils.py          # rasterize coords -> 128x128 binary
│   ├── episode_parser.py           # spatial_episode_v1.json -> aligned arrays
│   ├── replay_buffer_builder.py    # write zarr
│   └── occupancy_dataset.py        # two Dataset classes
├── config/
│   ├── task/
│   │   ├── spatial_pusht_image.yaml
│   │   └── spatial_pusht_flat.yaml
│   ├── train_spatial_pusht_image_workspace.yaml
│   └── train_spatial_pusht_flat_workspace.yaml
└── scripts/
    ├── build_replay_buffer.py      # JSON dir -> zarr (CLI)
    └── test_dataset.py             # one-batch sanity check
```

## Usage

All commands are run from the repo root with conda env `robodiff` active.

### 1. Build the replay buffer (one-off)

```bash
python -m exps.exp_pusht_real.spatial_pusht.scripts.build_replay_buffer \
    --json_dir data/spatial_episode_2026051 \
    --output   exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr
```

Flags:
- `--action-source next_agent` is the default. The spatial logs are
  post-action snapshots, so `action[t]` is reconstructed as the next pusher
  voxel position, matching official PushT's `state_t + command_t` replay
  semantics.
- `--sparse` use `tblock_coords` (~80 cells) instead of the denser
  `tblock_coords_full` (~470 cells).
- `--no_ffill` drop forward-filling on perception-failure frames.

### 2. Sanity-check the datasets

```bash
python -m exps.exp_pusht_real.spatial_pusht.scripts.test_dataset \
    --zarr exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr
```

Expected output (with `horizon=16, batch=4`):
- variant A: `obs.image (4,16,1,128,128)`, `obs.agent_pos (4,16,2)`, `action (4,16,2)`
- variant B: `obs.occupancy_flat (4,16,16384)`, `obs.agent_pos (4,16,2)`, `action (4,16,2)`

### 3. Train

```bash
# variant A: occupancy as 1-channel image
python train.py \
    --config-dir=exps/exp_pusht_real/spatial_pusht/config \
    --config-name=train_spatial_pusht_image_workspace

# variant B: flat occupancy
python train.py \
    --config-dir=exps/exp_pusht_real/spatial_pusht/config \
    --config-name=train_spatial_pusht_flat_workspace
```

Common CLI overrides:
- `horizon=8 n_obs_steps=2 n_action_steps=4` — your recordings are ~1.4 fps,
  so the default `horizon=16` covers ~11 seconds; tune to your task tempo.
- `task.dataset.zarr_path=...` — point to a different zarr.
- `dataloader.batch_size=32` — fits smaller GPUs.

### 4. Inference

Both variants produce a model that consumes
`obs={<occupancy_key>: ..., agent_pos: ...}` and emits an action chunk in voxel
coordinates `[x, y]`. Convert back to metric via `spatial_config.bbox_min/max`
on the deployment side.

## Coordinate convention (important)

`episode_viewer.py:245-246` shows the data uses `[x, y] = [row, col]`, i.e. `x`
is the **vertical** axis on the rendered grid. `occupancy_utils.rasterize_occupancy`
follows this convention (`grid[x, y] = 1`). Keep it consistent if you write any
downstream visualisation or env wrapper.

## What is NOT included

- RGB video — the `.mp4` files are unused; this is a purely state-space policy.
- A simulator env runner — real-robot only, so `env_runner` is a no-op.

## Notes on goal encoding

Goal is read per-frame from `spatial.goal_coords` (verified byte-identical
across all 38 recorded episodes — 88 cells forming a T outline). It is
rendered in the same single channel as the T-block, using value 0.5. T-block
(1.0) is drawn after the goal so overlap preserves T geometry. The shape stays
`[1, 128, 128]`; only the value semantics changed.
