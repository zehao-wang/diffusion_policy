# Spatial PushT — Occupancy-only Diffusion Policy

Four variants of "spatial-state" diffusion policy trained on the real-robot
PushT recordings under `data/spatial_episode_*/`. RGB frames are **not** used.
Variants A and B feed a 2-D **tri-valued** occupancy grid on the discretized
workspace (default 128×128) encoding both the goal mask and the T-block:

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

## Four variants

| Variant | obs source                  | obs shape (per step) | workspace                          |
|---------|------------------------------|----------------------|------------------------------------|
| A (image)         | rasterized occupancy `[1,128,128]` + `agent_pos` | image + 2 | `TrainDiffusionUnetHybridWorkspace` |
| B (flat)          | `occupancy.flatten()` + `agent_pos`              | `[16386]` | `TrainDiffusionUnetLowdimWorkspace` |
| C (tbar coords)   | padded T-bar voxel set + `agent_pos`             | `[K*2 + 2]` (default `K=101`)        | `TrainDiffusionUnetLowdimWorkspace` |
| D (tag keypoints) | fixed-slot AprilTag corner voxel xy + `agent_pos`| `[S*2 + 2]` (default `S=12 = 3*4`)   | `TrainDiffusionUnetLowdimWorkspace` |

Variant C feeds the policy a **small lowdim vector** built from the raw T-block
voxel coordinates instead of any rasterized occupancy. Variable per-frame coord
counts (~74–101 cells with the sparse `tblock_coords` key) are padded to a
dataset-wide-fixed `K` with sentinel `-1` so the obs vector has constant shape;
at inference time, frames with > K cells are uniformly sub-sampled to fit.

Variant D is the closest analogue to upstream pusht_lowdim's 9-keypoint
representation. The recording logs `tblock_apriltag_points_world` with
`(tag_id, corner_idx, coord_xy)` records: 3 object tags × 4 corners = **12
fixed slots** for this dataset, ordered ascending by `(tag_id, corner_idx)`.
Audited across all 8272 frames there are zero slot collisions in voxel
space, so no padding or sentinel is needed. At inference time the slots are
re-projected via `T_world_from_object × static_model.corner_points_by_tag`,
so even momentary tag occlusions still produce a complete 12-slot obs.

All four share the same zarr replay buffer; the dataset class decides shape on the fly.

## Layout

```
spatial_pusht/
├── data/
│   ├── occupancy_utils.py          # rasterize coords -> 128x128 binary
│   ├── episode_parser.py           # spatial_episode_v1.json -> aligned arrays
│   ├── replay_buffer_builder.py    # write zarr (auto-sizes T-bar pad K)
│   └── occupancy_dataset.py        # three Dataset classes
├── config/
│   ├── task/
│   │   ├── spatial_pusht_image.yaml
│   │   ├── spatial_pusht_flat.yaml
│   │   ├── spatial_pusht_tbar_coords.yaml
│   │   └── spatial_pusht_tag_keypoints.yaml
│   ├── train_spatial_pusht_image_workspace.yaml
│   ├── train_spatial_pusht_flat_workspace.yaml
│   ├── train_spatial_pusht_tbar_coords_workspace.yaml
│   └── train_spatial_pusht_tag_keypoints_workspace.yaml
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
  `tblock_coords_full` (~470 cells) for the **occupancy raster**.
- `--tbar-coord-key {tblock_coords|tblock_coords_full}` selects which JSON
  field feeds the **`tblock_coords` field** consumed by variant C
  (independent of `--sparse`). Default `tblock_coords`.
- `--tbar-pad-n K` override the auto-detected pad length. If unset, K is the
  observed max voxel count across the whole dump (101 on the default sparse
  source). Longer frames will be uniformly sub-sampled to fit.
- `--no_ffill` drop forward-filling on perception-failure frames.

### 2. Sanity-check the datasets

```bash
python -m exps.exp_pusht_real.spatial_pusht.scripts.test_dataset \
    --zarr exps/exp_pusht_real/spatial_pusht/data/spatial_pusht.zarr
```

Expected output (with `horizon=16, batch=4`, default `K=101`, default `S=12`):
- variant A: `obs.image (4,16,1,128,128)`, `obs.agent_pos (4,16,2)`, `action (4,16,2)`
- variant B: `obs (4,16,16386)`, `action (4,16,2)`
- variant C: `obs (4,16,204)`, `action (4,16,2)`   # 101*2 + 2
- variant D: `obs (4,16,26)`,  `action (4,16,2)`   #  12*2 + 2

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

# variant C: padded T-bar voxel coords (small lowdim)
python train.py \
    --config-dir=exps/exp_pusht_real/spatial_pusht/config \
    --config-name=train_spatial_pusht_tbar_coords_workspace

# variant D: fixed-slot AprilTag corner keypoints (very small lowdim)
python train.py \
    --config-dir=exps/exp_pusht_real/spatial_pusht/config \
    --config-name=train_spatial_pusht_tag_keypoints_workspace
```

Common CLI overrides:
- `horizon=8 n_obs_steps=2 n_action_steps=4` — your recordings are ~1.4 fps,
  so the default `horizon=16` covers ~11 seconds; tune to your task tempo.
- `task.dataset.zarr_path=...` — point to a different zarr.
- `dataloader.batch_size=32` — fits smaller GPUs.

### 4. Inference

All three variants emit an action chunk in voxel coordinates `[x, y]`. Convert
back to metric via `spatial_config.bbox_min/max` on the deployment side.

- Variant A: input is `{image: occupancy (1,128,128), agent_pos: (2,)}`.
- Variant B: input is the flat lowdim vector `concat(occupancy.flatten(), agent_pos)`.
- Variant C: input is `concat(tblock_coords_padded.flatten(), agent_pos)`. At
  runtime, sort the perceived T-block voxel coords by `(x, y)`, pad with `-1`
  to `K = task.tbar_pad_n`, and uniformly sub-sample if more than `K` cells
  are seen.
- Variant D: input is `concat(tag_keypoints.flatten(), agent_pos)`. The
  perception re-projects the canonical (T1-aligned) tag corners via
  `T_world_from_object` to current voxel xy on every frame, in the same
  `(tag_id, corner_idx)`-sorted slot order used at training time. No padding,
  no sentinel; the policy receives a fixed `(S, 2)` slot table per step.

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
