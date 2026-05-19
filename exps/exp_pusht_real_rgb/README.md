# exp_pusht_real_rgb

RGB-image diffusion policy on the real-robot pusht recordings (灰度工业相机
+ 3-channel repeat). Reuses the official `pusht_image` / hybrid UNet1D
workspace end-to-end — only the dataset is regenerated from
`(spatial_episode_*.json + .mp4)` pairs.

```
exps/exp_pusht_real_rgb/
├── config/
│   ├── task/pusht_real_rgb.yaml             # shape_meta + PushTImageDataset → new zarr
│   └── train_pusht_real_rgb_workspace.yaml  # hybrid workspace clone
├── data/
│   ├── replay_buffer_builder.py             # writes pusht-compatible zarr
│   ├── video_decoder.py                     # mp4 → (T, 96, 96, 3) uint8
│   └── pusht_real_rgb.zarr                  # produced by build script
└── scripts/
    ├── build_replay_buffer.py
    └── train.sh
```

## Data contract

| Field | Shape | Dtype | Source |
| --- | --- | --- | --- |
| `data/img`    | `(N, 96, 96, 3)` | `uint8`   | mp4 → grayscale → center-square crop → resize 96 → repeat to 3ch |
| `data/state`  | `(N, 2)`         | `float32` | `agent_pos` (pusher voxel xy, voxel domain [0, 128)) |
| `data/action` | `(N, 2)`         | `float32` | `target_coord` (target voxel xy) |
| `meta/episode_ends` | — | int | per-episode boundaries |

Layout matches the official `pusht_cchi_v7_replay.zarr`, so
`diffusion_policy.dataset.pusht_image_dataset.PushTImageDataset` loads
it without modification.

Per-frame alignment between `movements[*].frames[*]` and mp4 frames was
verified (all 38 episodes match `frame_count` exactly). Builder uses
`exps.exp_pusht_real.spatial_pusht.data.episode_parser.parse_episode`
under the hood to pull `agent_pos` / `action`.

## Build the zarr (one-shot)

```bash
python -m exps.exp_pusht_real_rgb.scripts.build_replay_buffer \
    --data_dir data/spatial_episode_2026051 \
    --output  exps/exp_pusht_real_rgb/data/pusht_real_rgb.zarr
# 38 episodes, 8272 frames, ~58 MB on disk
```

`--image_size 96` matches the official pusht resolution. Increasing it
also requires bumping `crop_shape` in the train config.

## Train

Foreground:
```bash
python train.py \
    --config-dir=exps/exp_pusht_real_rgb/config \
    --config-name=train_pusht_real_rgb_workspace \
    training.device=cuda:0
```

Or via the helper script (writes a tee'd log under `data/runs/`):
```bash
GPU=0 ./exps/exp_pusht_real_rgb/scripts/train.sh
```

Workspace + policy config are 1-to-1 with official PushT hybrid:
`horizon=16`, `n_obs_steps=2`, `n_action_steps=8`,
`down_dims=[512,1024,2048]`, `crop_shape=[76,76]`, ResNet18 + GroupNorm,
DDPM 100 steps, `obs_as_global_cond=True`. Only `task.dataset.zarr_path`
points at the new zarr; `env_runner` uses the no-op
`RealPushTImageRunner` since this is offline training.

### Normalize / unnormalize

- `image`: `get_image_range_normalizer()` maps `uint8/255 ∈ [0, 1]` → `[-1, 1]`.
- `agent_pos` / `action`: `LinearNormalizer(mode='limits', last_n_dims=1)`
  per-dim min-max → `[-1, 1]`.
- DDPM `clip_sample=True` keeps the sampled trajectory in `[-1, 1]`, so
  after unnormalize the action is guaranteed to land in the training
  data's `[min, max]` range. **No grid clip needed at inference.**

## Serve a checkpoint

The `exp_pusht_real` policy service auto-detects the policy kind from
the ckpt's embedded `cfg.task.shape_meta` (`policy_runner._build_info`,
policy_runner.py:92-109), so the RGB ckpt is loaded by the same entry
point without changes:

```bash
python -m exps.exp_pusht_real.spatial_pusht.policy_service.main \
    --ckpt data/outputs/<YYYY.MM.DD>/<HH.MM.SS>_train_pusht_real_rgb_pusht_real_rgb/checkpoints/latest.ckpt \
    --device cuda:0 \
    --api-port 8014
```

`PolicyRunner.predict` returns `int64` action voxels: continuous
unnormalized output is snapped with `np.rint().astype(np.int64)` (no
clip — `clip_sample=True` + LinearNormalizer already bound the action
to the training range). Matches the official PushT inference path with
the single addition of integer-voxel rounding.

### Wire contract (`/predict`)

| Input | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `image`     | `(n_obs_steps, 3, 96, 96)` | `float32` in `[0, 1]` | Already divided by 255. The 3 channels are repeated grayscale. |
| `agent_pos` | `(n_obs_steps, 2)`         | `float32`             | Pusher voxel xy. |

| Output | Shape | Dtype |
| --- | --- | --- |
| `action`    | `(n_action_steps, 2)` | `int64` (over JSON: ints) |

### Real-robot client preprocessing

The default `infer_viser` collects 128×128 binary occupancy for the
state-space exp; switching to the RGB service requires a different
obs pipeline. Mirror `data/video_decoder.py`:

1. Grab a frame from the industrial camera (grayscale, native resolution).
2. Center-square crop to `min(H, W)`.
3. `cv2.resize` to `96 × 96` with `INTER_AREA`.
4. Repeat the single channel to 3 along the channel axis.
5. Convert to `float32`, divide by 255.
6. Stack the most recent `n_obs_steps=2` frames → `(2, 3, 96, 96)`.
7. Pair with `(2, 2)` agent_pos in voxel coords.

Then POST to the service exactly like the state-space client does.
