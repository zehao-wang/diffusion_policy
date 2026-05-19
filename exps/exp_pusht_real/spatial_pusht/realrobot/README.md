# Real-robot deployment for `spatial_pusht_image`

End-to-end inference loop that turns a Blackfly RGB stream + the live
`pusht_service` arm into the obs / action chain expected by the trained
`SpatialPushTOccupancyImageDataset` policy.

```
Blackfly RGB ──► AprilTag detect ──► T-block mesh in AprilTag-world ─┐
                                                                     ├─► compute_spatial_language ─► obs
pusht_service /robot/state ──► link6 base → tip world  ──────────────┘
                                                                          │
                                                                          ▼
                                                       policy.predict_action  (voxel actions)
                                                                          │
                                          voxel → world → base → /robot/step ► real arm
```

## Layout

```
realrobot/
├── perception/                       # vendored from minye + glue
│   ├── apriltag_reconstruction.py    # 1415-line apriltag + PnP + Kalman
│   ├── spatial_language.py           # mesh → voxel grid
│   ├── world_frame.py                # 4-corner base ↔ world SE3 (load/save)
│   ├── camera_ipc.py                 # Unix socket + shm IPC
│   ├── pointgrey_client.py           # `PointGreyCamera` (reads from shm)
│   ├── pointgrey_capture_service.py  # standalone Blackfly server (sudo)
│   ├── pointgrey_calibration.py
│   └── state_extractor.py            # full detect → mesh → voxel pipeline
├── tools/                            # one-shot calibration helpers (NOT used at infer time)
│   ├── calibrate_world.py            # 4-corner world frame GUI (viser)
│   ├── arm_world_calibration.py      # joint stick-tip + base->world solver
│   ├── solve_tblock_mesh_pose.py     # CAD mesh → AprilTag corners ICP
│   └── align_apriltag_reference.py   # similarity-align two tag maps
├── arm_client.py                     # HTTP + world-frame math for pusht_service
├── arm_reader.py                     # background polling cache for arm_client
├── infer_loop.py                     # `InferLoopRunner` coordinator + headless CLI
├── infer_viser.py                    # thin CLI wiring → `gui/infer_app.InferViserApp`
├── gui/
│   └── infer_app.py                  # viser GUI; single main loop, 3D scene
├── configs/realrobot.yaml            # paths + bbox + service URLs
└── README.md (this file)

# Artifacts (shipped in robodata_Agilex@minye db676a8):
data/realrobot/
├── model/
│   ├── reference_aligned_to_model_filtered.json   # tag corners (apriltag-world)
│   ├── tblock.ply                                  # T-block CAD mesh
│   ├── tblock_aligned_to_tags.ply                  # debug viz
│   └── tblock_mesh_alignment.json                  # mesh→tags ICP result
├── pointgrey_calibration.json                      # Blackfly intrinsics (2448x2048)
└── records/
    ├── arm_world_calibration.json                  # joint solve: T_world_from_base + tip offset
    └── arm_tags_world_transforms.json              # per-sample debug dump
```

`pusht_service` is **not modified**. We talk to it over the existing HTTP
API (`/robot/state`, `/robot/step`, `/arm/connect`, `/arm/disconnect`).

## One-time dependencies

```bash
# robodiff env (where the model + the loop run)
conda activate robodiff
pip install trimesh pupil-apriltags
# pusht_service env (where PySpin already lives) -- no extra installs needed
```

## One-time calibration artifacts

All shipped in `data/realrobot/` (`robodata_Agilex@minye` commit `db676a8`,
copied here by the integration step). The yaml under `paths:` points at
this directory already.

| file | role |
|---|---|
| `data/realrobot/model/reference_aligned_to_model_filtered.json` | AprilTag corners in AprilTag-world frame (origin = tag 100) |
| `data/realrobot/model/tblock.ply` | T-block CAD mesh (16 verts, 28 faces) |
| `data/realrobot/pointgrey_calibration.json` | Blackfly intrinsics + distortion; native 2448x2048 |
| `data/realrobot/records/arm_world_calibration.json` | `T_world_from_base` + `tip_position_in_eef_m` (joint AprilTag solve); the **world frame here IS the AprilTag world**, so it works directly without another alignment |

The bbox / resolution under `realrobot.yaml` must match the values in your
training episode JSONs (`spatial_config.bbox_min/max/resolution_xyz`).
Defaults (`[-0.05,-0.1,0]` ~ `[0.45,0.45,0.1]`, `[128,128,12]`, `z_voxel=3`)
match `data/spatial_episode_2026051/*.json`.

> ⚠️ Camera resolution is locked to **2448 × 2048** by the calibration file —
> `pointgrey_calibration.merge_pointgrey_camera_info` rejects a mismatch.
> If you want to capture at a lower resolution, re-run minye's calibration
> at that resolution OR scale the intrinsics yourself.

> ⚠️ The world frame in `arm_world_calibration.json` is **minye's machine**.
> If you collected your training episodes on a different robot base, you
> need to re-run `tools/arm_world_calibration.py` on your arm. The
> AprilTag-world part is camera-independent so the rest of the artifacts
> (`tblock.ply`, `reference_aligned_to_model_filtered.json`,
> `pointgrey_calibration.json`) carry over unchanged.

## Run order

### 1. Start the Blackfly capture service (Terminal A, **sudo**)

PySpin needs root for USB. The service binds to a Unix socket and pushes
frames into a shared-memory ring; the inference loop reads from that ring.

```bash
sudo /home/zwa0839/miniconda3/envs/pusht_service/bin/python \
    -m exps.exp_pusht_real.spatial_pusht.realrobot.perception.pointgrey_capture_service \
    --width 640 --height 480 --fps 30 \
    --socket-path /tmp/pointgrey_capture.sock \
    --shm-prefix pointgrey_capture
```

Run from the repo root (so `-m` resolves the package). If you have multiple
Blackfly cameras, pass `--serial`.

### 2. Start `pusht_service` (Terminal B)

```bash
cd /home/zwa0839/Documents/Projects/pusht_service
conda activate pusht_service
python main.py --serve --api-port 8012
```

Leave it running. CAN connect/disconnect is driven by the inference loop.

### 3. Start the diffusion-policy service (Terminal C, `robodiff` env)

The trained checkpoint now lives behind its own HTTP service so the
coordinator process doesn't drag in `torch` / `hydra`. See
`policy_service/README.md` for the full API surface.

```bash
cd /home/zwa0839/Documents/Projects/robodata_Agilex/packages/diffusion_policy
conda activate robodiff

python -m exps.exp_pusht_real.spatial_pusht.policy_service.main \
    --ckpt data/outputs/2026.05.16/16.43.56_train_spatial_pusht_occupancy_image_spatial_pusht_image/checkpoints/latest.ckpt \
    --device cuda:1 \
    --api-port 8014
```

Pinning `--device cuda:1` keeps the policy off the GPU used for training.

### 4. Run the coordinator (Terminal D, `robodiff` env)

Two front-ends, same coordinator backend:

```bash
cd /home/zwa0839/Documents/Projects/robodata_Agilex/packages/diffusion_policy
conda activate robodiff

# Viser GUI — recommended for live operation
python -m exps.exp_pusht_real.spatial_pusht.realrobot.infer_viser --port 8013

# Headless --dry-run sanity loop (no arm motion, prints obs/action per tick)
python -m exps.exp_pusht_real.spatial_pusht.realrobot.infer_loop --dry-run
```

GUI flags:

| Flag | Effect |
|---|---|
| `--policy-url URL` | Override `cfg.policy_service.url` (default `http://localhost:8014`) |
| `--pusht-url URL`  | Override `cfg.pusht_service.url` (default `http://localhost:8012`) |
| `--no-arm`         | Skip the arm subsystem; perception + policy still run |
| `--no-camera`      | Skip the camera; useful for testing the GUI / service plumbing |
| `--no-wait-policy` | Don't block at startup waiting for `/health` |

## Behaviour summary

* `n_obs_steps=2`: warmup logs `(1/2)` once, then policy fires every tick.
* `n_action_steps=8`, `n_action_chunk=4`: the loop sends only the first 4
  voxel actions before re-observing, even though the policy emits 8.
* No outer reproj gate on the inference loop. The AprilTag Kalman smoother
  (`apriltag.enable_kalman: true`) drops bad measurements internally
  (`max_reproj_error_px_for_update=6.0`) and uses its prediction instead,
  matching minye's design. The State panel still displays raw reproj px
  for diagnostics.
* Ctrl-C cleanly stops the camera client, calls `/arm/disconnect` (which
  latches a hold on the arm), and exits.

## Sanity checks before pressing go

1. Inspect `obs.image` in `--dry-run`: it should show a T-shaped white
   region against black. If it's empty or scattered, the camera isn't
   seeing your T-block tags or the `world_config.json` is for the wrong
   arm base.
2. Inspect `obs.agent_pos`: should track the pusher tip when you move the
   arm. If it's stuck at `[0,0]` or near a corner, your `tip_in_eef_m`
   is wrong (or the world frame x/y axes are flipped vs. training time).
3. Compare a few episode-JSON frames against the live `obs.image` /
   `obs.agent_pos` to confirm the voxel coordinate system matches.

## Known footguns

* Three processes share work through HTTP / shm:
  - `pointgrey_capture_service` (sudo, `pusht_service` env — needs PySpin)
  - `pusht_service main.py --serve` (any env with `piper_sdk`)
  - `policy_service.main` (robodiff env — owns the checkpoint + GPU)
  - the coordinator (`infer_viser` or `infer_loop`) talks to all three and
    runs only perception (apriltag + voxelization) locally; it doesn't
    import `torch` or `hydra` anymore.
* The model was trained with `agent_pos`/`action` *un-normalized to voxel
  index* (`LinearNormalizer` rescales each dim by its empirical min/max).
  So the in/out voxel coords here can be **fractional** at inference time
  — we floor when rasterising the T-block but **send fractional voxels
  through** `voxel_xy_to_world` so action precision isn't lost.
* `_load_tip_offset` only reads `tip_position_in_eef_m`; if that file
  doesn't exist the tip is assumed to be the link6 origin, which is
  wrong by ~10cm on a typical setup. Run `arm_world_calibration` (or
  copy minye's JSON) before trusting the live `pusher_world`.
