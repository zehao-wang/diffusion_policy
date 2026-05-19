# Real-robot inference — start-up checklist

Operational sequence for running the `spatial_pusht_image` diffusion policy
against the live PIPER + Blackfly setup. **Four independent processes**, each
in its own terminal / env, talking over HTTP + shared-memory:

```
                              Blackfly capture service  (§1)
                                       │ shm + Unix socket
                                       ▼
   pusht_service ── HTTP :8012 ──►  coordinator  ◄── HTTP :8014 ── policy_service
        (§2)                          (§4)                              (§3)
        ▲                             │
        └────────── HTTP :8012 ───────┘   (writes /robot/step)
```

Bring them up in order: §1 (camera) → §2 (arm) → §3 (policy) → §4 (coordinator
GUI). §2/§3 can come up in either order, but the coordinator blocks on
`policy_service /health` at startup so §3 should not lag too far behind.

Repo root abbreviations:
```
DP_ROOT  = /home/zwa0839/Documents/Projects/robodata_Agilex/packages/diffusion_policy
PS_ROOT  = /home/zwa0839/Documents/Projects/pusht_service
```

---

## 0. Environment map

| Process | Python | Key deps | GPU |
|---|---|---|---|
| Blackfly capture service (`pointgrey_capture_service.py`) | **`/usr/bin/python3`** (system 3.10) | `PySpin` + `numpy<2` + `opencv-python-headless` in `~/.local/lib/python3.10/site-packages/` | — |
| `pusht_service` server (`main.py --serve`) | conda env `pusht_service` (3.10) | `piper_sdk`, `pyroki`, `jax`, `viser` | — |
| **`policy_service`** (`policy_service.main`) | conda env `robodiff` (3.9) | `torch`, `hydra` | **cuda:1** (CLI flag) |
| **Coordinator** (`infer_viser` / `infer_loop`) | conda env `robodiff` (3.9) | `viser`, `pupil_apriltags`, `trimesh`, `cv2`, `scipy` — **no torch** | — |

Verified working on this machine:
* PySpin enumerates `Blackfly S BFS-U3-51S5M` (serial **16276900**) under
  `/usr/bin/python3` after `pip install --user 'numpy<2'`.
* Sensor is **mono uint8**, native 2448×2048 @ ~31 fps. The `state_extractor`
  accepts 1- or 3-channel input transparently.
* End-to-end perception (PySpin → AprilTag → double PnP → mesh → voxel) is
  under 50 ms with PnP reproj ≈ 1.2–1.4 px on a desk capture.

> ⚠️ The capture service **must** run under `/usr/bin/python3` (where PySpin
> lives). Do not try to run it inside any conda env — its numpy 2.x conflicts
> with PySpin's numpy 1.x ABI.

> ⚠️ `policy_service` and the coordinator run in the **same** `robodiff`
> env but as **separate processes**. The coordinator no longer imports
> `torch` / `hydra`; only the policy service does. This lets the
> coordinator's viser front-end stay light and the GPU work stay isolated.

---

## 1. Terminal A — Blackfly capture service

Reads frames and pushes them into a Unix socket + shared-memory ring that
the coordinator consumes. **No sudo needed if your user already has
`/dev/bus/usb` access (default after `flir_setup`).** If you see
`SPINNAKER_ERROR_ACCESS_DENIED`, re-run with `sudo` AND pass
`PYTHONPATH=/home/zwa0839/.local/lib/python3.10/site-packages` because
sudo strips user site-packages.

```bash
cd "$DP_ROOT"

/usr/bin/python3 \
    -m exps.exp_pusht_real.spatial_pusht.realrobot.perception.pointgrey_capture_service \
    --width 2448 --height 2048 --fps 30 \
    --serial 16276900 \
    --socket-path /tmp/pointgrey_capture.sock \
    --shm-prefix pointgrey_capture
```

Expected: prints `[PointGreyCaptureService] running`, then frame stats.
Camera resolution **must** equal
`data/realrobot/pointgrey_calibration.json:resolution` (2448×2048) or the
coordinator's calibration merge step refuses to start.

Smoke test (Terminal C/D, after §1 is up, **without** §2/§3/§4):
```bash
conda activate robodiff
cd "$DP_ROOT"
python -c "
from exps.exp_pusht_real.spatial_pusht.realrobot.perception.pointgrey_client import PointGreyCamera
cam = PointGreyCamera(width=2448, height=2048, fps=30,
                     socket_path='/tmp/pointgrey_capture.sock',
                     shm_prefix='pointgrey_capture')
cam.start()
import time, numpy as np
for _ in range(5):
    color, depth, ts = cam.get_frames()
    print('frame', color.shape, color.dtype, 'mean', float(np.mean(color)))
    time.sleep(0.1)
cam.stop()
"
```

---

## 2. Terminal B — `pusht_service`

Owns CAN + arm. Coordinator talks to it over HTTP at `:8012`.

```bash
conda activate pusht_service
cd "$PS_ROOT"
python main.py --serve --api-port 8012
```

Expected: prints `[Server] POST /plan, /update_state, /robot/step` etc.
The viser visualisation is at the printed port; usually `:8011`.

If CAN isn't up yet:
```bash
cd "$PS_ROOT" && sudo ./setup_can.sh
```

Smoke test:
```bash
curl -s http://localhost:8012/health
```

---

## 3. Terminal C — `policy_service`

Loads a trained checkpoint and exposes it over HTTP at `:8014`. This is
the **only** process that imports `torch` / `hydra` and holds GPU
memory; the coordinator just calls `/predict`.

```bash
conda activate robodiff
cd "$DP_ROOT"

python -m exps.exp_pusht_real.spatial_pusht.policy_service.main \
    --ckpt data/outputs/2026.05.16/16.43.56_train_spatial_pusht_occupancy_image_spatial_pusht_image/checkpoints/latest.ckpt \
    --device cuda:1 \
    --api-port 8014
```

Expected (within a few seconds of startup, after the checkpoint loads):

```
[policy-service] listening on http://0.0.0.0:8014
[policy-service] ckpt=…/checkpoints/latest.ckpt
[policy-service] n_obs_steps=2 n_action_steps=8 image_shape=(1, 128, 128) agent_pos_dim=2 action_dim=2
```

`--device cuda:1` keeps the policy off the GPU your training run might be
using on `cuda:0`. Swap checkpoints by editing the `--ckpt` flag — there's
no checkpoint path in any yaml any more.

Smoke tests:

```bash
curl -s http://localhost:8014/health
# {"status":"ok","ready":true}

curl -s http://localhost:8014/status | python -m json.tool
# Shows ckpt_path, device, n_obs_steps, n_action_steps, image_shape, ...
```

The API surface (`/health`, `/status`, `/predict`, `/reset`) is documented
in `spatial_pusht/policy_service/README.md`.

---

## 4. Terminal D — coordinator

The coordinator (`infer_viser` for live use, `infer_loop` for headless
sanity) connects to all three services above. **It does no GPU work.**

### 4a. Viser GUI — primary tool

```bash
conda activate robodiff
cd "$DP_ROOT"

python -m exps.exp_pusht_real.spatial_pusht.realrobot.infer_viser \
    --port 8013
```

On startup it waits up to 60 s for `policy_service /health` to be ready,
then loads the AprilTag model, starts the camera client and the arm
poller, and opens viser at `http://localhost:8013`. The 3D scene shows
the world frame, the bbox cube, a pusher sphere, and the live T-block
mesh. The sidebar carries camera preview, arm controls (connect / lock /
disconnect / unlock), inference controls (Start Auto / Stop / Step Once
/ Reset Obs History), and a 2D occupancy overlay.

Execution is **off** by default. Click `Connect`, watch the camera +
occupancy preview look sane, then tick `Execute (send /robot/step)` and
click `Start Auto`.

CLI flags:

| Flag | Effect |
|---|---|
| `--policy-url URL` | Override `cfg.policy_service.url` (default `http://localhost:8014`) |
| `--pusht-url URL`  | Override `cfg.pusht_service.url` (default `http://localhost:8012`) |
| `--no-arm`         | Skip arm client + reader. Perception + policy still run |
| `--no-camera`      | Skip the camera. Useful for testing GUI / service plumbing |

### 4b. Headless CLI — sanity check

Same backend, no GUI. Useful for confirming the whole chain produces
sensible obs/actions before opening the GUI:

```bash
conda activate robodiff
cd "$DP_ROOT"

# Dry-run: perception + policy, no /robot/step
python -m exps.exp_pusht_real.spatial_pusht.realrobot.infer_loop --dry-run

# Real run (auto-connects arm, sends /robot/step)
python -m exps.exp_pusht_real.spatial_pusht.realrobot.infer_loop
```

It prints `step=N pusher_vox=… tblock_vox_n=… action_voxels=[…] policy=Xms`
on each successful tick.

---

## 5. Shutdown order

1. Ctrl-C the coordinator (Terminal D). It auto-calls `/arm/disconnect`
   (latches a hold), stops the arm poller, and `cam.stop()`s the shm reader.
2. Ctrl-C `policy_service` (Terminal C). The model is dropped; no
   external state to clean up.
3. Ctrl-C `pusht_service` (Terminal B).
4. Ctrl-C / `sudo killall` the capture service (Terminal A). It frees
   `/tmp/pointgrey_capture.sock` and the shm segments on exit.

---

## 6. Artifacts checklist (all already on disk)

```
$DP_ROOT/data/realrobot/
├── model/reference_aligned_to_model_filtered.json
├── model/tblock.ply
├── pointgrey_calibration.json
└── records/arm_world_calibration.json     # T_world_from_base + tip offset
```

Latest checkpoint (point `--ckpt` at this):
```
$DP_ROOT/data/outputs/2026.05.16/16.43.56_train_spatial_pusht_occupancy_image_spatial_pusht_image/checkpoints/latest.ckpt
```

(Also available: `epoch=0050-val_loss=0.0819.ckpt` — currently the best
val_loss on the run.)

---

## 7. Quick recovery

| Symptom | Likely cause | Fix |
|---|---|---|
| Coordinator hangs at `[infer] waiting for policy service at http://localhost:8014` | §3 not started yet, or wrong `--api-port` | Bring up Terminal C; it has 60 s before it errors out |
| `ConnectionError: Cannot reach policy service` | §3 crashed (likely bad `--ckpt` path) | Check Terminal C output; verify `ls -la <ckpt>` |
| `Connection refused` to `:8012` | `pusht_service` not started | Terminal B step |
| `Cannot open shared memory pointgrey_capture` | Capture service down or different shm prefix | Restart §1; check `--shm-prefix` matches yaml |
| `Calibration resolution mismatch` | Camera not running at 2448×2048 | Pass `--width 2448 --height 2048` to capture service |
| `No tags detected` every tick | Lighting / tags out of frame / wrong tag family | Confirm tag36h11 on the table; bright even light |
| State panel shows high `Reproj px:` (>4 px) | Calibration drift, tags partially occluded, lens dirty | Smoother absorbs it (`max_reproj_error_px_for_update=6.0` internally). If voxelized T-block stutters visibly, re-run calibration |
| `tip_in_eef` clearly wrong (pusher off-screen) | Wrong calibration file or wrong robot | Re-run `tools/arm_world_calibration.py` on your arm |
| GUI status shows `arm reader: ... arm not connected` | `pusht_service` is up but you haven't pressed Connect | Click **Connect** in the Arm panel |
| GUI status shows `policy service error: ...` | Policy service went down mid-run | Check Terminal C; restart §3, then click **Reset Obs History** |
