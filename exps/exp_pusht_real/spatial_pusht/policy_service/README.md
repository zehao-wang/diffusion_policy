# Diffusion-policy HTTP service

Wraps a trained `spatial_pusht_image` checkpoint behind a JSON HTTP API.
Designed to be started independently of the realrobot viser coordinator
so the GPU-heavy model lives in its own process (and conda env).

API style mirrors `pusht_service`: stdlib `http.server`, JSON in/out, no
extra runtime deps beyond what training already needs (`torch`, `hydra`).

## Start the service

Run from the diffusion_policy repo root so `-m` resolves the package:

```bash
cd /home/zwa0839/Documents/Projects/robodata_Agilex/packages/diffusion_policy
conda activate robodiff

python -m exps.exp_pusht_real.spatial_pusht.policy_service.main \
    --ckpt data/outputs/2026.05.16/16.43.56_train_spatial_pusht_occupancy_image_spatial_pusht_image/checkpoints/latest.ckpt \
    --device cuda:1 \
    --api-port 8014
```

On startup the service loads the checkpoint, prefers the EMA model when
`training.use_ema` is true, and prints the resolved metadata
(`n_obs_steps`, `n_action_steps`, expected `image_shape`, `agent_pos_dim`,
`action_dim`).

## Endpoints

| Method | Path     | Body | Returns |
|--------|----------|------|---------|
| GET    | /health  | —    | `{"status":"ok","ready":bool}` |
| GET    | /status  | —    | `{"status":"ok","ready":true,"policy":{ckpt_path, device, n_obs_steps, n_action_steps, image_shape, agent_pos_dim, action_dim}}` |
| POST   | /predict | `{"agent_pos": <(T,agent_pos_dim) list>,` plus **one of** `"image": <(T,C,H,W) list>` (image/lowdim ckpts) or `"coords": <(T,K,2) list>` (tbar_coords ckpts; pre-pad/sub-sample to K = `status.tbar_pad_n`) | `{"action": <(n_action_steps,action_dim) list>, "took_ms": float}` |
| POST   | /reset   | `{}` | `{"status":"ok"}` (no-op; diffusion policy is stateless) |

Errors return HTTP 4xx with `{"error": "..."}` for bad input, 503 before
the model finishes loading, 500 for unexpected exceptions.

`T` must equal `n_obs_steps` (see `/status`). The caller is responsible
for maintaining the obs history window.

## Quick check

```bash
# Health
curl http://localhost:8014/health

# Metadata
curl http://localhost:8014/status | python -m json.tool
```

## Client usage (from the coordinator)

```python
from exps.exp_pusht_real.spatial_pusht.policy_service.client import PolicyClient

client = PolicyClient("http://localhost:8014")
client.wait_ready(timeout_s=60)

info = client.status()
T = info.n_obs_steps                          # 2
C, H, W = info.image_shape                    # (1, 128, 128)

image_window = np.zeros((T, C, H, W), dtype=np.float32)
agent_pos_window = np.zeros((T, info.agent_pos_dim), dtype=np.float32)

result = client.predict(image_window, agent_pos_window)
# result["action"]: (n_action_steps, action_dim) float32
# result["took_ms"]: server-side inference latency
```

## Notes

* The HTTP server is `ThreadingHTTPServer` so `/health` and `/status`
  stay responsive while `/predict` runs. The model itself is serialised
  by an internal lock — concurrent `/predict` calls execute one at a
  time, not in parallel.
* Wire format is plain JSON. For the spatial_pusht obs (one float32
  channel at 128×128 plus a 2-D pose, T=2), a single request encodes to
  ~2 MB of text. At 10 Hz this is well within the loopback bandwidth.
  Switch to msgpack if you ever move the service across the network.
* On startup the runner picks a policy kind from `cfg.task`:
  `shape_meta` → `image`, `tbar_pad_n` → `tbar_coords`, otherwise
  `obs_dim`/`action_dim` → `lowdim`. `status.policy_kind` tells the client
  which wire field to send (`image` vs `coords`).
