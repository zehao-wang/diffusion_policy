"""HTTP JSON API server wrapping a `PolicyRunner`.

Endpoints
---------
GET  /health   Liveness check (no model touch).
GET  /status   Loaded model metadata: ckpt path, device, policy_kind
               ("image" | "lowdim"), n_obs_steps, n_action_steps,
               image_shape (may be null for lowdim), agent_pos_dim,
               action_dim, plus obs_dim/keypoint_dim for lowdim.
POST /predict  Body: {"image": [[[[...]]]] of shape (T,C,H,W),
                      "agent_pos": [[...]] of shape (T,agent_pos_dim)}
               The wire format is the same for image and lowdim policies;
               the server flattens+concats internally for lowdim.
               Returns: {"action": [[...]] of shape (n_action_steps,action_dim),
                         "took_ms": float}
POST /reset    No-op for the stateless diffusion policy. Returned for
               protocol parity with arm services that have buffered state.

Uses stdlib `http.server` (no extra deps) to match `pusht_service`. Runs
on a `ThreadingHTTPServer` so /health and /status stay responsive while
/predict is busy; the model itself is serialised by an internal lock.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .policy_runner import PolicyRunner


# Will be set by `serve()` before the server starts accepting requests.
_RUNNER: PolicyRunner | None = None


class _PolicyHandler(BaseHTTPRequestHandler):
    _POST_ROUTES = {"/predict", "/reset"}
    _GET_ROUTES = {"/health", "/status"}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path not in self._GET_ROUTES:
            self._respond(404, {"error": f"Unknown endpoint: {self.path}"})
            return
        try:
            handler = getattr(self, f"_handle_{self.path.strip('/')}")
            handler()
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):  # noqa: N802
        if self.path not in self._POST_ROUTES:
            self._respond(404, {"error": f"Unknown endpoint: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError) as exc:
            self._respond(400, {"error": f"Invalid JSON: {exc}"})
            return
        try:
            handler = getattr(self, f"_handle_{self.path.strip('/')}")
            handler(body)
        except ValueError as exc:
            self._respond(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _handle_health(self) -> None:
        self._respond(200, {"status": "ok", "ready": _RUNNER is not None})

    def _handle_status(self) -> None:
        if _RUNNER is None:
            self._respond(503, {"error": "Policy not loaded yet"})
            return
        info = asdict(_RUNNER.info)
        if info.get("image_shape") is not None:
            info["image_shape"] = list(info["image_shape"])
        self._respond(200, {"status": "ok", "ready": True, "policy": info})

    def _handle_predict(self, body: dict) -> None:
        if _RUNNER is None:
            self._respond(503, {"error": "Policy not loaded yet"})
            return

        agent_pos = body.get("agent_pos")
        image = body.get("image")
        coords = body.get("coords")
        if agent_pos is None or (image is None and coords is None):
            raise ValueError(
                "Body must contain 'agent_pos' plus one of 'image' / 'coords'"
            )
        if image is not None and coords is not None:
            raise ValueError("Body must contain only one of 'image' / 'coords'")

        agent_pos_arr = np.asarray(agent_pos, dtype=np.float32)
        if coords is not None:
            coords_arr = np.asarray(coords, dtype=np.float32)
            result = _RUNNER.predict(
                agent_pos_window=agent_pos_arr,
                coords_window=coords_arr,
            )
        else:
            # Preserve int dtype so the runner can detect raw uint8 camera frames
            # (preprocessed server-side); pre-processed floats stay float.
            image_arr = np.asarray(image)
            if image_arr.dtype.kind in ("i", "u"):
                image_arr = image_arr.astype(np.uint8)
            else:
                image_arr = image_arr.astype(np.float32)
            result = _RUNNER.predict(
                image_window=image_arr,
                agent_pos_window=agent_pos_arr,
            )
        self._respond(
            200,
            {
                "action": result["action"].tolist(),
                "took_ms": float(result["took_ms"]),
            },
        )

    def _handle_reset(self, _body: dict) -> None:
        # Diffusion policy is stateless across calls; nothing to reset.
        self._respond(200, {"status": "ok"})

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------
    def _respond(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        # Compact one-line logging instead of the noisy default.
        print(f"[policy-service] {self.address_string()} - {fmt % args}")


def serve(
    runner: PolicyRunner,
    *,
    host: str = "0.0.0.0",
    port: int = 8014,
) -> None:
    global _RUNNER
    _RUNNER = runner
    httpd = ThreadingHTTPServer((host, port), _PolicyHandler)
    httpd.daemon_threads = True
    print(f"[policy-service] listening on http://{host}:{port}")
    print(f"[policy-service] ckpt={runner.info.ckpt_path}")
    extra = ""
    if runner.info.policy_kind == "lowdim":
        extra = f" obs_dim={runner.info.obs_dim} keypoint_dim={runner.info.keypoint_dim}"
    elif runner.info.policy_kind == "tbar_coords":
        extra = (
            f" obs_dim={runner.info.obs_dim} keypoint_dim={runner.info.keypoint_dim} "
            f"tbar_pad_n={runner.info.tbar_pad_n}"
        )
    elif runner.info.policy_kind == "tag_keypoints":
        extra = (
            f" obs_dim={runner.info.obs_dim} keypoint_dim={runner.info.keypoint_dim} "
            f"n_tag_keypoints={runner.info.n_tag_keypoints} "
            f"tag_ids={runner.info.tag_ids}"
        )
    print(
        f"[policy-service] policy_kind={runner.info.policy_kind} "
        f"n_obs_steps={runner.info.n_obs_steps} "
        f"n_action_steps={runner.info.n_action_steps} "
        f"image_shape={runner.info.image_shape} "
        f"agent_pos_dim={runner.info.agent_pos_dim} "
        f"action_dim={runner.info.action_dim}"
        + extra
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[policy-service] interrupted")
    finally:
        httpd.server_close()
        print("[policy-service] stopped")
