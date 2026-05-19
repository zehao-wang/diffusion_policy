"""Thin HTTP client for the diffusion-policy service.

Coordinator-side helper. Uses `urllib` (no extra deps) so it can run in
either the robodiff env or a leaner deployment env.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib import error, request

import numpy as np


@dataclass
class PolicyStatus:
    ckpt_path: str
    device: str
    policy_kind: str                          # "image" | "lowdim"
    n_obs_steps: int
    n_action_steps: int
    agent_pos_dim: int
    action_dim: int
    # Image policy: (C, H, W). Lowdim policy: best-effort inferred shape or None.
    image_shape: Optional[tuple[int, int, int]] = None
    # Lowdim only.
    obs_dim: Optional[int] = None
    keypoint_dim: Optional[int] = None


class PolicyClient:
    def __init__(
        self,
        service_url: str = "http://127.0.0.1:8014",
        *,
        request_timeout_s: float = 30.0,
    ):
        self._url = service_url.rstrip("/")
        self._timeout = float(request_timeout_s)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def health(self) -> dict:
        return self._get("/health", timeout=2.0)

    def status(self) -> PolicyStatus:
        body = self._get("/status")
        policy = body["policy"]
        raw_image_shape = policy.get("image_shape")
        image_shape: Optional[tuple[int, int, int]] = (
            tuple(int(v) for v in raw_image_shape)  # type: ignore[assignment]
            if raw_image_shape is not None
            else None
        )
        obs_dim = policy.get("obs_dim")
        keypoint_dim = policy.get("keypoint_dim")
        return PolicyStatus(
            ckpt_path=str(policy["ckpt_path"]),
            device=str(policy["device"]),
            policy_kind=str(policy.get("policy_kind", "image")),
            n_obs_steps=int(policy["n_obs_steps"]),
            n_action_steps=int(policy["n_action_steps"]),
            image_shape=image_shape,
            agent_pos_dim=int(policy["agent_pos_dim"]),
            action_dim=int(policy["action_dim"]),
            obs_dim=int(obs_dim) if obs_dim is not None else None,
            keypoint_dim=int(keypoint_dim) if keypoint_dim is not None else None,
        )

    def wait_ready(self, timeout_s: float = 60.0, poll_interval_s: float = 1.0) -> None:
        deadline = time.time() + float(timeout_s)
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                if self.health().get("ready", False):
                    return
            except Exception as exc:
                last_err = exc
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Policy service at {self._url} not ready within {timeout_s:.1f}s "
            f"(last error: {last_err})"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        image_window: np.ndarray,
        agent_pos_window: np.ndarray,
    ) -> dict[str, Any]:
        """Send one obs window through the policy.

        Args:
            image_window: either (T, C, H, W) float (pre-processed occupancy
                for lowdim ckpts) OR (T, H, W, 3) / (T, H, W) uint8 raw camera
                frames (for image ckpts; the server applies the training-time
                transform).
            agent_pos_window: (T, agent_pos_dim) float.

        Returns:
            {"action": int64 ndarray (n_action_steps, action_dim) -- integer
             voxel coordinates, "took_ms": float}
        """
        image_arr = np.asarray(image_window)
        # Preserve uint8 raw frames; everything else goes as float32.
        if image_arr.dtype != np.uint8:
            image_arr = image_arr.astype(np.float32)
        payload = {
            "image": image_arr.tolist(),
            "agent_pos": np.asarray(agent_pos_window, dtype=np.float32).tolist(),
        }
        body = self._post("/predict", payload)
        return {
            "action": np.asarray(body["action"], dtype=np.int64),
            "took_ms": float(body.get("took_ms", float("nan"))),
        }

    def reset(self) -> dict:
        return self._post("/reset", {})

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _get(self, path: str, *, timeout: float | None = None) -> dict:
        return self._request("GET", path, None, timeout=timeout)

    def _post(self, path: str, payload: dict, *, timeout: float | None = None) -> dict:
        return self._request("POST", path, payload, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None,
        *,
        timeout: float | None,
    ) -> dict:
        url = f"{self._url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = request.Request(url, data=data, headers=headers, method=method)
        t = self._timeout if timeout is None else float(timeout)
        try:
            with request.urlopen(req, timeout=t) as resp:
                return json.loads(resp.read())
        except error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"{method} {path} -> {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise ConnectionError(f"Cannot reach policy service at {url}: {exc}") from exc
