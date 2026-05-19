"""Load a trained diffusion-policy checkpoint and run inference on it.

Supports both spatial_pusht workspaces:

* ``image``  — hybrid image+agent_pos policy (``cfg.task.shape_meta``).
  obs_dict = {"image": (B,T,C,H,W), "agent_pos": (B,T,Da_pos)}
* ``lowdim`` — flat occupancy+agent_pos policy (``cfg.task.obs_dim``).
  obs_dict = {"obs": (B,T,obs_dim)} where each row is
  ``concat(occupancy.flatten(), agent_pos)`` — matching
  ``SpatialPushTOccupancyFlatDataset._sample_to_data``.

Policy kind is detected from the checkpoint cfg, not specified by the
caller. The wire format from the client stays the same in both cases
(image + agent_pos); for lowdim the runner does the flatten+concat
itself before calling ``predict_action``.

Stateless wrapper: the caller is responsible for maintaining the obs
history window of length ``n_obs_steps`` and sending it on every call.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import hydra
import numpy as np
import torch


@dataclass
class PolicyInfo:
    ckpt_path: str
    device: str
    policy_kind: str                                  # "image" | "lowdim"
    n_obs_steps: int
    n_action_steps: int
    agent_pos_dim: int
    action_dim: int
    # Image policy: (C, H, W) expected per frame.
    # Lowdim policy: derived (C, H, W) such that C*H*W == keypoint_dim, or
    # None if it can't be inferred from cfg. Inputs on the wire are still
    # accepted as long as they flatten to the right size.
    image_shape: Optional[tuple[int, int, int]] = None
    # Lowdim only: total flat obs dim and the size of the flattened image slice.
    obs_dim: Optional[int] = None
    keypoint_dim: Optional[int] = None
    # Lowdim only: non-overlapping mean-pool factor applied by the runner
    # before flattening the image. 1 = no pool. The advertised image_shape
    # is the *raw* shape on the wire (pre-pool).
    occupancy_pool: int = 1


def _has(cfg, key: str) -> bool:
    """OmegaConf-friendly hasattr."""
    try:
        return key in cfg
    except Exception:
        return hasattr(cfg, key)


class PolicyRunner:
    def __init__(
        self,
        ckpt_path: "str | Path",
        device: str = "cuda:0",
        num_inference_steps: Optional[int] = None,
        scheduler: Optional[str] = None,
    ):
        self._ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not self._ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self._ckpt_path}")
        self._device = torch.device(device)
        self._lock = threading.Lock()
        self._policy, self._train_cfg = self._load(self._ckpt_path, self._device)
        if scheduler is not None:
            self._swap_scheduler(scheduler)
        if num_inference_steps is not None:
            # Trained schedulers are typically DDPM with num_inference_steps
            # equal to num_train_timesteps (often 100). Reducing this is the
            # single biggest inference-time speedup, at some quality cost.
            trained = int(self._policy.num_inference_steps)
            self._policy.num_inference_steps = int(num_inference_steps)
            print(
                f"[policy-service] num_inference_steps override: "
                f"{trained} -> {self._policy.num_inference_steps}",
                flush=True,
            )
        self.info = self._build_info(self._train_cfg)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load(ckpt_path: Path, device: torch.device):
        payload = torch.load(ckpt_path.open("rb"), map_location="cpu")
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=str(ckpt_path.parent))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema and getattr(workspace, "ema_model", None) is not None:
            policy = workspace.ema_model
        policy.eval().to(device)
        return policy, cfg

    def _swap_scheduler(self, name: str) -> None:
        """Replace the trained scheduler with one of {ddpm, ddim}.

        Uses ``from_config`` so betas / prediction_type / clip_sample carry
        over from the training scheduler -- the model itself is unchanged.
        """
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        key = name.lower()
        cls_map = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}
        if key not in cls_map:
            raise ValueError(f"Unknown scheduler {name!r}; expected one of {list(cls_map)}")
        old = self._policy.noise_scheduler
        new = cls_map[key].from_config(old.config)
        self._policy.noise_scheduler = new
        print(
            f"[policy-service] scheduler override: "
            f"{type(old).__name__} -> {type(new).__name__}",
            flush=True,
        )

    def _build_info(self, cfg) -> PolicyInfo:
        task = cfg.task
        if _has(task, "shape_meta"):
            kind = "image"
            shape_meta = task.shape_meta
            image_shape = tuple(int(v) for v in shape_meta.obs.image.shape)
            agent_pos_shape = tuple(int(v) for v in shape_meta.obs.agent_pos.shape)
            action_shape = tuple(int(v) for v in shape_meta.action.shape)
            return PolicyInfo(
                ckpt_path=str(self._ckpt_path),
                device=str(self._device),
                policy_kind=kind,
                n_obs_steps=int(cfg.n_obs_steps),
                n_action_steps=int(cfg.n_action_steps),
                image_shape=image_shape,
                agent_pos_dim=int(agent_pos_shape[-1]),
                action_dim=int(action_shape[-1]),
            )

        if _has(task, "obs_dim") and _has(task, "action_dim"):
            kind = "lowdim"
            obs_dim = int(task.obs_dim)
            action_dim = int(task.action_dim)
            keypoint_dim = int(task.keypoint_dim) if _has(task, "keypoint_dim") else None
            if keypoint_dim is None:
                # Fall back: assume the trailing slice is agent_pos of dim 2.
                agent_pos_dim = 2
                keypoint_dim = obs_dim - agent_pos_dim
            else:
                agent_pos_dim = obs_dim - keypoint_dim
            if keypoint_dim <= 0 or agent_pos_dim <= 0:
                raise ValueError(
                    f"Bad lowdim shapes: obs_dim={obs_dim} keypoint_dim={keypoint_dim} "
                    f"agent_pos_dim={agent_pos_dim}"
                )
            # Recover the dataset-side pool factor so the runner can apply the
            # same downsample on the wire input before flatten. Default 1 keeps
            # legacy checkpoints (without the field) behaving as before.
            pool = 1
            if _has(task, "dataset") and _has(task.dataset, "occupancy_pool"):
                pool = int(task.dataset.occupancy_pool)
            if pool < 1:
                raise ValueError(f"Bad occupancy_pool={pool}")
            # Best-effort (C, H, W) inference: assume single-channel square grid.
            # keypoint_dim is the *pooled* slice; advertise the *raw* shape that
            # the client should send so the runner can pool it down to match.
            pooled_side = int(round(keypoint_dim ** 0.5))
            if pooled_side * pooled_side != keypoint_dim:
                image_shape = None
            else:
                raw_side = pooled_side * pool
                image_shape = (1, raw_side, raw_side)
            return PolicyInfo(
                ckpt_path=str(self._ckpt_path),
                device=str(self._device),
                policy_kind=kind,
                n_obs_steps=int(cfg.n_obs_steps),
                n_action_steps=int(cfg.n_action_steps),
                image_shape=image_shape,
                agent_pos_dim=agent_pos_dim,
                action_dim=action_dim,
                obs_dim=obs_dim,
                keypoint_dim=keypoint_dim,
                occupancy_pool=pool,
            )

        raise ValueError(
            "Cannot determine policy kind from cfg.task: expected either "
            "`shape_meta.obs.image` (image policy) or `obs_dim`/`action_dim` "
            "(lowdim policy)."
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _preprocess_raw_image_window(self, raw: np.ndarray) -> np.ndarray:
        """Match video_decoder.decode_episode_video + PushTImageDataset.

        Accepts a (T, H, W) gray or (T, H, W, 3) RGB uint8 stack from the
        industry camera and returns a (T, 3, target, target) float32 array
        in [0, 1] ready for the image policy. ``target`` is read from
        ``self.info.image_shape``.
        """
        if self.info.image_shape is None:
            raise ValueError("info.image_shape is None; cannot preprocess raw frames")
        C_t, H_t, W_t = self.info.image_shape
        if C_t != 3 or H_t != W_t:
            raise ValueError(
                f"raw preprocessing currently assumes a 3-channel square target, "
                f"got image_shape={self.info.image_shape}"
            )
        if raw.ndim == 4 and raw.shape[-1] == 3:
            grayT = np.stack(
                [cv2.cvtColor(raw[t], cv2.COLOR_RGB2GRAY) for t in range(raw.shape[0])],
                axis=0,
            )  # (T, H, W) uint8
        elif raw.ndim == 3:
            grayT = raw
        else:
            raise ValueError(
                f"raw image_window must be (T,H,W) or (T,H,W,3) uint8, got {raw.shape}"
            )

        T_, H, W = grayT.shape
        side = min(H, W)
        top = (H - side) // 2
        left = (W - side) // 2
        cropped = grayT[:, top:top + side, left:left + side]  # (T, side, side)
        resized = np.empty((T_, H_t, W_t), dtype=np.uint8)
        for t in range(T_):
            resized[t] = cv2.resize(
                cropped[t], (W_t, H_t), interpolation=cv2.INTER_AREA
            )
        rgb = np.repeat(resized[..., None], 3, axis=-1)  # (T, H_t, W_t, 3) uint8
        return (np.moveaxis(rgb, -1, 1).astype(np.float32) / 255.0)  # (T, 3, H_t, W_t)

    def predict(
        self,
        image_window: np.ndarray,
        agent_pos_window: np.ndarray,
    ) -> dict[str, Any]:
        """Run one inference.

        Args:
            image_window: (T, C, H, W) float array; T must equal n_obs_steps.
                For lowdim policies, this is flattened per-timestep before being
                concatenated with agent_pos.
            agent_pos_window: (T, agent_pos_dim) float array.

        Returns:
            {"action": int64 ndarray (n_action_steps, action_dim) -- snapped to
             integer voxel grid and clipped to image bounds, "took_ms": float}
        """
        T = self.info.n_obs_steps
        if image_window.shape[0] != T:
            raise ValueError(
                f"image_window has T={image_window.shape[0]}, expected n_obs_steps={T}"
            )
        if agent_pos_window.shape[0] != T:
            raise ValueError(
                f"agent_pos_window has T={agent_pos_window.shape[0]}, expected n_obs_steps={T}"
            )
        if agent_pos_window.shape[1] != self.info.agent_pos_dim:
            raise ValueError(
                f"agent_pos_window dim {agent_pos_window.shape[1]} != "
                f"expected {self.info.agent_pos_dim}"
            )

        is_cuda = self._device.type == "cuda"
        def _sync_now() -> float:
            if is_cuda:
                torch.cuda.synchronize(self._device)
            return time.perf_counter()

        stages: list[tuple[str, float]] = []
        t_pre0 = _sync_now()

        if self.info.policy_kind == "image":
            if image_window.dtype == np.uint8:
                # Raw industry-camera frames. Apply the same transform as
                # video_decoder.decode_episode_video used during training:
                # gray -> center-crop square -> resize -> repeat to 3 channels.
                image_window = self._preprocess_raw_image_window(image_window)
            if tuple(image_window.shape[1:]) != self.info.image_shape:
                raise ValueError(
                    f"image_window frame shape {image_window.shape[1:]} != "
                    f"expected {self.info.image_shape}"
                )
            t_h2d0 = time.perf_counter()
            stages.append(("preprocess", (t_h2d0 - t_pre0) * 1000.0))
            image = torch.from_numpy(image_window[None]).to(
                device=self._device, dtype=torch.float32
            )
            agent_pos = torch.from_numpy(agent_pos_window[None]).to(
                device=self._device, dtype=torch.float32
            )
            obs_dict = {"image": image, "agent_pos": agent_pos}

        elif self.info.policy_kind == "lowdim":
            assert self.info.keypoint_dim is not None  # invariant for lowdim
            keypoint_dim = self.info.keypoint_dim
            pool = self.info.occupancy_pool
            if (self.info.image_shape is not None
                    and tuple(image_window.shape[1:]) != self.info.image_shape):
                raise ValueError(
                    f"image_window frame shape {image_window.shape[1:]} != "
                    f"expected {self.info.image_shape} (raw, pre-pool)"
                )
            img = image_window.astype(np.float32)
            # Apply the same non-overlapping mean-pool as the training dataset
            # (see SpatialPushTOccupancyFlatDataset._pool_occupancy) so the wire
            # contract is the raw 128x128 grid even when the model was trained
            # on a pooled grid.
            if pool > 1:
                _, C, H, W = img.shape
                if H % pool != 0 or W % pool != 0:
                    raise ValueError(
                        f"image_window H,W=({H},{W}) not divisible by "
                        f"occupancy_pool={pool}"
                    )
                img = img.reshape(T, C, H // pool, pool, W // pool, pool).mean(axis=(3, 5))
            flat_per_frame = int(np.prod(img.shape[1:]))
            if flat_per_frame != keypoint_dim:
                raise ValueError(
                    f"image_window flattens to {flat_per_frame} after pool={pool}, "
                    f"expected keypoint_dim={keypoint_dim}"
                )
            # Match SpatialPushTOccupancyFlatDataset: [occ.flatten() || agent_pos]
            flat_image = img.reshape(T, keypoint_dim)
            obs_vec = np.concatenate(
                [flat_image, agent_pos_window.astype(np.float32)], axis=-1
            )
            t_h2d0 = time.perf_counter()
            stages.append(("preprocess", (t_h2d0 - t_pre0) * 1000.0))
            obs = torch.from_numpy(obs_vec[None]).to(
                device=self._device, dtype=torch.float32
            )
            obs_dict = {"obs": obs}

        else:
            raise RuntimeError(f"Unknown policy_kind: {self.info.policy_kind!r}")

        t_sample0 = _sync_now()
        stages.append(("h2d", (t_sample0 - t_h2d0) * 1000.0))
        with self._lock, torch.no_grad():
            result = self._policy.predict_action(obs_dict)
        t_d2h0 = _sync_now()
        stages.append(("diffusion_sample", (t_d2h0 - t_sample0) * 1000.0))

        action = result["action"][0].detach().cpu().numpy()
        stages.append(("d2h", (time.perf_counter() - t_d2h0) * 1000.0))
        took_ms = sum(ms for _, ms in stages)
        # Action space is the integer voxel grid. Snap to the nearest voxel.
        # No explicit clip: `noise_scheduler.clip_sample=True` already constrains
        # the normalized sample to [-1, 1], so after LinearNormalizer.unnormalize
        # the value is guaranteed to fall inside the training-data min/max range.
        # This matches the official PushT inference path (env_runner.pusht_image
        # _runner.py:198-208) which feeds the unmodified action straight to env.
        action = np.rint(action).astype(np.int64)

        # Per-prediction log so the operator can sanity-check the chunk against
        # training-data action statistics (every training step is ~1 voxel apart,
        # range x:[11,104] y:[29,110]). Out-of-range here would point at a
        # normalizer / units / orientation mismatch.
        agent_now = agent_pos_window[-1]
        deltas = np.linalg.norm(np.diff(action, axis=0), axis=1) if len(action) > 1 else np.array([])
        breakdown = "  ".join(f"{name}={ms:.1f}ms" for name, ms in stages)
        with np.printoptions(precision=2, suppress=True):
            print(
                f"[policy] took={took_ms:.1f}ms  ({breakdown})  agent_pos_now={agent_now}\n"
                f"[policy] action chunk ({action.shape}):\n{action}\n"
                f"[policy] |delta| consecutive: {deltas}  "
                f"(min={deltas.min() if deltas.size else float('nan'):.3f}, "
                f"max={deltas.max() if deltas.size else float('nan'):.3f}, "
                f"mean={deltas.mean() if deltas.size else float('nan'):.3f})",
                flush=True,
            )
        return {"action": action, "took_ms": took_ms}
