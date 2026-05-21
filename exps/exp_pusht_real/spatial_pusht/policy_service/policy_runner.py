"""Load a trained diffusion-policy checkpoint and run inference on it.

Supports the four spatial_pusht workspaces:

* ``image``         — hybrid image+agent_pos policy (``cfg.task.shape_meta``).
  obs_dict = {"image": (B,T,C,H,W), "agent_pos": (B,T,Da_pos)}
* ``lowdim``        — flat occupancy+agent_pos policy (``cfg.task.obs_dim``).
  obs_dict = {"obs": (B,T,obs_dim)} where each row is
  ``concat(occupancy.flatten(), agent_pos)`` — matching
  ``SpatialPushTOccupancyFlatDataset._sample_to_data``.
* ``tbar_coords``   — padded T-bar voxel coords + agent_pos
  (``cfg.task.tbar_pad_n``). obs_dict = {"obs": (B,T,K*2+2)} where each row
  is ``concat(coords.flatten(), agent_pos)`` — matching
  ``SpatialPushTTBarCoordsDataset._sample_to_data``.
* ``tag_keypoints`` — fixed-slot AprilTag corner xy + agent_pos
  (``cfg.task.n_tag_keypoints``). Wire and obs identical in shape to
  tbar_coords but the slots have *semantic identity* (one per
  (tag_id, corner_idx)), matching ``SpatialPushTTagKeypointsDataset``.

Policy kind is detected from the checkpoint cfg, not specified by the
caller. Wire format depends on the kind (image/lowdim: image+agent_pos;
tbar_coords/tag_keypoints: coords+agent_pos).

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
    policy_kind: str                                  # "image" | "lowdim" | "tbar_coords" | "tag_keypoints"
    n_obs_steps: int
    n_action_steps: int
    agent_pos_dim: int
    action_dim: int
    # Image policy: (C, H, W) expected per frame.
    # Lowdim policy: derived (C, H, W) such that C*H*W == keypoint_dim, or
    # None if it can't be inferred from cfg. Inputs on the wire are still
    # accepted as long as they flatten to the right size.
    # tbar_coords: unused (None).
    image_shape: Optional[tuple[int, int, int]] = None
    # Lowdim / tbar_coords only: total flat obs dim and the size of the
    # non-agent_pos slice (flattened image, or K*2 for tbar_coords).
    obs_dim: Optional[int] = None
    keypoint_dim: Optional[int] = None
    # Lowdim only: non-overlapping mean-pool factor applied by the runner
    # before flattening the image. 1 = no pool. The advertised image_shape
    # is the *raw* shape on the wire (pre-pool).
    occupancy_pool: int = 1
    # tbar_coords only: padded T-bar voxel-set length K. The client must
    # send a (T, K, 2) float array, pre-padded/sub-sampled to this K with
    # sentinel TBAR_PAD (= -1) for empty slots.
    tbar_pad_n: Optional[int] = None
    # tag_keypoints only: number of fixed (tag_id, corner_idx) slots S. The
    # client must send a (T, S, 2) float array of voxel xy per slot (no
    # sentinel — every slot is always populated via T_world_from_object
    # re-projection of the canonical tag corners).
    n_tag_keypoints: Optional[int] = None
    # tag_keypoints only: the AprilTag IDs that contribute slots, in the same
    # ascending order used at training time. S = 4 * len(tag_ids). The
    # inference extractor reads this to restrict its slot ordering away from
    # the full static-model `object_tag_ids` (which may include calibrated-
    # but-unused tags on this rig).
    tag_ids: Optional[list[int]] = None


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

        if _has(task, "n_tag_keypoints") and _has(task, "obs_dim") and _has(task, "action_dim"):
            # Tag-keypoint variant: obs = [tag_keypoints.flatten() || agent_pos],
            # S fixed (tag_id, corner_idx) slots set at build time.
            n_tag_kp = int(task.n_tag_keypoints)
            obs_dim = int(task.obs_dim)
            action_dim = int(task.action_dim)
            agent_pos_dim = obs_dim - n_tag_kp * 2
            if agent_pos_dim <= 0:
                raise ValueError(
                    f"Bad tag_keypoints shapes: obs_dim={obs_dim} "
                    f"n_tag_keypoints={n_tag_kp} -> agent_pos_dim={agent_pos_dim}"
                )
            if not _has(task, "tag_ids"):
                raise ValueError(
                    "tag_keypoints ckpt is missing cfg.task.tag_ids -- "
                    "required so the inference extractor can match the trained "
                    "slot subset (defaulting to all static-model object tags "
                    "may produce a different slot count)."
                )
            tag_ids = [int(t) for t in task.tag_ids]
            if 4 * len(tag_ids) != n_tag_kp:
                raise ValueError(
                    f"tag_ids={tag_ids} implies S={4 * len(tag_ids)} but "
                    f"n_tag_keypoints={n_tag_kp}; rebuild the zarr or fix the "
                    f"task config so they agree.")
            return PolicyInfo(
                ckpt_path=str(self._ckpt_path),
                device=str(self._device),
                policy_kind="tag_keypoints",
                n_obs_steps=int(cfg.n_obs_steps),
                n_action_steps=int(cfg.n_action_steps),
                agent_pos_dim=agent_pos_dim,
                action_dim=action_dim,
                obs_dim=obs_dim,
                keypoint_dim=n_tag_kp * 2,
                n_tag_keypoints=n_tag_kp,
                tag_ids=tag_ids,
            )

        if _has(task, "tbar_pad_n") and _has(task, "obs_dim") and _has(task, "action_dim"):
            # T-bar coords variant: obs = [coords.flatten() || agent_pos],
            # coords have a fixed K = task.tbar_pad_n set at build time.
            tbar_pad_n = int(task.tbar_pad_n)
            obs_dim = int(task.obs_dim)
            action_dim = int(task.action_dim)
            agent_pos_dim = obs_dim - tbar_pad_n * 2
            if agent_pos_dim <= 0:
                raise ValueError(
                    f"Bad tbar_coords shapes: obs_dim={obs_dim} "
                    f"tbar_pad_n={tbar_pad_n} -> agent_pos_dim={agent_pos_dim}"
                )
            return PolicyInfo(
                ckpt_path=str(self._ckpt_path),
                device=str(self._device),
                policy_kind="tbar_coords",
                n_obs_steps=int(cfg.n_obs_steps),
                n_action_steps=int(cfg.n_action_steps),
                agent_pos_dim=agent_pos_dim,
                action_dim=action_dim,
                obs_dim=obs_dim,
                keypoint_dim=tbar_pad_n * 2,
                tbar_pad_n=tbar_pad_n,
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
        image_window: Optional[np.ndarray] = None,
        agent_pos_window: Optional[np.ndarray] = None,
        *,
        coords_window: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Run one inference.

        Exactly one of ``image_window`` / ``coords_window`` is expected per
        invocation; which one depends on ``info.policy_kind``:

        * image / lowdim → image_window: (T, C, H, W) float array (lowdim
          flattens it per-timestep before concatenating with agent_pos).
        * tbar_coords   → coords_window: (T, K, 2) float array, pre-padded
          to K = ``info.tbar_pad_n`` with sentinel TBAR_PAD (= -1).

        ``agent_pos_window`` is always (T, agent_pos_dim) float.

        Returns:
            {"action": int64 ndarray (n_action_steps, action_dim) -- snapped to
             integer voxel grid, "took_ms": float}
        """
        T = self.info.n_obs_steps
        kind = self.info.policy_kind
        coords_kinds = ("tbar_coords", "tag_keypoints")
        if kind in coords_kinds:
            if coords_window is None:
                raise ValueError(f"{kind} policy requires coords_window")
            primary = coords_window
            primary_name = "coords_window"
        else:
            if image_window is None:
                raise ValueError(f"{kind} policy requires image_window")
            primary = image_window
            primary_name = "image_window"
        if primary.shape[0] != T:
            raise ValueError(
                f"{primary_name} has T={primary.shape[0]}, expected n_obs_steps={T}"
            )
        if agent_pos_window is None or agent_pos_window.shape[0] != T:
            raise ValueError(
                f"agent_pos_window has T={None if agent_pos_window is None else agent_pos_window.shape[0]}, "
                f"expected n_obs_steps={T}"
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

        elif self.info.policy_kind in coords_kinds:
            # tbar_coords: K slots, sentinel-padded. tag_keypoints: S slots,
            # always populated. Both flatten to [coords.flatten() || agent_pos].
            S = (self.info.tbar_pad_n if self.info.policy_kind == "tbar_coords"
                 else self.info.n_tag_keypoints)
            assert S is not None
            if coords_window.ndim != 3 or coords_window.shape[1:] != (S, 2):
                raise ValueError(
                    f"coords_window frame shape {coords_window.shape[1:]} != "
                    f"expected ({S}, 2)"
                )
            coords = coords_window.astype(np.float32).reshape(T, S * 2)
            obs_vec = np.concatenate(
                [coords, agent_pos_window.astype(np.float32)], axis=-1
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

        # Re-slice from action_pred to skip the leading "history" action.
        # The policy's default slice is action_pred[To-1 : To-1 + N] -- the
        # first element is the action AT the last observation step, which on
        # the real robot maps to a reconstructed *past* pusher pose (sending
        # the arm there walks it backward at every chunk boundary). Slicing
        # [To : To + N] instead keeps the chunk length at n_action_steps but
        # makes all N elements purely future predictions. Falls back to the
        # original slice if the prediction horizon is too short (defensive;
        # both shipped ckpts have horizon=16 >> To + N).
        To = int(self.info.n_obs_steps)
        N = int(self.info.n_action_steps)
        if "action_pred" in result and result["action_pred"].shape[1] >= To + N:
            action = result["action_pred"][0, To:To + N].detach().cpu().numpy()
        else:
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
