"""CLI entry-point for the diffusion-policy HTTP service.

Run from the diffusion_policy repo root so package imports resolve:

    python -m exps.exp_pusht_real.spatial_pusht.policy_service.main \
        --ckpt data/outputs/2026.05.16/.../checkpoints/latest.ckpt \
        --device cuda:1 \
        --api-port 8014

The service then accepts inference requests over HTTP. See README.md
for the API surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .policy_runner import PolicyRunner
from .server import serve


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serve a trained diffusion-policy checkpoint over HTTP.",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to a `latest.ckpt` produced by spatial_pusht training.",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for the policy (e.g. cuda:0, cuda:1, cpu).",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--api-port", type=int, default=8014)
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "Override the trained scheduler's inference step count. "
            "Trained DDPM value is typically 100; 50/20 trade quality for "
            "lower latency. Leave unset to keep the trained value."
        ),
    )
    p.add_argument(
        "--scheduler",
        choices=["ddpm", "ddim"],
        default=None,
        help=(
            "Replace the trained noise scheduler at load time. The model "
            "weights are unchanged. 'ddim' tolerates far fewer inference "
            "steps (try 10-16) with similar action quality."
        ),
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    runner = PolicyRunner(
        Path(args.ckpt),
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        scheduler=args.scheduler,
    )
    serve(runner, host=args.host, port=args.api_port)


if __name__ == "__main__":
    main()
