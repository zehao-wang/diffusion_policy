"""CLI entry-point for the viser EVALUATION GUI.

Same wiring as `infer_viser.py` — parses flags, builds the camera /
arm / policy subsystems, hands them to `InferLoopRunner` — but launches
`EvalViserApp` instead of `InferViserApp`. The eval app drops the
free-running auto/step/reset controls in favor of a Start-Trial flow
that runs the standard pusht evaluation protocol (stop on >=95%
coverage or after max_steps).

Run from the diffusion_policy repo root:

    python -m exps.exp_pusht_real.spatial_pusht.realrobot.eval_viser \
        --port 8013
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .gui.eval_app import EvalViserApp
from .infer_loop import InferLoopRunner, build_subsystems, load_cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Viser GUI coordinator for spatial_pusht real-robot evaluation."
    )
    p.add_argument(
        "--config",
        default=str(
            Path(__file__).parent / "configs" / "realrobot.yaml"
        ),
        help="Path to realrobot.yaml.",
    )
    p.add_argument("--host", default="0.0.0.0", help="Viser bind host.")
    p.add_argument("--port", type=int, default=8013, help="Viser HTTP port.")

    p.add_argument(
        "--policy-url",
        default=None,
        help="Override cfg.policy_service.url (e.g. http://localhost:8014).",
    )
    p.add_argument(
        "--pusht-url",
        default=None,
        help="Override cfg.pusht_service.url (e.g. http://localhost:8012).",
    )

    p.add_argument(
        "--no-arm",
        action="store_true",
        help="Skip the arm client + reader. Perception + policy still run.",
    )
    p.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip the camera. Perception/policy will report status='camera disabled'.",
    )
    return p


def _apply_overrides(cfg, args: argparse.Namespace) -> None:
    if args.policy_url:
        cfg.policy_service.url = args.policy_url
    if args.pusht_url:
        cfg.pusht_service.url = args.pusht_url


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(args.config)
    _apply_overrides(cfg, args)

    subsystems = build_subsystems(
        cfg,
        no_arm=args.no_arm,
        no_camera=args.no_camera,
    )
    runner = InferLoopRunner(**subsystems)
    app = EvalViserApp(runner, host=args.host, port=args.port)
    app.run()


if __name__ == "__main__":
    main()
