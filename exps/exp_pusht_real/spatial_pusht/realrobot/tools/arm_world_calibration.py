"""Arm-to-world calibration using repeated contact measurements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ArmWorldCalibrationSample:
    """One arm pose sample whose stick tip touches a known world point."""

    target_name: str
    world_point: np.ndarray
    eef_position_base: np.ndarray
    eef_rotation_base: np.ndarray
    qpos_rad: np.ndarray
    gripper_m: float
    timestamp_s: float


@dataclass(frozen=True)
class ArmWorldCalibrationResult:
    """Solved stick-tip offset and base-to-world rigid transform."""

    tip_position_in_eef_m: np.ndarray
    T_world_from_base: np.ndarray
    T_base_from_world: np.ndarray
    rmse_m: float
    max_error_m: float
    sample_counts_by_target: dict[str, int]
    residual_norms_m: np.ndarray


def _sample_to_payload(sample: ArmWorldCalibrationSample) -> dict:
    return {
        "target_name": sample.target_name,
        "world_point": np.asarray(sample.world_point, dtype=np.float64).tolist(),
        "eef_position_base": np.asarray(sample.eef_position_base, dtype=np.float64).tolist(),
        "eef_rotation_base": np.asarray(sample.eef_rotation_base, dtype=np.float64).tolist(),
        "qpos_rad": np.asarray(sample.qpos_rad, dtype=np.float64).tolist(),
        "gripper_m": float(sample.gripper_m),
        "timestamp_s": float(sample.timestamp_s),
    }


def _sample_from_payload(payload: dict) -> ArmWorldCalibrationSample:
    return ArmWorldCalibrationSample(
        target_name=str(payload["target_name"]),
        world_point=np.asarray(payload["world_point"], dtype=np.float64),
        eef_position_base=np.asarray(payload["eef_position_base"], dtype=np.float64),
        eef_rotation_base=np.asarray(payload["eef_rotation_base"], dtype=np.float64),
        qpos_rad=np.asarray(payload["qpos_rad"], dtype=np.float64),
        gripper_m=float(payload["gripper_m"]),
        timestamp_s=float(payload["timestamp_s"]),
    )


def _samples_by_target_to_payload(samples_by_target: dict[str, list[ArmWorldCalibrationSample]] | None) -> dict:
    if samples_by_target is None:
        return {}
    return {
        str(target_name): [_sample_to_payload(sample) for sample in samples]
        for target_name, samples in samples_by_target.items()
    }


def _samples_by_target_from_payload(payload: dict | None) -> dict[str, list[ArmWorldCalibrationSample]]:
    if not payload:
        return {}
    return {
        str(target_name): [_sample_from_payload(sample_payload) for sample_payload in sample_payloads]
        for target_name, sample_payloads in payload.items()
    }


def _estimate_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if source_points.shape != target_points.shape or source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Expected matching Nx3 source/target arrays.")
    if source_points.shape[0] < 3:
        raise ValueError("Need at least 3 point correspondences.")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = target_centered.T @ source_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def _transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        translation, dtype=np.float64
    )


def _build_world_from_base(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _solve_tip_from_repeat_contacts(
    samples: list[ArmWorldCalibrationSample],
) -> np.ndarray | None:
    """Estimate tip offset from repeated touches of the same target.

    For two samples i, j touching the same world point:
        p_i + R_i d = p_j + R_j d
    which gives:
        (R_i - R_j) d = p_j - p_i
    """

    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    by_target: dict[str, list[ArmWorldCalibrationSample]] = {}
    for sample in samples:
        by_target.setdefault(sample.target_name, []).append(sample)

    for target_samples in by_target.values():
        if len(target_samples) < 2:
            continue
        for i in range(len(target_samples)):
            for j in range(i + 1, len(target_samples)):
                sample_i = target_samples[i]
                sample_j = target_samples[j]
                rows_a.append(sample_i.eef_rotation_base - sample_j.eef_rotation_base)
                rows_b.append(sample_j.eef_position_base - sample_i.eef_position_base)

    if not rows_a:
        return None

    matrix_a = np.concatenate(rows_a, axis=0)
    vector_b = np.concatenate(rows_b, axis=0)
    if matrix_a.shape[0] < 3 or np.linalg.matrix_rank(matrix_a) < 3:
        return None

    init_tip, *_ = np.linalg.lstsq(matrix_a, vector_b, rcond=None)

    def residuals(tip_in_eef: np.ndarray) -> np.ndarray:
        return matrix_a @ np.asarray(tip_in_eef, dtype=np.float64) - vector_b

    result = least_squares(
        residuals,
        x0=np.asarray(init_tip, dtype=np.float64),
        method="trf",
        loss="soft_l1",
        f_scale=0.001,
        max_nfev=200,
    )
    return np.asarray(result.x, dtype=np.float64)


def _solve_tip_then_rigid_init(
    world_points: np.ndarray,
    eef_positions_base: np.ndarray,
    eef_rotations_base: np.ndarray,
    tip_init_guesses: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a good initialization by eliminating the rigid transform."""

    def contact_points_base(tip_in_eef: np.ndarray) -> np.ndarray:
        tip_in_eef = np.asarray(tip_in_eef, dtype=np.float64).reshape(3)
        return eef_positions_base + np.einsum("nij,j->ni", eef_rotations_base, tip_in_eef)

    def residuals(tip_in_eef: np.ndarray) -> np.ndarray:
        contacts_base = contact_points_base(tip_in_eef)
        rotation, translation = _estimate_rigid_transform(contacts_base, world_points)
        contacts_world = _transform_points(contacts_base, rotation, translation)
        return (contacts_world - world_points).reshape(-1)

    best = None
    init_guesses = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.05], dtype=np.float64),
        np.array([0.0, 0.0, 0.10], dtype=np.float64),
        np.array([0.0, 0.0, -0.05], dtype=np.float64),
        np.array([0.05, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.05, 0.0], dtype=np.float64),
    ]
    if tip_init_guesses is not None:
        init_guesses = [np.asarray(init, dtype=np.float64).reshape(3) for init in tip_init_guesses] + init_guesses
    for init in init_guesses:
        result = least_squares(
            residuals,
            x0=init,
            method="trf",
            loss="soft_l1",
            f_scale=0.002,
            max_nfev=300,
        )
        if best is None or result.cost < best.cost:
            best = result

    assert best is not None
    tip_position_in_eef_m = np.asarray(best.x, dtype=np.float64).reshape(3)
    contacts_base = contact_points_base(tip_position_in_eef_m)
    rotation, translation = _estimate_rigid_transform(contacts_base, world_points)
    return tip_position_in_eef_m, rotation, translation


def solve_arm_world_calibration(
    samples: list[ArmWorldCalibrationSample],
    *,
    _allow_outlier_rejection: bool = True,
) -> ArmWorldCalibrationResult:
    """Jointly solve tip-in-EEF and base-to-world using known contact points."""

    if len(samples) < 3:
        raise ValueError("Need at least 3 recorded arm poses.")

    target_names = sorted({sample.target_name for sample in samples})
    if len(target_names) < 3:
        raise ValueError("Need samples from all 3 calibration targets.")

    world_points = np.asarray([sample.world_point for sample in samples], dtype=np.float64)
    eef_positions_base = np.asarray([sample.eef_position_base for sample in samples], dtype=np.float64)
    eef_rotations_base = np.asarray([sample.eef_rotation_base for sample in samples], dtype=np.float64)

    def contact_points_base(tip_in_eef: np.ndarray) -> np.ndarray:
        tip_in_eef = np.asarray(tip_in_eef, dtype=np.float64).reshape(3)
        return eef_positions_base + np.einsum("nij,j->ni", eef_rotations_base, tip_in_eef)

    repeat_contact_tip = _solve_tip_from_repeat_contacts(samples)
    tip_seed_guesses = [repeat_contact_tip] if repeat_contact_tip is not None else None
    init_tip, init_rotation, init_translation = _solve_tip_then_rigid_init(
        world_points,
        eef_positions_base,
        eef_rotations_base,
        tip_init_guesses=tip_seed_guesses,
    )
    init_rotvec = Rotation.from_matrix(init_rotation).as_rotvec()

    def residuals_joint(params: np.ndarray) -> np.ndarray:
        tip_in_eef = np.asarray(params[:3], dtype=np.float64)
        rotation = Rotation.from_rotvec(np.asarray(params[3:6], dtype=np.float64)).as_matrix()
        translation = np.asarray(params[6:9], dtype=np.float64)
        contacts_base = contact_points_base(tip_in_eef)
        contacts_world = _transform_points(contacts_base, rotation, translation)
        return (contacts_world - world_points).reshape(-1)

    candidate_inits = [np.concatenate([init_tip, init_rotvec, init_translation])]
    if repeat_contact_tip is not None:
        repeat_contacts_base = contact_points_base(repeat_contact_tip)
        repeat_rotation, repeat_translation = _estimate_rigid_transform(repeat_contacts_base, world_points)
        candidate_inits.append(
            np.concatenate(
                [
                    repeat_contact_tip,
                    Rotation.from_matrix(repeat_rotation).as_rotvec(),
                    repeat_translation,
                ]
            )
        )

    best_result = None
    for candidate_init in candidate_inits:
        result = least_squares(
            residuals_joint,
            x0=candidate_init,
            method="trf",
            loss="soft_l1",
            f_scale=0.002,
            max_nfev=400,
        )
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    assert best_result is not None
    result = best_result
    tip_position_in_eef_m = np.asarray(result.x[:3], dtype=np.float64)
    rotation = Rotation.from_rotvec(np.asarray(result.x[3:6], dtype=np.float64)).as_matrix()
    translation = np.asarray(result.x[6:9], dtype=np.float64)
    contacts_base = contact_points_base(tip_position_in_eef_m)
    contacts_world = _transform_points(contacts_base, rotation, translation)
    residual_vectors = contacts_world - world_points
    residual_norms_m = np.linalg.norm(residual_vectors, axis=1)

    T_world_from_base = _build_world_from_base(rotation, translation)
    T_base_from_world = np.linalg.inv(T_world_from_base)
    sample_counts_by_target = {
        target_name: sum(1 for sample in samples if sample.target_name == target_name)
        for target_name in sorted({sample.target_name for sample in samples})
    }
    solution = ArmWorldCalibrationResult(
        tip_position_in_eef_m=tip_position_in_eef_m,
        T_world_from_base=T_world_from_base,
        T_base_from_world=T_base_from_world,
        rmse_m=float(np.sqrt(np.mean(np.square(residual_norms_m)))),
        max_error_m=float(np.max(residual_norms_m)),
        sample_counts_by_target=sample_counts_by_target,
        residual_norms_m=residual_norms_m,
    )
    if _allow_outlier_rejection and len(samples) >= 6:
        median = float(np.median(residual_norms_m))
        mad = float(np.median(np.abs(residual_norms_m - median)))
        robust_sigma = 1.4826 * mad
        outlier_threshold_m = max(0.0035, median + 2.5 * robust_sigma)
        inlier_samples = [
            sample
            for sample, residual_norm in zip(samples, residual_norms_m)
            if float(residual_norm) <= outlier_threshold_m
        ]
        inlier_target_names = {sample.target_name for sample in inlier_samples}
        if (
            3 <= len(inlier_target_names) == 3
            and len(inlier_samples) >= max(6, len(samples) - 2)
            and len(inlier_samples) < len(samples)
        ):
            refined_solution = solve_arm_world_calibration(
                inlier_samples,
                _allow_outlier_rejection=False,
            )
            if refined_solution.rmse_m <= solution.rmse_m:
                return refined_solution
    return solution


def save_arm_world_calibration(
    result: ArmWorldCalibrationResult,
    output_path: str | Path,
    *,
    samples_by_target: dict[str, list[ArmWorldCalibrationSample]] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "arm_world_from_apriltag_points_v1",
        "tip_position_in_eef_m": result.tip_position_in_eef_m.tolist(),
        "T_world_from_base": result.T_world_from_base.tolist(),
        "T_base_from_world": result.T_base_from_world.tolist(),
        "rmse_m": float(result.rmse_m),
        "max_error_m": float(result.max_error_m),
        "sample_counts_by_target": result.sample_counts_by_target,
        "residual_norms_m": result.residual_norms_m.tolist(),
        "samples_by_target": _samples_by_target_to_payload(samples_by_target),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_arm_world_calibration(
    path: str | Path,
    *,
    return_samples: bool = False,
    rerun_from_samples: bool = True,
) -> ArmWorldCalibrationResult | tuple[ArmWorldCalibrationResult | None, dict[str, list[ArmWorldCalibrationSample]]]:
    path = Path(path)
    if not path.exists():
        return (None, {}) if return_samples else None
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples_by_target = _samples_by_target_from_payload(payload.get("samples_by_target"))

    result: ArmWorldCalibrationResult | None
    all_samples = [sample for samples in samples_by_target.values() for sample in samples]
    if rerun_from_samples and all_samples:
        result = solve_arm_world_calibration(all_samples)
    else:
        result = ArmWorldCalibrationResult(
        tip_position_in_eef_m=np.asarray(payload["tip_position_in_eef_m"], dtype=np.float64),
        T_world_from_base=np.asarray(payload["T_world_from_base"], dtype=np.float64),
        T_base_from_world=np.asarray(payload["T_base_from_world"], dtype=np.float64),
        rmse_m=float(payload["rmse_m"]),
        max_error_m=float(payload["max_error_m"]),
        sample_counts_by_target={str(k): int(v) for k, v in payload["sample_counts_by_target"].items()},
        residual_norms_m=np.asarray(payload.get("residual_norms_m", []), dtype=np.float64),
        )
    if return_samples:
        return result, samples_by_target
    return result
