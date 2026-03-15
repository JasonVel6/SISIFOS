"""
Seed screening utilities for rejecting poorly conditioned trajectories before rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from modules.config import InertiaConfig, SceneConfig
from modules.trajectory.trajectory_io import read_camera_trajectory
from modules.trajectory.trajectory_math import q2R


def _safe_percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu = np.asarray(u, dtype=float)
    vv = np.asarray(v, dtype=float)
    denom = np.linalg.norm(uu, axis=-1) * np.linalg.norm(vv, axis=-1)
    valid = denom > 0.0
    dots = np.zeros(uu.shape[:-1], dtype=float)
    dots[valid] = np.sum(uu[valid] * vv[valid], axis=-1) / denom[valid]
    dots = np.clip(dots, -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def _rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-10:
        return np.zeros(3, dtype=float)

    sin_angle = np.sin(angle)
    if abs(sin_angle) < 1e-8:
        eigvals, eigvecs = np.linalg.eig(R)
        axis = np.real(eigvecs[:, np.argmin(np.abs(np.real(eigvals) - 1.0))])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return np.zeros(3, dtype=float)
        axis = axis / axis_norm
        return axis * angle

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    ) / (2.0 * sin_angle)
    return axis * angle


def _characteristic_target_span_m(inertia: InertiaConfig) -> float:
    if inertia.inertia_type == "box":
        dims = [float(v) for v in (inertia.l, inertia.w, inertia.h) if v is not None]
        if not dims:
            return 1.0
        return float(np.linalg.norm(dims))
    if inertia.inertia_type == "cylinder":
        if inertia.r is None or inertia.h is None:
            return 1.0
        return float(np.sqrt((2.0 * inertia.r) ** 2 + inertia.h**2))
    if inertia.inertia_type == "sphere":
        if inertia.r is None:
            return 1.0
        return float(2.0 * inertia.r)
    if inertia.inertia_type == "custom":
        dims = [float(v) for v in (inertia.Jx, inertia.Jy, inertia.Jz) if v is not None]
        if not dims:
            return 1.0
        return float(max(dims))
    return 1.0


class ScreeningCheck(BaseModel):
    name: str
    value: float
    threshold: float
    relation: str
    passed: bool
    reason: str


class AgentScreeningReport(BaseModel):
    agent_folder: str
    verdict: str
    score: float
    seed: int | None
    metrics: dict[str, float]
    failed_checks: list[ScreeningCheck]
    passed_checks: list[ScreeningCheck]
    warning_checks: list[ScreeningCheck]
    diagnosis: list[str]


class SeedScreeningReport(BaseModel):
    verdict: str
    seed: int | None
    config_prefix: str
    attempts_used: int = 1
    agent_reports: list[AgentScreeningReport]
    diagnosis: list[str]
    warnings: list[str] = []


def _check_min(name: str, value: float, threshold: float, reason: str) -> ScreeningCheck:
    return ScreeningCheck(
        name=name,
        value=value,
        threshold=threshold,
        relation=">=",
        passed=bool(value >= threshold),
        reason=reason,
    )


def _check_max(name: str, value: float, threshold: float, reason: str) -> ScreeningCheck:
    return ScreeningCheck(
        name=name,
        value=value,
        threshold=threshold,
        relation="<=",
        passed=bool(value <= threshold),
        reason=reason,
    )


def _warn_min(name: str, value: float, threshold: float, reason: str) -> ScreeningCheck:
    return ScreeningCheck(
        name=name,
        value=value,
        threshold=threshold,
        relation="warn_if_<",
        passed=bool(value >= threshold),
        reason=reason,
    )


def _warn_max(name: str, value: float, threshold: float, reason: str) -> ScreeningCheck:
    return ScreeningCheck(
        name=name,
        value=value,
        threshold=threshold,
        relation="warn_if_>",
        passed=bool(value <= threshold),
        reason=reason,
    )


def evaluate_agent_trajectory(agent_folder: Path, config: SceneConfig) -> AgentScreeningReport:
    screening = config.screening
    trajectory = read_camera_trajectory(str(agent_folder / "camera_traj.csv"))

    p_G_I = trajectory["p_G_I"]
    p_C_I = trajectory["p_C_I"]
    q_I_G = trajectory["q_I_G"]
    q_I_C = trajectory["q_I_C"]
    if len(p_G_I) < 2:
        diagnosis = ["trajectory is too short to estimate screening geometry"]
        return AgentScreeningReport(
            agent_folder=str(agent_folder),
            verdict="REJECT",
            score=0.0,
            seed=config.trajectory.seed,
            metrics={"num_frames": float(len(p_G_I))},
            failed_checks=[
                ScreeningCheck(
                    name="num_frames",
                    value=float(len(p_G_I)),
                    threshold=2.0,
                    relation=">=",
                    passed=False,
                    reason=diagnosis[0],
                )
            ],
            passed_checks=[],
            warning_checks=[],
            diagnosis=diagnosis,
        )

    horizon = max(1, min(int(screening.horizon_frames), len(p_G_I) - 1))

    r_TC_I = p_C_I - p_G_I
    ranges = np.linalg.norm(r_TC_I, axis=1)
    safe_ranges = np.maximum(ranges, 1e-9)
    u_TC_I = r_TC_I / safe_ranges[:, None]

    body_los = np.zeros_like(u_TC_I)
    R_CG_series = []
    for i, q in enumerate(q_I_G):
        R_IG = q2R(q)
        body_los[i] = R_IG.T @ u_TC_I[i]
        R_IC = q2R(q_I_C[i])
        R_CG_series.append(R_IC.T @ R_IG)
    R_CG_series = np.asarray(R_CG_series, dtype=float)

    delta = r_TC_I[horizon:] - r_TC_I[:-horizon]
    u_mid = u_TC_I[:-horizon]
    delta_norm = np.linalg.norm(delta, axis=1)
    radial = np.sum(delta * u_mid, axis=1)
    transverse = delta - radial[:, None] * u_mid
    transverse_norm = np.linalg.norm(transverse, axis=1)
    mean_range = 0.5 * (safe_ranges[horizon:] + safe_ranges[:-horizon])

    los_change_deg = _angle_deg(u_TC_I[:-horizon], u_TC_I[horizon:])
    body_view_change_deg = _angle_deg(body_los[:-horizon], body_los[horizon:])
    transverse_baseline_ratio = transverse_norm / np.maximum(mean_range, 1e-9)
    radial_dominance = np.abs(radial) / np.maximum(delta_norm, 1e-9)

    relrot_step_deg = []
    optical_axis_spin_deg = []
    tilt_component_deg = []
    optical_axis_spin_fraction = []
    for i in range(len(R_CG_series) - horizon):
        dR = R_CG_series[i + horizon] @ R_CG_series[i].T
        rotvec = _rotvec_from_matrix(dR)
        mag = float(np.linalg.norm(rotvec))
        spin = float(abs(rotvec[2]))
        tilt = float(np.linalg.norm(rotvec[:2]))
        relrot_step_deg.append(np.rad2deg(mag))
        optical_axis_spin_deg.append(np.rad2deg(spin))
        tilt_component_deg.append(np.rad2deg(tilt))
        optical_axis_spin_fraction.append(spin / max(mag, 1e-12))
    relrot_step_deg = np.asarray(relrot_step_deg, dtype=float)
    optical_axis_spin_deg = np.asarray(optical_axis_spin_deg, dtype=float)
    tilt_component_deg = np.asarray(tilt_component_deg, dtype=float)
    optical_axis_spin_fraction = np.asarray(optical_axis_spin_fraction, dtype=float)

    target_span_m = _characteristic_target_span_m(config.trajectory.inertia_config)
    projected_diameter_px = config.camera.focal_length_px * target_span_m / safe_ranges
    image_motion_proxy_px = config.camera.focal_length_px * transverse_baseline_ratio

    metrics = {
        "range_min_m": float(np.min(ranges)),
        "range_max_m": float(np.max(ranges)),
        "range_mean_m": float(np.mean(ranges)),
        "target_span_m": float(target_span_m),
        "target_diameter_p10_px": _safe_percentile(projected_diameter_px, 10),
        "target_diameter_median_px": _safe_percentile(projected_diameter_px, 50),
        "image_motion_p25_px": _safe_percentile(image_motion_proxy_px, 25),
        "image_motion_median_px": _safe_percentile(image_motion_proxy_px, 50),
        "los_parallax_p25_deg": _safe_percentile(los_change_deg, 25),
        "los_parallax_median_deg": _safe_percentile(los_change_deg, 50),
        "body_view_change_p25_deg": _safe_percentile(body_view_change_deg, 25),
        "body_view_change_median_deg": _safe_percentile(body_view_change_deg, 50),
        "transverse_baseline_ratio_p25": _safe_percentile(transverse_baseline_ratio, 25),
        "transverse_baseline_ratio_median": _safe_percentile(transverse_baseline_ratio, 50),
        "radial_dominance_p75": _safe_percentile(radial_dominance, 75),
        "radial_dominance_mean": _safe_mean(radial_dominance),
        "relative_rotation_p25_deg": _safe_percentile(relrot_step_deg, 25),
        "relative_rotation_median_deg": _safe_percentile(relrot_step_deg, 50),
        "tilt_component_p25_deg": _safe_percentile(tilt_component_deg, 25),
        "tilt_component_median_deg": _safe_percentile(tilt_component_deg, 50),
        "optical_axis_spin_p75_deg": _safe_percentile(optical_axis_spin_deg, 75),
        "optical_axis_spin_fraction_p75": _safe_percentile(optical_axis_spin_fraction, 75),
        "optical_axis_spin_fraction_mean": _safe_mean(optical_axis_spin_fraction),
        "horizon_frames": float(horizon),
    }

    checks = [
        _check_min(
            "target_diameter_p10_px",
            metrics["target_diameter_p10_px"],
            screening.min_target_diameter_p10_px,
            "target appears too small for stable feature extraction over much of the run",
        ),
        _check_min(
            "image_motion_p25_px",
            metrics["image_motion_p25_px"],
            screening.min_image_motion_p25_px,
            "projected translational motion is too weak to support robust feature tracks",
        ),
        _check_min(
            "los_parallax_p25_deg",
            metrics["los_parallax_p25_deg"],
            screening.min_los_parallax_p25_deg,
            "line-of-sight change is too small, so monocular triangulation is poorly conditioned",
        ),
        _check_min(
            "body_view_change_p25_deg",
            metrics["body_view_change_p25_deg"],
            screening.min_body_view_change_p25_deg,
            "the target face being observed changes too little over time",
        ),
        _check_min(
            "transverse_baseline_ratio_p25",
            metrics["transverse_baseline_ratio_p25"],
            screening.min_transverse_baseline_ratio_p25,
            "baseline is too radial relative to range, limiting useful parallax",
        ),
        _check_max(
            "radial_dominance_p75",
            metrics["radial_dominance_p75"],
            screening.max_radial_dominance_p75,
            "camera motion is too line-of-sight dominated instead of lateral",
        ),
        _check_min(
            "relative_rotation_p25_deg",
            metrics["relative_rotation_p25_deg"],
            screening.min_relative_rotation_p25_deg,
            "target-relative appearance changes too little between frames",
        ),
        _check_min(
            "tilt_component_p25_deg",
            metrics["tilt_component_p25_deg"],
            screening.min_tilt_component_p25_deg,
            "relative attitude change is dominated by in-plane spin instead of viewpoint tilt",
        ),
        _check_max(
            "optical_axis_spin_fraction_p75",
            metrics["optical_axis_spin_fraction_p75"],
            screening.max_optical_axis_spin_fraction_p75,
            "target-relative motion is dominated by optical-axis spin, which can destabilize feature tracks",
        ),
    ]

    warning_checks = [
        _warn_min(
            "los_parallax_p25_deg",
            metrics["los_parallax_p25_deg"],
            screening.warn_los_parallax_p25_deg,
            "line-of-sight change is still borderline low for monocular VO, even though the seed passes the hard gate",
        ),
        _warn_min(
            "transverse_baseline_ratio_p25",
            metrics["transverse_baseline_ratio_p25"],
            screening.warn_transverse_baseline_ratio_p25,
            "lateral baseline is still modest relative to range, so triangulation may remain fragile",
        ),
        _warn_max(
            "radial_dominance_p75",
            metrics["radial_dominance_p75"],
            screening.warn_radial_dominance_p75,
            "motion remains fairly line-of-sight dominated, which can limit useful parallax despite acceptance",
        ),
        _warn_max(
            "optical_axis_spin_fraction_p75",
            metrics["optical_axis_spin_fraction_p75"],
            screening.warn_optical_axis_spin_fraction_p75,
            "a large fraction of target-relative motion is still optical-axis spin, which can destabilize VO tracks",
        ),
    ]

    failed_checks = [check for check in checks if not check.passed]
    passed_checks = [check for check in checks if check.passed]
    triggered_warnings = [check for check in warning_checks if not check.passed]
    diagnosis = [check.reason for check in failed_checks]
    for check in triggered_warnings:
        if check.reason not in diagnosis:
            diagnosis.append(check.reason)
    passed_ratio = len(passed_checks) / max(len(checks), 1)
    score = float(np.clip(passed_ratio, 0.0, 1.0))

    return AgentScreeningReport(
        agent_folder=str(agent_folder),
        verdict="ACCEPT" if not failed_checks else "REJECT",
        score=score,
        seed=config.trajectory.seed,
        metrics=metrics,
        failed_checks=failed_checks,
        passed_checks=passed_checks,
        warning_checks=triggered_warnings,
        diagnosis=diagnosis,
    )


def evaluate_seed_candidate(agent_folders: list[Path], config: SceneConfig, config_prefix: str) -> SeedScreeningReport:
    agent_reports = [evaluate_agent_trajectory(agent_folder, config) for agent_folder in agent_folders]
    if config.screening.require_all_agents_pass:
        accepted = all(report.verdict == "ACCEPT" for report in agent_reports)
    else:
        accepted = any(report.verdict == "ACCEPT" for report in agent_reports)

    diagnosis: list[str] = []
    warnings: list[str] = []
    for report in agent_reports:
        for reason in report.diagnosis:
            if reason not in diagnosis:
                diagnosis.append(reason)
        for warning in report.warning_checks:
            if warning.reason not in warnings:
                warnings.append(warning.reason)

    return SeedScreeningReport(
        verdict="ACCEPT" if accepted else "REJECT",
        seed=config.trajectory.seed,
        config_prefix=config_prefix,
        agent_reports=agent_reports,
        diagnosis=diagnosis,
        warnings=warnings,
    )


def write_screening_report(report: SeedScreeningReport, output_path: Path) -> None:
    output_path.write_text(json.dumps(report.model_dump(), indent=2))
