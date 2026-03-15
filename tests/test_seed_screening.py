import csv
from pathlib import Path

import numpy as np

from modules.config import SceneConfig
from modules.trajectory.seed_screening import evaluate_agent_trajectory


def _yaw_quat(deg: float) -> list[float]:
    half = np.deg2rad(deg) * 0.5
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _roll_quat(deg: float) -> list[float]:
    half = np.deg2rad(deg) * 0.5
    return [float(np.cos(half)), float(np.sin(half)), 0.0, 0.0]


def _write_camera_traj(
    agent_dir: Path,
    camera_positions: np.ndarray,
    q_i_g: np.ndarray | None = None,
    q_i_c: np.ndarray | None = None,
) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "p_G_I_x",
        "p_G_I_y",
        "p_G_I_z",
        "q_I_G_w",
        "q_I_G_x",
        "q_I_G_y",
        "q_I_G_z",
        "p_C_I_x",
        "p_C_I_y",
        "p_C_I_z",
        "q_I_C_w",
        "q_I_C_x",
        "q_I_C_y",
        "q_I_C_z",
        "sun_az",
        "sun_el",
    ]

    if q_i_g is None:
        q_i_g = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (len(camera_positions), 1))
    if q_i_c is None:
        q_i_c = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (len(camera_positions), 1))

    with open(agent_dir / "camera_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, p_C_I in enumerate(camera_positions):
            writer.writerow(
                [
                    float(i),
                    0.0,
                    0.0,
                    0.0,
                    float(q_i_g[i, 0]),
                    float(q_i_g[i, 1]),
                    float(q_i_g[i, 2]),
                    float(q_i_g[i, 3]),
                    float(p_C_I[0]),
                    float(p_C_I[1]),
                    float(p_C_I[2]),
                    float(q_i_c[i, 0]),
                    float(q_i_c[i, 1]),
                    float(q_i_c[i, 2]),
                    float(q_i_c[i, 3]),
                    0.0,
                    0.0,
                ]
            )


def test_seed_screening_accepts_lateral_motion(tmp_path):
    config = SceneConfig(
        selected_model="RF_Integral",
        screening={
            "enabled": True,
            "horizon_frames": 2,
            "min_target_diameter_p10_px": 20.0,
            "min_image_motion_p25_px": 2.0,
            "min_los_parallax_p25_deg": 0.1,
            "min_body_view_change_p25_deg": 0.1,
            "min_transverse_baseline_ratio_p25": 0.001,
            "max_radial_dominance_p75": 0.9,
        },
    )
    agent_dir = tmp_path / "Agent_0"
    xs = np.linspace(-4.0, 4.0, 12)
    camera_positions = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 25.0)])
    _write_camera_traj(agent_dir, camera_positions)

    report = evaluate_agent_trajectory(agent_dir, config)

    assert report.verdict == "ACCEPT"
    assert report.failed_checks == []
    assert report.metrics["image_motion_p25_px"] > 2.0


def test_seed_screening_rejects_radial_motion(tmp_path):
    config = SceneConfig(
        selected_model="RF_Integral",
        screening={
            "enabled": True,
            "horizon_frames": 2,
            "min_target_diameter_p10_px": 20.0,
            "min_image_motion_p25_px": 2.0,
            "min_los_parallax_p25_deg": 0.1,
            "min_body_view_change_p25_deg": 0.1,
            "min_transverse_baseline_ratio_p25": 0.001,
            "max_radial_dominance_p75": 0.8,
        },
    )
    agent_dir = tmp_path / "Agent_0"
    zs = np.linspace(20.0, 28.0, 12)
    camera_positions = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])
    _write_camera_traj(agent_dir, camera_positions)

    report = evaluate_agent_trajectory(agent_dir, config)

    failed_names = {check.name for check in report.failed_checks}
    assert report.verdict == "REJECT"
    assert "image_motion_p25_px" in failed_names
    assert "radial_dominance_p75" in failed_names


def test_seed_screening_reports_orientation_metrics(tmp_path):
    config = SceneConfig(
        selected_model="RF_Integral",
        screening={
            "enabled": True,
            "horizon_frames": 1,
            "min_target_diameter_p10_px": 20.0,
            "min_image_motion_p25_px": 0.0,
            "min_los_parallax_p25_deg": 0.0,
            "min_body_view_change_p25_deg": 0.0,
            "min_transverse_baseline_ratio_p25": 0.0,
            "max_radial_dominance_p75": 1.0,
            "min_relative_rotation_p25_deg": 1.0,
            "max_optical_axis_spin_fraction_p75": 0.8,
        },
    )
    agent_dir = tmp_path / "Agent_0"
    xs = np.linspace(-1.0, 1.0, 10)
    camera_positions = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 25.0)])
    q_i_g = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (len(camera_positions), 1))
    q_i_c = np.array([_yaw_quat(3.0 * i) for i in range(len(camera_positions))], dtype=float)
    _write_camera_traj(agent_dir, camera_positions, q_i_g=q_i_g, q_i_c=q_i_c)

    report = evaluate_agent_trajectory(agent_dir, config)

    failed_names = {check.name for check in report.failed_checks}
    assert report.metrics["relative_rotation_p25_deg"] > 2.5
    assert report.metrics["optical_axis_spin_fraction_p75"] > 0.95
    assert report.verdict == "REJECT"
    assert "optical_axis_spin_fraction_p75" in failed_names


def test_seed_screening_can_measure_tilt_component(tmp_path):
    config = SceneConfig(
        selected_model="RF_Integral",
        screening={
            "enabled": True,
            "horizon_frames": 1,
            "min_target_diameter_p10_px": 20.0,
            "min_image_motion_p25_px": 0.0,
            "min_los_parallax_p25_deg": 0.0,
            "min_body_view_change_p25_deg": 0.0,
            "min_transverse_baseline_ratio_p25": 0.0,
            "max_radial_dominance_p75": 1.0,
            "min_relative_rotation_p25_deg": 1.0,
            "min_tilt_component_p25_deg": 2.0,
        },
    )
    agent_dir = tmp_path / "Agent_0"
    xs = np.linspace(-1.0, 1.0, 10)
    camera_positions = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 25.0)])
    q_i_g = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (len(camera_positions), 1))
    q_i_c = np.array([_roll_quat(3.0 * i) for i in range(len(camera_positions))], dtype=float)
    _write_camera_traj(agent_dir, camera_positions, q_i_g=q_i_g, q_i_c=q_i_c)

    report = evaluate_agent_trajectory(agent_dir, config)

    assert report.metrics["tilt_component_p25_deg"] > 2.5


def test_seed_screening_accepts_borderline_seed_with_warnings(tmp_path):
    config = SceneConfig(
        selected_model="RF_Integral",
        screening={
            "enabled": True,
            "horizon_frames": 2,
            "min_target_diameter_p10_px": 20.0,
            "min_image_motion_p25_px": 1.0,
            "min_los_parallax_p25_deg": 0.02,
            "min_body_view_change_p25_deg": 0.0,
            "min_transverse_baseline_ratio_p25": 0.0004,
            "max_radial_dominance_p75": 1.0,
            "warn_los_parallax_p25_deg": 0.05,
            "warn_transverse_baseline_ratio_p25": 0.001,
        },
    )
    agent_dir = tmp_path / "Agent_0"
    xs = np.linspace(-0.5, 0.5, 12)
    camera_positions = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 25.0)])
    _write_camera_traj(agent_dir, camera_positions)

    report = evaluate_agent_trajectory(agent_dir, config)

    warning_names = {check.name for check in report.warning_checks}
    assert report.verdict == "ACCEPT"
    assert "los_parallax_p25_deg" in warning_names
    assert "transverse_baseline_ratio_p25" in warning_names
