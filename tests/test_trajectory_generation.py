from pathlib import Path

import numpy as np

from modules.config import CameraConfig, TrajectoryConfig, default_inertia_config
from modules.trajectory.generateTrajectoriesUnified import generate_trajectories_dynamical
from modules.trajectory.trajectory_io import read_camera_trajectory


class TestTrajectoryGenerationSmoke:
    def test_fixed_seed_is_reproducible(self, tmp_path):
        results = []
        for run in range(2):
            run_dir = tmp_path / f"run{run}"
            config = TrajectoryConfig(
                inertia_config=default_inertia_config("RF_Hubble"),
                path_mode="tumbling",
                seed=77777,
                num_agents=1,
                num_mc=1,
                tend=3.0,
                tstep=1.0,
                MIN_F2F_PX_MED=1.0,
            )
            folders = generate_trajectories_dynamical(
                config=config,
                base_output_file=str(run_dir),
                config_prefix="Det",
                camera_config=CameraConfig(resolution=(64, 64)),
            )
            results.append(read_camera_trajectory(str(Path(folders[0]) / "camera_traj.csv")))

        np.testing.assert_array_equal(results[0]["p_G_I"], results[1]["p_G_I"])

    def test_illumination_seed_decouples_sun_from_trajectory(self, tmp_path):
        results = []
        for run_seed in (11111, 22222):
            run_dir = tmp_path / f"run_{run_seed}"
            config = TrajectoryConfig(
                inertia_config=default_inertia_config("RF_Hubble"),
                path_mode="tumbling",
                seed=run_seed,
                illumination_seed=500001,
                num_agents=1,
                num_mc=1,
                tend=3.0,
                tstep=1.0,
                MIN_F2F_PX_MED=1.0,
            )
            folders = generate_trajectories_dynamical(
                config=config,
                base_output_file=str(run_dir),
                config_prefix="Illum",
                camera_config=CameraConfig(resolution=(64, 64)),
            )
            results.append(read_camera_trajectory(str(Path(folders[0]) / "camera_traj.csv")))

        np.testing.assert_array_equal(results[0]["sun_az"], results[1]["sun_az"])
        np.testing.assert_array_equal(results[0]["sun_el"], results[1]["sun_el"])
        assert not np.array_equal(results[0]["p_G_I"], results[1]["p_G_I"])

    def test_camera_lookat_modes_cover_i_g_and_hybrid(self, tmp_path):
        results = {}
        for lookat_mode in ("I", "G", "hybrid-weak", "hybrid-medium", "hybrid-strong"):
            run_dir = tmp_path / f"run_{lookat_mode}"
            config = TrajectoryConfig(
                inertia_config=default_inertia_config("RF_Integral"),
                path_mode="tumbling",
                seed=24680,
                illumination_seed=13579,
                num_agents=1,
                num_mc=1,
                tend=3.0,
                tstep=1.0,
                MIN_F2F_PX_MED=1.0,
                pointing_offset_G=[0.8, 0.5, 0.2],
                camera_lookat_mode=lookat_mode,
                camera_pitchyaw_follow_gain=0.55,
                camera_roll_follow_gain=0.2,
            )
            folders = generate_trajectories_dynamical(
                config=config,
                base_output_file=str(run_dir),
                config_prefix=f"Lookat_{lookat_mode}",
                camera_config=CameraConfig(resolution=(64, 64)),
                save_scene_plots=False,
            )
            results[lookat_mode] = read_camera_trajectory(str(Path(folders[0]) / "camera_traj.csv"))

        q_i = results["I"]["q_I_C"]
        q_g = results["G"]["q_I_C"]
        q_h_weak = results["hybrid-weak"]["q_I_C"]
        q_h_medium = results["hybrid-medium"]["q_I_C"]
        q_h_strong = results["hybrid-strong"]["q_I_C"]

        assert not np.allclose(q_i, q_g)
        assert not np.allclose(q_h_weak, q_i)
        assert not np.allclose(q_h_weak, q_g)
        assert not np.allclose(q_h_medium, q_h_weak)
        assert not np.allclose(q_h_strong, q_h_medium)
