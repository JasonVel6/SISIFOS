import json
from pathlib import Path

import numpy as np

from modules.config import CameraConfig, InitialConditionConfig, TrajectoryConfig, default_inertia_config
from modules.trajectory.generateTrajectoriesUnified import generate_trajectories_dynamical
from modules.trajectory.motion_cases import init_tumbling_new
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

    def test_tumbling_init_condition_config_is_resolved_and_saved(self, tmp_path):
        run_dir = tmp_path / "run_ic_cfg"
        config = TrajectoryConfig(
            inertia_config=default_inertia_config("RF_Hubble"),
            path_mode="tumbling",
            seed=12345,
            illumination_seed=54321,
            num_agents=1,
            num_mc=1,
            tend=3.0,
            tstep=1.0,
            MIN_F2F_PX_MED=1.0,
            init_condition_config=InitialConditionConfig(
                x=80.0,
                y=-5.0,
                z=2.5,
                xdot=0.01,
                ydot=-0.02,
                zdot=0.03,
                omega=(0.04, -0.01, 0.02),
                inclination_rad=0.4,
                eccentricity=0.02,
                sun_elevation_I_rad=0.1,
                sun_azimuth_I_rad=0.2,
                yaw=0.3,
                pitch=0.4,
                roll=0.5,
            ),
        )
        folders = generate_trajectories_dynamical(
            config=config,
            base_output_file=str(run_dir),
            config_prefix="ICCfg",
            camera_config=CameraConfig(resolution=(64, 64)),
            save_scene_plots=False,
        )

        agent_dir = Path(folders[0])
        with open(run_dir / "initial_configs.json") as f:
            initial_configs = json.load(f)
        with open(agent_dir / "initial_config.json") as f:
            resolved_initial = json.load(f)
        with open(agent_dir / "trajectory_config.json") as f:
            resolved_trajectory = json.load(f)

        expected_initial = {
            "x": 80.0,
            "y": -5.0,
            "z": 2.5,
            "xdot": 0.01,
            "ydot": -0.02,
            "zdot": 0.03,
        }
        for key, value in expected_initial.items():
            assert resolved_initial[key] == value
            assert initial_configs["mc_00"]["agent_00"][key] == value

        assert resolved_initial["omega"] == [0.04, -0.01, 0.02]
        assert resolved_trajectory["init_condition_config"]["x"] == 80.0

    def test_tumbling_grid_sampling_spreads_cro_ranges_evenly(self):
        rngs_mc = [np.random.default_rng(100), np.random.default_rng(101)]
        init_cfg = InitialConditionConfig(
            sampling_mode="grid sampling",
            R_mid_range=(10.0, 40.0),
            span_frac_range=(1.0, 2.0),
            phi=0.0,
            omega_mag_range=(0.1, 0.4),
        )

        _, _, _, _, _, _, omega_GI_G_0, cro_fields = init_tumbling_new(
            num_mc=2,
            num_agents=2,
            rngs_mc=rngs_mc,
            n_scalar=0.05,
            init_condition_config=init_cfg,
            J=default_inertia_config("RF_Hubble").J,
        )

        observed_pairs = {
            (round(float(r_mid), 6), round(float(span_frac), 6))
            for r_mid, span_frac in zip(cro_fields["R_mid"].ravel(), cro_fields["span_frac"].ravel())
        }
        assert observed_pairs == {
            (10.0, 1.0),
            (10.0, 2.0),
            (40.0, 1.0),
            (40.0, 2.0),
        }
        np.testing.assert_allclose(np.sort(cro_fields["phi"].ravel()), np.zeros(4))
        np.testing.assert_allclose(np.linalg.norm(omega_GI_G_0, axis=1), np.array([0.1, 0.4]))

    def test_grid_sampling_mode_is_saved_to_output_configs(self, tmp_path):
        run_dir = tmp_path / "run_grid_cfg"
        config = TrajectoryConfig(
            inertia_config=default_inertia_config("RF_Hubble"),
            path_mode="tumbling",
            seed=24680,
            illumination_seed=13579,
            num_agents=1,
            num_mc=1,
            tend=2.0,
            tstep=1.0,
            MIN_F2F_PX_MED=1.0,
            init_condition_config=InitialConditionConfig(
                sampling_mode="grid sampling",
                R_mid_range=(20.0, 30.0),
                span_frac=1.5,
                phi=0.25,
            ),
        )

        folders = generate_trajectories_dynamical(
            config=config,
            base_output_file=str(run_dir),
            config_prefix="GridCfg",
            camera_config=CameraConfig(resolution=(64, 64)),
            save_scene_plots=False,
        )

        agent_dir = Path(folders[0])
        with open(agent_dir / "initial_config.json") as f:
            resolved_initial = json.load(f)
        with open(agent_dir / "trajectory_config.json") as f:
            resolved_trajectory = json.load(f)

        assert resolved_initial["sampling_mode"] == "grid sampling"
        assert resolved_trajectory["init_condition_config"]["sampling_mode"] == "grid sampling"
