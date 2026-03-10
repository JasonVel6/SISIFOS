from pathlib import Path

import numpy as np

from modules.config import CameraConfig, TrajectoryConfig
from modules.trajectory.generateTrajectoriesUnified import generate_trajectories_dynamical
from modules.trajectory.trajectory_io import read_camera_trajectory


class TestTrajectoryGenerationSmoke:
    def test_fixed_seed_is_reproducible(self, tmp_path):
        results = []
        for run in range(2):
            run_dir = tmp_path / f"run{run}"
            config = TrajectoryConfig(
                path_mode="tumbling", seed=77777, num_agents=1, num_mc=1, tend=3.0, tstep=1.0, MIN_F2F_PX_MED=1.0
            )
            folders = generate_trajectories_dynamical(
                config=config,
                base_output_file=str(run_dir),
                config_prefix="Det",
                camera_config=CameraConfig(resolution=(64, 64)),
            )
            results.append(read_camera_trajectory(str(Path(folders[0]) / "camera_traj.csv")))

        np.testing.assert_array_equal(results[0]["p_G_I"], results[1]["p_G_I"])
