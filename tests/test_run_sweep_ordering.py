from pathlib import Path

from main import run_sweep
from modules.config import SceneConfig, SweepConfig


def test_run_sweep_generates_all_trajectories_before_rendering(monkeypatch, tmp_path):
    sweep = SweepConfig(
        base_config=SceneConfig(
            scene_blend_path=str(tmp_path / "scene.blend"),
            hdri_path=str(tmp_path / "stars.exr"),
            objects={},
        ),
        sweep_parameters={"trajectory.camera_lookat_mode": ["I", "G", "hybrid"]},
    )

    events = []

    monkeypatch.setattr("main.get_timestamp_folder", lambda: "ordering_test")
    monkeypatch.setattr("main.setup_logger", lambda log_file: None)

    def fake_generate_trajectories(config, output_dir, config_prefix):
        events.append(("generate", config_prefix, config.trajectory.camera_lookat_mode))
        agent_dir = Path(output_dir) / f"{config_prefix}_{config.trajectory.camera_lookat_mode}" / "Agent_0"
        agent_dir.mkdir(parents=True, exist_ok=True)
        return [str(agent_dir)]

    def fake_run_sisfos_with_config(config, agent_folder):
        events.append(("render", Path(agent_folder).name, config.trajectory.camera_lookat_mode))

    monkeypatch.setattr("main.generate_trajectories", fake_generate_trajectories)
    monkeypatch.setattr("main.run_sisfos_with_config", fake_run_sisfos_with_config)

    run_sweep(sweep)

    assert [event[0] for event in events[:3]] == ["generate", "generate", "generate"]
    assert [event[0] for event in events[3:]] == ["render", "render", "render"]
