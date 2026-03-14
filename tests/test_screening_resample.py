from pathlib import Path

from main import _generate_screened_trajectories
from modules.config import SceneConfig
from modules.trajectory.seed_screening import AgentScreeningReport, SeedScreeningReport


def test_generate_screened_trajectories_resamples_until_accept(monkeypatch, tmp_path):
    config = SceneConfig(
        selected_model="RF_Integral",
        screening={"enabled": True, "max_attempts": 3},
        trajectory={"seed": 12345},
    )
    output_dir = tmp_path / "renders"
    screening_root = tmp_path / "screening"
    output_dir.mkdir()
    screening_root.mkdir()

    generated_seeds = []

    def fake_generate_trajectories(local_config, base_dir, config_prefix):
        generated_seeds.append(local_config.trajectory.seed)
        run_dir = Path(base_dir) / f"{config_prefix}_RF_Integral"
        agent_dir = run_dir / "Agent_0"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (Path(base_dir) / f"{config_prefix}_trajectory.json").write_text("{}")
        return [str(agent_dir)]

    def fake_evaluate_seed_candidate(agent_folders, local_config, config_prefix):
        verdict = "REJECT" if len(generated_seeds) == 1 else "ACCEPT"
        agent_report = AgentScreeningReport(
            agent_folder=str(agent_folders[0]),
            verdict=verdict,
            score=0.0 if verdict == "REJECT" else 1.0,
            seed=local_config.trajectory.seed,
            metrics={},
            failed_checks=[],
            passed_checks=[],
            warning_checks=[],
            diagnosis=["too little parallax"] if verdict == "REJECT" else [],
        )
        return SeedScreeningReport(
            verdict=verdict,
            seed=local_config.trajectory.seed,
            config_prefix=config_prefix,
            agent_reports=[agent_report],
            diagnosis=agent_report.diagnosis,
            warnings=[],
        )

    monkeypatch.setattr("main.generate_trajectories", fake_generate_trajectories)
    monkeypatch.setattr("main.evaluate_seed_candidate", fake_evaluate_seed_candidate)
    monkeypatch.setattr("main.write_screening_report", lambda report, output_path: None)

    agent_folders, screening_info = _generate_screened_trajectories(
        config,
        output_dir,
        config_prefix="Config_1",
        screening_root=screening_root,
    )

    assert len(generated_seeds) == 2
    assert generated_seeds[0] != generated_seeds[1]
    assert config.trajectory.seed == generated_seeds[1]
    assert screening_info["status"] == "accepted"
    assert screening_info["accepted_seed"] == generated_seeds[1]
    assert agent_folders == [output_dir / "Config_1_RF_Integral" / "Agent_0"]
