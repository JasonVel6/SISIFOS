import pytest

from modules.config import InertiaConfig, SceneConfig, SweepConfig, default_inertia_config


def test_scene_config_propagates_selected_model_to_trajectory_defaults():
    config = SceneConfig(selected_model="RF_Integral")

    assert config.trajectory.inertia_config == default_inertia_config("RF_Integral")
    assert config.model_rotation_A_model_euler == (-90.0, 0.0, 0.0)


def test_scene_config_rejects_unknown_selected_model():
    with pytest.raises(ValueError, match="Unknown selected_model"):
        SceneConfig(selected_model="DOES_NOT_EXIST")


def test_sweep_selected_model_revalidates_dependent_defaults():
    sweep = SweepConfig(
        base_config=SceneConfig(selected_model="RF_Hubble"),
        sweep_parameters={"selected_model": ["RF_Integral"]},
    )

    [config] = sweep.generate_sweep_configs()

    assert config.selected_model == "RF_Integral"
    assert config.trajectory.inertia_config == default_inertia_config("RF_Integral")
    assert config.model_rotation_A_model_euler == (-90.0, 0.0, 0.0)


def test_sweep_can_override_default_scene_selected_model():
    sweep = SweepConfig.model_validate(
        {
            "base_config": {
                "trajectory_type": "const_rotate",
            },
            "sweep_parameters": {"selected_model": ["RF_Integral"]},
        }
    )

    [config] = sweep.generate_sweep_configs()

    assert config.selected_model == "RF_Integral"
    assert config.trajectory.inertia_config == default_inertia_config("RF_Integral")


def test_scene_config_preserves_explicit_inertia_override():
    override = InertiaConfig(inertia_type="custom", Jx=10.0, Jy=20.0, Jz=30.0)

    config = SceneConfig(
        selected_model="RF_Integral",
        trajectory={"inertia_config": override.model_dump()},
    )

    assert config.trajectory.inertia_config == override
    assert config.model_rotation_A_model_euler == (-90.0, 0.0, 0.0)


def test_scene_config_rejects_conflicting_nested_selected_model():
    with pytest.raises(ValueError, match="trajectory.selected_model must match selected_model"):
        SceneConfig.model_validate(
            {
                "selected_model": "RF_Integral",
                "trajectory": {"selected_model": "RF_Hubble"},
            }
        )
