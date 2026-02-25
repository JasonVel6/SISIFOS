import pytest
from types import SimpleNamespace
from modules.config import CameraConfig
from modules.addon_ground_truth_generation import get_camera_parameters_intrinsic


def _make_fake_scene(
    focal_length_mm=50.0, sensor_width_mm=36.0, res_x=1920, res_y=1080, shift_x=0.0
):
    cam_data = SimpleNamespace(
        lens=focal_length_mm,
        sensor_width=sensor_width_mm,
        sensor_height=24.0,
        sensor_fit="AUTO",
        shift_x=shift_x,
        shift_y=0.0,
    )
    camera = SimpleNamespace(data=cam_data)
    render = SimpleNamespace(
        resolution_x=res_x,
        resolution_y=res_y,
        resolution_percentage=100,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    )
    return SimpleNamespace(camera=camera, render=render)


class TestCameraIntrinsicConsistency:
    @pytest.mark.parametrize(
        "focal_mm,sensor_w,res", [(50.0, 36.0, 1920), (100.0, 36.0, 480)]
    )
    def test_focal_length_px_matches_addon(self, focal_mm, sensor_w, res):
        config = CameraConfig(
            focal_length=focal_mm, sensor_width=sensor_w, resolution=(res, res)
        )
        scene = _make_fake_scene(
            focal_length_mm=focal_mm, sensor_width_mm=sensor_w, res_x=res, res_y=res
        )
        f_x, f_y, _, _ = get_camera_parameters_intrinsic(scene)

        assert abs(f_x - config.focal_length_px) < 1e-6
        assert abs(f_y - config.focal_length_px) < 1e-6
