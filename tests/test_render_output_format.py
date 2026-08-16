"""Beauty-pass output format: the PNG default must stay untouched.

Pure-pydantic checks, so these run without Blender. The Blender-side assertions
(image_settings actually applied, .exr extension) are exercised by the
sensor-aware pilot's apparatus and by test_e2e_pipeline under bpy.
"""

import pytest
from pydantic import ValidationError

from modules.config import RenderConfig


class TestRenderOutputFormat:
    def test_default_is_unchanged_display_referred_png(self):
        """Existing renders must not shift encoding just because the knob exists."""
        cfg = RenderConfig()
        assert cfg.output_format == "PNG"
        assert cfg.color_depth is None  # -> resolves to "8" for PNG
        assert cfg.cycles_seed == 0

    def test_exr_is_opt_in(self):
        cfg = RenderConfig(output_format="OPEN_EXR")
        assert cfg.output_format == "OPEN_EXR"
        assert cfg.color_depth is None  # -> resolves to "32" for OPEN_EXR

    def test_explicit_color_depth_round_trips(self):
        assert RenderConfig(output_format="OPEN_EXR", color_depth="16").color_depth == "16"
        assert RenderConfig(color_depth="16").color_depth == "16"

    def test_unknown_format_is_rejected(self):
        """Guard against a typo silently falling back to PNG mid-experiment."""
        with pytest.raises(ValidationError):
            RenderConfig(output_format="TIFF")

    def test_cycles_seed_is_settable(self):
        assert RenderConfig(cycles_seed=20260816).cycles_seed == 20260816
