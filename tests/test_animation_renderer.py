"""
Tests for the keyframe-animation rendering pipeline.

These tests run inside Blender (via run_in_blender.py) and verify that
render_animation produces the same data products as the old per-frame path.

NOTE: The e2e pipeline test (test_e2e_pipeline.py) already validates the full
rendering pipeline.  These tests focus on the animation-specific helpers and
a single lightweight render to verify keyframe-based rendering works.
"""

import json
import pytest
import numpy as np
from pathlib import Path

from main import run_sweep
from modules.config import SweepConfig


class TestAnimationRenderer:
    """Verify render_animation preserves all data products (single run)."""

    def test_keyframe_pipeline_produces_all_outputs(self):
        """Run the pipeline once and check all expected data products exist.

        The e2e test (test_e2e_pipeline.py) validates the full GT pipeline.
        This test focuses on verifying the keyframe-animation render produces
        the correct image files and imgList.
        """
        config_path = "configs/ci_test.json"
        with open(config_path, "r") as f:
            sweep_json = json.load(f)

        sweep_json["base_config"]["trajectory"]["MIN_F2F_PX_MED"] = 0.0
        # Disable video to avoid path-resolution failures unrelated to rendering
        sweep_json["base_config"]["setup"]["generate_video"] = False
        sweep_config = SweepConfig.model_validate(sweep_json)
        run_sweep(sweep_config)

        # Locate outputs
        renders_dir = Path("renders")
        latest = sorted(
            [d for d in renders_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
        )[-1]

        agent_dirs = list(latest.rglob("Agent_0"))
        assert agent_dirs, "Agent_0 directory not found"
        agent_dir = agent_dirs[0]

        # --- Check rendered images ---
        raw_dir = agent_dir / "images_raw"
        pngs = list(raw_dir.rglob("frame_*.png"))
        expected = len(sweep_json["base_config"].get("frame_ids") or [])
        assert len(pngs) == expected, (
            f"Expected {expected} rendered frames, found {len(pngs)}"
        )

        # --- Check imgList ---
        imglist = agent_dir / "imgList.txt"
        assert imglist.exists(), "imgList.txt not found"

        # --- Image not black ---
        import matplotlib.pyplot as plt
        img = plt.imread(str(pngs[0]))
        assert img.max() > 0.01, "Rendered image is completely black"


class TestContiguousDetection:
    """Unit test for the _are_contiguous helper (no Blender rendering needed)."""

    def test_contiguous_range(self):
        from modules.renderer import BlenderRenderer
        assert BlenderRenderer._are_contiguous([0, 1, 2, 3])

    def test_sparse_ids(self):
        from modules.renderer import BlenderRenderer
        assert not BlenderRenderer._are_contiguous([0, 2, 5])

    def test_single_frame(self):
        from modules.renderer import BlenderRenderer
        assert BlenderRenderer._are_contiguous([7])

    def test_empty(self):
        from modules.renderer import BlenderRenderer
        assert not BlenderRenderer._are_contiguous([])
