import json
import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from main import run_sweep
from modules.config import SweepConfig

class TestE2EPipeline:
    def test_full_pipeline_image_and_data_integrity(self):
        config_path = "configs/ci_test.json"
        
        with open(config_path, 'r') as f:
            sweep_config_json = json.load(f)
            
        # Bypass degeneracy checks for low-resolution CI tests
        sweep_config_json["base_config"]["trajectory"]["MIN_F2F_PX_MED"] = 0.0
        sweep_config = SweepConfig.model_validate(sweep_config_json)
        
        # Execute the pipeline
        run_sweep(sweep_config)
        
        # Locate outputs
        renders_dir = Path("renders")
        latest_run = sorted([d for d in renders_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)[-1]
        
        agent_dirs = list(latest_run.rglob("Agent_0"))
        assert len(agent_dirs) > 0, "Could not find Agent_0 output directory."
        agent_dir = agent_dirs[0]
        raw_images_dir = agent_dir / "images_raw" / "Stars_OFF"
        
        # 1. Analyze the Rendered Images
        img_paths = list(raw_images_dir.glob("frame_*.png"))
        assert len(img_paths) == 4, f"Expected 4 rendered frames, found {len(img_paths)}."
        
        target_found, earth_found = False, False
        for img_path in img_paths:
            img = plt.imread(str(img_path))
            if np.sum(img[:, :, 0] > 0.8) > 0: target_found = True  # Red
            if np.sum(img[:, :, 2] > 0.8) > 0: earth_found = True   # Blue
            
        assert target_found, "Target (red pixels) not found in any frame."
        assert earth_found, "Earth (blue pixels) not found in any frame. Orbit scaling or clipping failed."
        
        # 2. Analyze the Ground Truth NPZ on the first frame
        npz_dir = agent_dir / "GTAnnotations" / "NPZ"
        npz_path = list(npz_dir.glob("*.npz"))[0]
        gt_data = np.load(npz_path, allow_pickle=True)
        depth = gt_data["depth_map"]
        seg = gt_data["segmentation_masks"]
        
        # Validate Target scaling
        target_mask = (seg == 1)
        if np.any(target_mask):
            mean_depth = np.mean(depth[target_mask])
            assert 10.0 < mean_depth < 100.0, f"Expected target depth in [10,100]m, got {mean_depth:.2f}m."