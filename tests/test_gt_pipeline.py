import matplotlib.pyplot as plt
import numpy as np
import pytest

from modules.io_utils import ensure_dir, handle_gt_from_npz


def _make_synthetic_npz(path, res=32, include_depth=True):
    data = {}
    if include_depth:
        data["depth_map"] = np.full((res, res), 25.0, dtype=np.float32)
    np.savez(path, **data)


class TestHandleGtPipeline:
    def test_no_depth_map_handles_gracefully_regression(self, tmp_path):
        raw_dir, masked_dir = ensure_dir(tmp_path / "images_raw"), tmp_path / "images"
        gt_dirs = {k: ensure_dir(tmp_path / "GT" / k) for k in ["NPZ", "Depth", "Normal", "Flow", "Seg"]}
        npz_path = raw_dir / "0001.npz"
        _make_synthetic_npz(npz_path, include_depth=False)
        img_filename = "frame_0001.png"
        plt.imsave(str(raw_dir / img_filename), np.random.rand(32, 32, 4).astype(np.float32))

        with pytest.raises((NameError, UnboundLocalError, KeyError)):
            handle_gt_from_npz(
                npz_src=npz_path,
                gt_npz_dir=gt_dirs["NPZ"],
                gt_depth_dir=gt_dirs["Depth"],
                gt_norm_dir=gt_dirs["Normal"],
                gt_flow_dir=gt_dirs["Flow"],
                gt_seg_dir=gt_dirs["Seg"],
                target_dist=60.0,
                raw_image_filename=img_filename,
                raw_images_dir=str(raw_dir),
                masked_images_dir=str(masked_dir),
            )
