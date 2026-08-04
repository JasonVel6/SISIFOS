import matplotlib.pyplot as plt
import numpy as np

from modules.io_utils import ensure_dir, handle_gt_from_npz


def _make_synthetic_npz(path, res=32, include_depth=True):
    data = {}
    if include_depth:
        data["depth_map"] = np.full((res, res), 25.0, dtype=np.float32)
    np.savez(path, **data)


class TestHandleGtPipeline:
    def test_no_depth_map_handles_gracefully_regression(self, tmp_path):
        """With neither segmentation nor depth in the npz, no mask can be derived.

        The documented behaviour is to pass the raw render through unmasked rather
        than crash (see the comment in handle_gt_from_npz): the cropped onboard-style
        frames legitimately arrive without a usable mask.
        """
        raw_dir, masked_dir = ensure_dir(tmp_path / "images_raw"), tmp_path / "images"
        gt_dirs = {k: ensure_dir(tmp_path / "GT" / k) for k in ["NPZ", "Depth", "Normal", "Flow", "Seg"]}
        npz_path = raw_dir / "0001.npz"
        _make_synthetic_npz(npz_path, include_depth=False)
        img_filename = "frame_0001.png"
        plt.imsave(str(raw_dir / img_filename), np.random.rand(32, 32, 4).astype(np.float32))
        raw_img = plt.imread(str(raw_dir / img_filename))

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

        # The npz is moved into the GT folder regardless of its contents.
        assert (gt_dirs["NPZ"] / "0001.npz").exists()
        assert not npz_path.exists()

        # The masked image is written, and is the raw render passed through.
        masked_path = masked_dir / img_filename
        assert masked_path.exists()
        np.testing.assert_allclose(plt.imread(str(masked_path)), raw_img, atol=1.0 / 255.0)

        # No depth/normal/flow/seg products are derivable from an empty npz.
        for key in ["Depth", "Normal", "Flow", "Seg"]:
            assert list(gt_dirs[key].iterdir()) == []
