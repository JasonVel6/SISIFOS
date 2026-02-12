import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from .vis_utils import _depth_vis_and_mask_from_rrpo, _norm_to_rgb, _flow_to_rgb, _id_to_color

def vprint(msg: str, verbose: bool = True):
    if verbose:
        print(msg)

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_R_RPO(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"R{int(round(value))}"
    # one decimal place, replace '.' with 'p'
    return f"R{str(round(value, 1)).replace('.', 'p')}"

def get_timestamp_folder():
    return datetime.now().strftime("%Y-%m-%d_%H%M")

def handle_gt_from_npz(
    npz_src: Path,
    gt_npz_dir: Path,
    gt_depth_dir: Path,
    gt_norm_dir: Path,
    gt_flow_dir: Path,
    gt_seg_dir: Path,
    R_RPO: float
):
    
    npz_src = Path(npz_src)
    gt_npz_dir = Path(gt_npz_dir)
    gt_depth_dir = Path(gt_depth_dir)
    gt_norm_dir = Path(gt_norm_dir)
    gt_flow_dir = Path(gt_flow_dir)
    gt_seg_dir = Path(gt_seg_dir)

    gt_npz_dir.mkdir(parents=True, exist_ok=True)
    gt_depth_dir.mkdir(parents=True, exist_ok=True)
    gt_norm_dir.mkdir(parents=True, exist_ok=True)
    gt_flow_dir.mkdir(parents=True, exist_ok=True)
    gt_seg_dir.mkdir(parents=True, exist_ok=True)

    # --- move npz into GT NPZ folder ---
    npz_dst = gt_npz_dir / npz_src.name
    if npz_dst.resolve() != npz_src.resolve():
        try:
            npz_src.replace(npz_dst)   # atomic move if possible
        except Exception:
            # fallback: copy then remove
            import shutil
            shutil.copy2(npz_src, npz_dst)
            npz_src.unlink(missing_ok=True)

    base = npz_dst.stem  # e.g. "frame_0001" or "frame_0001_sun_00"

    data = np.load(npz_dst, allow_pickle=True)

    
   
    # --------- DEPTH (masked + colormap) ---------
    if "depth_map" in data:
        d = data["depth_map"].astype(np.float32)
        depth_rgb, near_mask = _depth_vis_and_mask_from_rrpo(
            d, R_RPO,
            cmap_name="viridis",
        )
        plt.imsave(str(gt_depth_dir / f"{base}_Depth.png"), depth_rgb)

        # Save the near-mask too (handy for debugging / training)
        plt.imsave(str(gt_seg_dir / f"{base}_SegDepthGate.png"), near_mask.astype(np.float32), cmap="gray")

    # --------- NORMALS ---------
    if "normal_map" in data:
        n = data["normal_map"].astype(np.float32)
        plt.imsave(str(gt_norm_dir / f"{base}_Normal.png"), _norm_to_rgb(n))

    # --------- OPTICAL FLOW ---------
    if "optical_flow" in data:
        flow = data["optical_flow"].astype(np.float32)
        plt.imsave(str(gt_flow_dir / f"{base}_Flow.png"), _flow_to_rgb(flow))

    # --------- SEGMENTATION (addon-provided) ---------
    if "segmentation_masks" in data:
        seg = data["segmentation_masks"]
        plt.imsave(str(gt_seg_dir / f"{base}_SegMaterial.png"), _id_to_color(seg))
    if "segmentation_masks_collection" in data:
        seg = data["segmentation_masks_collection"]
        plt.imsave(str(gt_seg_dir / f"{base}_SegCollection.png"), _id_to_color(seg))