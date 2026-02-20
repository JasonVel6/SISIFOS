import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from .vis_utils import _depth_vis_and_mask_from_rrpo, _norm_to_rgb, _flow_to_rgb, _id_to_color
import os
import shutil
import bpy
import subprocess

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
    target_dist: float,
    raw_image_filename: str,
    raw_images_dir: str,
    masked_images_dir: str
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
        depth_rgb, near_mask = _depth_vis_and_mask_from_rrpo(d, target_dist=target_dist, cmap_name="viridis")
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
        plt.imsave(str(gt_seg_dir / f"{base}_Seg.png"), _id_to_color(seg))

    # Create masked images
    ensure_dir(Path(masked_images_dir))
    rendered_img_path = os.path.join(raw_images_dir, raw_image_filename)
    rendered_img = plt.imread(rendered_img_path)
    masked_img = np.zeros_like(rendered_img)
    masked_img[near_mask] = rendered_img[near_mask]
    masked_img_path = os.path.join(masked_images_dir, raw_image_filename)
    plt.imsave(masked_img_path, masked_img)

def create_image_list(renders_base_dir: str, timestamps: list, image_paths):
    """
    Create imgList.txt with timestamp-image pairs.
    """
    imglist_path = os.path.join(renders_base_dir, "imgList.txt")
    with open(imglist_path, "w") as f:
        for i in range(len(timestamps)):
            ts = timestamps[i]
            f.write(f"{ts:.6f} {image_paths[i]}\n")
    print(f"  Created: {imglist_path}")
    return imglist_path

def images_to_video_ffmpeg(
    input_pattern,
    output_path,
    fps=24,
    crf=18,
    overwrite=True
):
    """
    Convert rendered images to a video using FFmpeg.

    input_pattern: e.g. "//renders/INE_24_11_2025/frame_%04d.png"
    output_path:   e.g. "//renders/INE_24_11_2025/output.mp4"
    fps: frames per second
    crf: quality (lower = better, default 18)
    overwrite: allow overwriting existing file
    """

    # Expand Blender paths ("//")
    abs_input = bpy.path.abspath(input_pattern)
    abs_output = bpy.path.abspath(output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    # -y means overwrite
    overwrite_flag = "-y" if overwrite else "-n"

    cmd = [
        "ffmpeg",
        overwrite_flag,
        "-framerate", str(fps),
        "-i", abs_input,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",   # safest for compatibility
        "-crf", str(crf),
        abs_output
    ]

    print("Running FFmpeg command:\n", " ".join(cmd))

    # Execute FFmpeg
    try:
        subprocess.run(cmd, check=True)
        print("Video generated successfully:", abs_output)
    except subprocess.CalledProcessError as e:
        print("Error during video generation:", e)
