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
    target_dist: float
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

def prepare_slam_dataset(traj_root, traj_path, timestamps, num_frames):
    """
    Prepare the rendered output for SLAM pipeline consumption.

    This function performs the post-processing that was previously done by run_ue5.py:
    1. Creates imgList.txt with timestamp-image pairs
    2. Copies gtValues.txt and other source files to the render output

    Note: imgList.txt always references images/ folder, which contains:
    - Raw renders (when masking disabled)
    - Masked images with Earth/stars removed (when masking enabled)
    When masking is enabled, raw renders are in images_raw/.

    Parameters:
    -----------
    traj_root : str
        Root directory of the rendered output (e.g., renders/1203_Tumbling_mc0_cro_agent0_2024_...)
    traj_path : str
        Source trajectory folder path (contains gtValues.txt, Config.yaml, etc.)
    timestamps : array-like
        Timestamps for each frame
    num_frames : int
        Number of frames rendered
    """
    print("\n" + "="*60)
    print("PREPARING SLAM DATASET")
    print("="*60)

    images_dir = os.path.join(traj_root, "images")

    # Create imgList.txt
    imglist_path = os.path.join(traj_root, "imgList.txt")
    with open(imglist_path, "w") as f:
        for i in range(num_frames):
            ts = timestamps[i] if i < len(timestamps) else i * 1.0
            f.write(f"{ts:.6f} images/img_{i:05d}.png\n")
    print(f"  Created: {imglist_path}")

    # Copy source files from trajectory folder to render output
    files_to_copy = [
        "gtValues.txt",
        "Config.yaml",
        "sensormeasurements.txt",
        "camera_traj.csv",
    ]

    for filename in files_to_copy:
        src = os.path.join(traj_path, filename)
        dst = os.path.join(traj_root, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")
        else:
            print(f"  [WARN] Not found: {filename}")

    print("\n  SLAM dataset preparation complete!")
    print(f"  Output directory ready for SLAM: {traj_root}")
    print("="*60 + "\n")

    return images_dir, imglist_path

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
