import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import bpy
import matplotlib.pyplot as plt
import numpy as np

from .log_utils import get_logger
from .vis_utils import _depth_vis_and_mask_from_rrpo, _flow_to_rgb, _id_to_color, _norm_to_rgb

logger = get_logger()


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
    masked_images_dir: str,
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
            npz_src.replace(npz_dst)  # atomic move if possible
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

    mask = None

    # --------- SEGMENTATION (addon-provided) ---------
    if "segmentation_masks" in data:
        seg = data["segmentation_masks"]
        plt.imsave(str(gt_seg_dir / f"{base}_Seg.png"), _id_to_color(seg))
        # vision_blender segments by per-object/material index, NOT by the
        # pass_index the renderer assigns to "Target" (the spacecraft surfaces
        # carry their material indices, e.g. {2,3,6,7,8,...}; the bare "Target"
        # empty has no pixels). So `seg == 1` masked out everything. Foreground
        # = any non-background index; with earth_mode=off the only background is
        # index 0 (empty space + camera/sun/light), so `seg != 0` is the clean
        # target silhouette (verified: drops the bg noise floor, keeps shadowed
        # target pixels). NOTE: relies on Earth/Clouds/Atmo not being rendered.
        mask = seg != 0

    # Create masked images. Three things can go wrong here that we need to
    # tolerate gracefully:
    #   1) Neither segmentation nor depth was saved -> no mask available.
    #   2) Render border crop is on -> rendered_img is the cropped frame but
    #      vision_blender's depth_map / segmentation_masks come back at the
    #      full-frame resolution. We can't index a small image with a big mask.
    # In either case we just copy the raw render through unmasked rather than
    # crashing - the cropped onboard-style frames don't need masking anyway.
    ensure_dir(Path(masked_images_dir))
    rendered_img_path = os.path.join(raw_images_dir, raw_image_filename)
    rendered_img = plt.imread(rendered_img_path)
    if mask is None and "depth_map" in data:
        mask = near_mask
    img_h, img_w = rendered_img.shape[:2]
    if mask is None or mask.shape[:2] != (img_h, img_w):
        masked_img = rendered_img
    else:
        masked_img = np.zeros_like(rendered_img)
        masked_img[mask] = rendered_img[mask]
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
    logger.info("  Created: %s", imglist_path)
    return imglist_path


def images_to_video_blender_sequence(
    image_dir: str | Path,
    image_filenames: list[str],
    output_path: str | Path,
    fps: int = 24,
) -> str:
    """Assemble a video from pre-rendered frames via ffmpeg.

    Uses a concat-demuxer list so the input ordering is exactly the supplied
    image_filenames sequence (handles non-contiguous frame ranges). The pad
    filter rounds width/height up to the next even number, which is required
    by H.264 and works around the Blender render-border off-by-one that
    occasionally leaves an odd-width crop (e.g. 255x256 instead of 256x256).

    Args:
        image_dir: Directory containing rendered frames.
        image_filenames: Ordered list of image filenames to include.
        output_path: Target .mp4 filepath.
        fps: Output frames per second.
    """
    if not image_filenames:
        raise ValueError("Cannot generate video: no image filenames provided.")

    image_dir = Path(image_dir).resolve()
    output_path = Path(output_path).resolve()

    frames = []
    for name in image_filenames:
        if (image_dir / name).exists():
            frames.append(name)
        else:
            logger.warning("Skipping missing frame in video assembly: %s", image_dir / name)

    if not frames:
        raise ValueError("Cannot generate video: no existing frames found in image_dir.")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH; cannot assemble video.")

    # ffmpeg's concat demuxer takes a plain text list of `file '<name>'` /
    # `duration <seconds>` pairs. The last entry must be repeated without a
    # duration (a documented quirk that ensures the final frame is encoded).
    list_path = image_dir / ".video_concat_list.txt"
    duration = 1.0 / float(fps)
    with open(list_path, "w") as f:
        for name in frames:
            f.write(f"file '{name}'\n")
            f.write(f"duration {duration}\n")
        f.write(f"file '{frames[-1]}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-vsync", "vfr",
        # pad: round W/H up to next multiple of 2; format=yuv420p for broad H.264 compat
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-r", str(int(fps)),
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    finally:
        list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg video assembly failed (returncode={result.returncode}): {result.stderr.strip()}"
        )

    logger.info(
        "Video generated successfully via ffmpeg (auto-padded to even dims): %s",
        output_path,
    )
    return str(output_path)
