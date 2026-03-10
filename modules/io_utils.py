import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from .vis_utils import _depth_vis_and_mask_from_rrpo, _norm_to_rgb, _flow_to_rgb, _id_to_color
import os
import shutil
import bpy

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

def images_to_video_blender_sequence(
    image_dir: str | Path,
    image_filenames: list[str],
    output_path: str | Path,
    fps: int = 24,
) -> str:
    """
    Assemble a video from pre-rendered frames using Blender's sequence editor.

    Args:
        image_dir: Directory containing rendered frames.
        image_filenames: Ordered list of image filenames to include.
        output_path: Target .mp4 filepath.
        fps: Output frames per second.
    """
    if not image_filenames:
        raise ValueError("Cannot generate video: no image filenames provided.")

    image_dir = Path(image_dir)
    output_path = Path(output_path)
    abs_output = Path(output_path).resolve()

    abs_dir = Path(image_dir).resolve()
    frames = []
    for name in image_filenames:
        p = abs_dir / name
        if p.exists():
            frames.append({"name": p.name})
        else:
            print(f"Skipping missing frame in video assembly: {p}")

    if not frames:
        raise ValueError("Cannot generate video: no existing frames found in image_dir.")

    render_scene = bpy.data.scenes.new(name="SISIFOS_VideoAssembly")
    try:
        render_scene.sequence_editor_create()
        seq = render_scene.sequence_editor
        first_frame_path = abs_dir / frames[0]["name"]

        seq.sequences.new_image(
            name="RenderFrames",
            filepath=str(first_frame_path),
            channel=1,
            frame_start=1,
        )
        image_strip = seq.sequences_all["RenderFrames"]
        # new_image already creates the first element, so append the rest.
        for frame in frames[1:]:
            image_strip.elements.append(frame["name"])

        # Match output dimensions to source frames to avoid stretching/cropping.
        first_img = bpy.data.images.load(str(first_frame_path), check_existing=True)
        src_w, src_h = int(first_img.size[0]), int(first_img.size[1])

        render_scene.frame_start = 1
        render_scene.frame_end = len(frames)
        render_scene.render.use_sequencer = True
        render_scene.render.resolution_x = src_w
        render_scene.render.resolution_y = src_h
        bpy.data.images.remove(first_img)  # cleanup loaded image to avoid memory bloat
        render_scene.render.resolution_percentage = 100
        render_scene.render.pixel_aspect_x = 1.0
        render_scene.render.pixel_aspect_y = 1.0
        render_scene.render.fps = int(fps)
        render_scene.render.fps_base = 1.0
        render_scene.render.image_settings.file_format = "FFMPEG"
        render_scene.render.ffmpeg.format = "MPEG4"
        render_scene.render.ffmpeg.codec = "H264"
        render_scene.render.ffmpeg.constant_rate_factor = "HIGH"
        render_scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        render_scene.render.ffmpeg.gopsize = 12
        render_scene.render.ffmpeg.audio_codec = "NONE"
        render_scene.render.filepath = str(abs_output)

        bpy.ops.render.render(animation=True, scene=render_scene.name)
        print(f"Video generated successfully: {abs_output}")
        return str(abs_output)
    finally:
        bpy.data.scenes.remove(render_scene)
