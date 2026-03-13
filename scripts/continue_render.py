"""
Resume a copied render folder from its last rendered frame.

Run with Blender so the renderer and bpy dependencies are available:
blender -b -P scripts/continue_render.py -- --render_path <path-to-render>
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    _resolve_frame_ids,
    get_render_output_dirs,
    resolve_config_asset_paths,
    run_sisfos_with_config,
)
from modules.config import SceneConfig
from modules.io_utils import create_image_list, get_timestamp_folder, handle_gt_from_npz, images_to_video_blender_sequence
from modules.log_utils import get_logger, setup_logger
from modules.trajectory.trajectory_io import get_scaled_trajectory_in_ECI, make_frames_from_trajectory, read_camera_trajectory

FRAME_FILE_RE = re.compile(r"frame_(\d+)(?:_blurred)?\.png$")
NPZ_FILE_RE = re.compile(r"(\d+)(?:_.*)?\.npz$")
CONFIG_TAG_RE = re.compile(r"^(Config_\d+)")


def summarize_frame_ids(frame_ids: list[int], limit: int = 8) -> str:
    if not frame_ids:
        return "none"

    ranges: list[str] = []
    start = prev = frame_ids[0]
    for fid in frame_ids[1:]:
        if fid == prev + 1:
            prev = fid
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = fid
    ranges.append(f"{start}-{prev}" if start != prev else str(start))

    if len(ranges) <= limit:
        return ", ".join(ranges)
    head = ", ".join(ranges[:limit])
    return f"{head}, ... ({len(frame_ids)} frames total)"


def copy_render_tree(source: Path) -> Path:
    destination = PROJECT_ROOT / "renders" / f"{source.name}_continue_{get_timestamp_folder()}"
    shutil.copytree(source, destination, symlinks=True)
    return destination


def maybe_copy_missing_config(source: Path, copied_root: Path, logger) -> None:
    if any(copied_root.glob("Config_*.json")):
        return

    config_tag = next((match.group(1) for part in source.parts if (match := CONFIG_TAG_RE.match(part))), None)
    if config_tag is None:
        return

    for parent in [source, *source.parents]:
        candidate = parent / f"{config_tag}.json"
        if candidate.exists():
            shutil.copy2(candidate, copied_root / candidate.name)
            logger.info("Copied config into resume folder: %s", candidate)
            return


def find_render_jobs(render_root: Path) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    root_config_files = sorted(render_root.glob("Config_*.json"))

    if (render_root / "camera_traj.csv").exists() and len(root_config_files) == 1:
        return [(root_config_files[0], render_root)]

    for camera_traj in sorted(render_root.rglob("camera_traj.csv")):
        render_dir = camera_traj.parent
        config_tag = next((match.group(1) for part in render_dir.relative_to(render_root).parts if (match := CONFIG_TAG_RE.match(part))), None)
        if config_tag is None and (match := CONFIG_TAG_RE.match(render_dir.name)):
            config_tag = match.group(1)
        if config_tag is None:
            continue

        config_path = render_root / f"{config_tag}.json"
        if config_path.exists():
            jobs.append((config_path, render_dir))

    return jobs


def collect_existing_frame_ids(image_dir: Path) -> list[int]:
    if not image_dir.exists():
        return []

    frame_ids = []
    for path in sorted(image_dir.glob("frame_*.png")):
        match = FRAME_FILE_RE.match(path.name)
        if match:
            frame_ids.append(int(match.group(1)))
    return frame_ids


def collect_npz_frame_ids(npz_dir: Path) -> list[int]:
    if not npz_dir.exists():
        return []

    frame_ids = []
    for path in sorted(npz_dir.glob("*.npz")):
        match = NPZ_FILE_RE.fullmatch(path.name)
        if match:
            frame_ids.append(int(match.group(1)))
    return frame_ids


def get_frame_filename(frame_id: int, num_digits: int, enable_blur: bool) -> str:
    suffix = "_blurred" if enable_blur else ""
    return f"frame_{frame_id:0{num_digits}d}{suffix}.png"


def rebuild_outputs(config: SceneConfig, render_dir: Path, intended_frame_ids: list[int]) -> None:
    logger = get_logger()
    trajectory = read_camera_trajectory(str(render_dir / "camera_traj.csv"))
    timestamps = trajectory["timestamps"]
    scaled_trajectory = get_scaled_trajectory_in_ECI(trajectory, earth_dist_scale_factor=config.render.earth_dist_scale_factor)
    frames = make_frames_from_trajectory(scaled_trajectory)
    image_out_dir, masked_out_dir = get_render_output_dirs(config, render_dir)
    gt_root = render_dir / "GTAnnotations"
    gt_npz_dir = gt_root / "NPZ"
    num_digits = max(4, int(math.log10(len(frames))) + 1)
    enable_blur = str(config.setup.enable_blur).casefold() == "on"

    available_npz_ids = set(collect_npz_frame_ids(gt_npz_dir))

    rebuilt_masked = 0
    for frame_id in intended_frame_ids:
        raw_image_filename = get_frame_filename(frame_id, num_digits, enable_blur)
        raw_image_path = image_out_dir / raw_image_filename
        npz_path = gt_npz_dir / f"{frame_id:04d}.npz"
        if frame_id not in available_npz_ids or not raw_image_path.exists():
            continue

        fr = frames[frame_id]
        target_dist = float(np.linalg.norm(fr["p_G_I"] - fr["p_C_I"]))
        handle_gt_from_npz(
            npz_src=npz_path,
            gt_npz_dir=gt_npz_dir,
            gt_depth_dir=gt_root / "Depth",
            gt_norm_dir=gt_root / "Normal",
            gt_flow_dir=gt_root / "Flow",
            gt_seg_dir=gt_root / "Seg",
            target_dist=target_dist,
            raw_image_filename=raw_image_filename,
            raw_images_dir=str(image_out_dir),
            masked_images_dir=str(masked_out_dir),
        )
        rebuilt_masked += 1

    image_filenames = []
    image_timestamps = []
    for frame_id in intended_frame_ids:
        filename = get_frame_filename(frame_id, num_digits, enable_blur)
        if (masked_out_dir / filename).exists():
            image_filenames.append(filename)
            image_timestamps.append(float(timestamps[frame_id]))

    create_image_list(str(render_dir), image_timestamps, [f"images/{name}" for name in image_filenames])
    logger.info("Rebuilt derived outputs for %d frames and imgList.txt with %d frames", rebuilt_masked, len(image_filenames))

    if config.setup.generate_video:
        images_to_video_blender_sequence(
            image_dir=image_out_dir,
            image_filenames=image_filenames,
            output_path=render_dir / "frames.mp4",
            fps=config.setup.video_fps,
        )


def resume_render_job(config_path: Path, render_dir: Path) -> None:
    logger = get_logger()
    logger.info("Checking render folder: %s", render_dir)
    logger.info("Using config file: %s", config_path)

    with open(config_path) as f:
        config = SceneConfig.model_validate(json.load(f))
    resolve_config_asset_paths(config, PROJECT_ROOT)

    trajectory = read_camera_trajectory(str(render_dir / "camera_traj.csv"))
    intended_frame_ids = _resolve_frame_ids(config, int(trajectory["N"]))
    image_out_dir, masked_out_dir = get_render_output_dirs(config, render_dir)
    gt_npz_dir = render_dir / "GTAnnotations" / "NPZ"
    raw_frame_ids = collect_existing_frame_ids(image_out_dir)
    masked_frame_ids = collect_existing_frame_ids(masked_out_dir)
    npz_frame_ids = collect_npz_frame_ids(gt_npz_dir)

    raw_in_scope = sorted(set(raw_frame_ids).intersection(intended_frame_ids))
    masked_in_scope = sorted(set(masked_frame_ids).intersection(intended_frame_ids))
    npz_in_scope = sorted(set(npz_frame_ids).intersection(intended_frame_ids))
    completed_frame_ids = sorted(set(raw_in_scope).intersection(npz_in_scope))
    completed_frame_id_set = set(completed_frame_ids)
    rerender_frame_ids = [frame_id for frame_id in intended_frame_ids if frame_id not in completed_frame_id_set]
    if completed_frame_ids:
        safety_frame_id = completed_frame_ids[-1]
        if safety_frame_id not in rerender_frame_ids:
            rerender_frame_ids.insert(0, safety_frame_id)

    logger.info(
        "Checked outputs in %s. Intended=%d [%s]. Raw=%d [%s]. Masked=%d [%s]. NPZ=%d [%s]. Complete(raw+npz)=%d [%s].",
        image_out_dir,
        len(intended_frame_ids),
        summarize_frame_ids(intended_frame_ids),
        len(raw_in_scope),
        summarize_frame_ids(raw_in_scope),
        len(masked_in_scope),
        summarize_frame_ids(masked_in_scope),
        len(npz_in_scope),
        summarize_frame_ids(npz_in_scope),
        len(completed_frame_ids),
        summarize_frame_ids(completed_frame_ids),
    )
    logger.info(
        "Re-rendering %d frames including safety replay of the latest complete frame [%s].",
        len(rerender_frame_ids),
        summarize_frame_ids(rerender_frame_ids),
    )

    if not rerender_frame_ids:
        logger.info("No frames selected for rerender in %s", render_dir)
        rebuild_outputs(config, render_dir, intended_frame_ids)
        return

    config.frame_ids = rerender_frame_ids
    config.from_frame_id = None
    should_generate_video = config.setup.generate_video
    config.setup.generate_video = False
    try:
        run_sisfos_with_config(config, render_dir)
    finally:
        config.setup.generate_video = should_generate_video
    rebuild_outputs(config, render_dir, intended_frame_ids)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue a SISIFOS render from the last rendered frame.")
    parser.add_argument("--render_path", required=True, help="Path to the render folder to copy and continue.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        raise RuntimeError("Pass script arguments after '--', e.g. --render_path renders/<folder>")

    args = parse_args(argv)
    source = Path(args.render_path).resolve()
    if not source.exists() or not source.is_dir():
        raise ValueError(f"Render path does not exist or is not a directory: {source}")

    copied_root = copy_render_tree(source)
    setup_logger(log_file=copied_root / "continue_render.log")
    logger = get_logger()
    logger.info("Copied render tree from %s to %s", source, copied_root)

    maybe_copy_missing_config(source, copied_root, logger)

    jobs = find_render_jobs(copied_root)

    if not jobs:
        raise RuntimeError(f"No render jobs with camera_traj.csv and matching Config_*.json were found in {copied_root}")

    logger.info("Found %d render job(s) to continue", len(jobs))
    for config_path, render_dir in jobs:
        resume_render_job(config_path, render_dir)


if __name__ == "__main__":
    main(sys.argv)
