"""
Resume a copied render folder from its last rendered frame.

Run with Blender so the renderer and bpy dependencies are available:
blender -b -P scripts/continue_render.py -- --render_path <path-to-render>
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from main import (
    _resolve_frame_ids,
    get_render_output_dirs,
    resolve_config_asset_paths,
    run_sisfos_with_config,
)
from modules.config import SceneConfig
from modules.io_utils import create_image_list, get_timestamp_folder, images_to_video_blender_sequence
from modules.log_utils import get_logger, setup_logger
from modules.trajectory.trajectory_io import read_camera_trajectory

FRAME_FILE_RE = re.compile(r"frame_(\d+)\.png$")
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


def rebuild_outputs(config: SceneConfig, render_dir: Path, intended_frame_ids: list[int]) -> None:
    logger = get_logger()
    trajectory = read_camera_trajectory(str(render_dir / "camera_traj.csv"))
    image_out_dir, _masked_out_dir = get_render_output_dirs(config, render_dir)
    num_digits = max(4, len(str(int(trajectory["N"]) - 1)))

    image_filenames = []
    timestamps = []
    for frame_id in intended_frame_ids:
        filename = f"frame_{frame_id:0{num_digits}d}.png"
        if (image_out_dir / filename).exists():
            image_filenames.append(filename)
            timestamps.append(float(trajectory["timestamps"][frame_id]))

    create_image_list(str(render_dir), timestamps, [f"images/{name}" for name in image_filenames])
    logger.info("Rebuilt imgList.txt with %d frames", len(image_filenames))

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
    image_out_dir, _masked_out_dir = get_render_output_dirs(config, render_dir)
    existing_frame_ids = collect_existing_frame_ids(image_out_dir)

    rendered_in_scope = sorted(set(existing_frame_ids).intersection(intended_frame_ids))
    if rendered_in_scope:
        resume_from = rendered_in_scope[-1]
        rerender_frame_ids = [frame_id for frame_id in intended_frame_ids if frame_id >= resume_from]
    else:
        resume_from = None
        rerender_frame_ids = intended_frame_ids

    logger.info(
        "Checked %d existing frame files in %s. Intended=%d [%s]. Existing in scope=%d [%s].",
        len(existing_frame_ids),
        image_out_dir,
        len(intended_frame_ids),
        summarize_frame_ids(intended_frame_ids),
        len(rendered_in_scope),
        summarize_frame_ids(rendered_in_scope),
    )
    logger.info(
        "Resume start: %s. Re-rendering %d frames [%s].",
        "none found" if resume_from is None else resume_from,
        len(rerender_frame_ids),
        summarize_frame_ids(rerender_frame_ids),
    )

    if not rerender_frame_ids:
        logger.info("No frames selected for rerender in %s", render_dir)
        rebuild_outputs(config, render_dir, intended_frame_ids)
        return

    config.frame_ids = rerender_frame_ids
    config.from_frame_id = None
    config.setup.generate_video = False
    run_sisfos_with_config(config, render_dir)
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
