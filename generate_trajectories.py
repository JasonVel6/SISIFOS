"""
© Dynamics and Control Systems Laboratory, Georgia Institute of Technology
SISIFOS: Specialized Illumination SImulator For Orbiting Spacecraft

Blender-free trajectory generation.

This is `main.py` with the renderer removed: it consumes the same config files
and produces the same trajectory/ground-truth artifacts, but never imports
`bpy`, so it runs under a plain Python interpreter instead of `blender -b -P`.

Use it when you need the trajectory + ground truth (SLAM ingestion, sweep
design, observability studies) and not the images. When you also need images,
use `main.py` under Blender — this script always skips rendering, and warns
when it is handed a config with `setup.render_frames: true`.

Requirements: the trajectory modules import Blender's `mathutils`, which is on
PyPI as a standalone C extension. Latest (5.x) needs Python >= 3.13; on 3.10 /
3.11 pin 3.3.0:

    pip install "mathutils==3.3.0" numpy scipy pydantic matplotlib pyyaml

Usage (from the repo root, same flags as main.py):

    python generate_trajectories.py --sweep_config_path configs/probe_hybrid.json
    python generate_trajectories.py --config_path renders/<ts>/Config_1.json

Also accepts a bare TrajectoryConfig JSON (the `Config_<i>_trajectory.json` this
script writes). That shape carries no camera block, and camera intrinsics feed
the initial-condition sampler, so pass `--camera_from` to borrow a real camera —
otherwise CameraConfig defaults silently change the trajectory for a given seed.

    python generate_trajectories.py --config_path out/Config_1_trajectory.json \
        --camera_from configs/probe_hybrid.json

Output layout matches main.py: `renders/<timestamp>/` (or $SISIFOS_OUTPUT_DIR,
or --output_dir) containing sweep_configs.json, Config_<i>.json, and per-agent
folders with camera_traj.csv, gtValues.txt, sensormeasurements.txt, Config.yaml
and imgList.txt.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Import `modules` relative to this file so the script runs from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.config import SceneConfig, SweepConfig
from modules.log_utils import get_logger, setup_logger
from modules.path_utils import create_image_list, ensure_dir, get_timestamp_folder
from modules.trajectory.generateTrajectoriesUnified import generate_trajectories_dynamical
from modules.trajectory.sampling_trajectory import (
    write_camera_trajectory_const_rotation,
    write_camera_trajectory_fib,
)
from modules.trajectory.trajectory_io import read_camera_trajectory


def _sanitize_folder_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in value)
    return token.strip("_") or "Unknown"


def load_scene_configs(config_path: str) -> list[SceneConfig]:
    """Expand any of the three on-disk config shapes into SceneConfigs.

    1. sweep       {"base_config": ..., "sweep_parameters": ...}  -> N configs
    2. SceneConfig flat, e.g. renders/<ts>/Config_1.json          -> 1 config
    3. TrajectoryConfig bare, e.g. Config_1_trajectory.json       -> 1 config,
       wrapped with SceneConfig defaults (camera included — see --camera_from).
    """
    raw = json.load(open(config_path))

    if "base_config" in raw:
        return SweepConfig.model_validate(raw).generate_sweep_configs()
    if "trajectory" in raw:
        return [SceneConfig.model_validate(raw)]
    return [SceneConfig.model_validate({"trajectory": raw})]


def generate_trajectories(config: SceneConfig, output_dir: Path, config_prefix: str) -> list[str]:
    """Mirror of main.py's generate_trajectories(); none of its branches need bpy."""
    model_token = _sanitize_folder_token(config.selected_model)

    if config.trajectory_type == "trajectory_generator":
        return generate_trajectories_dynamical(
            config.trajectory,
            str(output_dir),
            config_prefix=config_prefix,
            model_name=model_token,
            camera_config=config.camera,
            save_scene_plots=config.save_scene_plots,
            scene_plot_max_frames=config.scene_plot_max_frames,
        )

    if config.trajectory_type == "sampling_trajectory":
        agent_folder = ensure_dir(output_dir / f"{config_prefix}_{model_token}")
        return write_camera_trajectory_fib(
            str(agent_folder),
            N=config.trajectory_sampling.num_frames,
            R_LEO=config.trajectory_sampling.R_LEO,
            R_RPO=config.trajectory_sampling.R_RPO,
            sun_az=config.trajectory_sampling.sun_az,
            sun_el=config.trajectory_sampling.sun_el,
            fixed_q_IG_wxyz=config.trajectory_sampling.fixed_q_IG_wxyz,
        )

    if config.trajectory_type == "const_rotate":
        agent_folder = ensure_dir(output_dir / f"{config_prefix}_{model_token}")
        return write_camera_trajectory_const_rotation(
            str(agent_folder),
            R_LEO=config.trajectory_const_rotate.R_LEO,
            R_RPO=config.trajectory_const_rotate.R_RPO,
            tstep=config.trajectory_const_rotate.tstep,
            tend=config.trajectory_const_rotate.tend,
            angular_velocity=config.trajectory_const_rotate.angular_velocity,
            sun_az=config.trajectory_const_rotate.sun_az,
            sun_el=config.trajectory_const_rotate.sun_el,
        )

    if config.trajectory_type == "filepath":
        if not config.trajectory_filepath:
            raise ValueError(
                "Trajectory type is set to 'filepath' but no trajectory_filepath is provided in the config."
            )

        src_folder = Path(config.trajectory_filepath)
        if not src_folder.exists() or not src_folder.is_dir():
            raise ValueError(
                f"Provided trajectory_filepath '{config.trajectory_filepath}' does not exist or is not a directory."
            )

        dest_folder = ensure_dir(output_dir / f"{config_prefix}_{model_token}")
        for item in src_folder.iterdir():
            dest = dest_folder / item.name
            if not dest.exists():
                os.symlink(item.resolve(), dest)  # symlink to avoid copying large files/folders
        return [str(dest_folder)]

    raise ValueError(
        f"Invalid trajectory type: {config.trajectory_type}. Must be 'trajectory_generator', "
        "'sampling_trajectory', 'const_rotate' or 'filepath'."
    )


def prepare_image_list(config: SceneConfig, renders_base_dir: Path) -> None:
    """Emit imgList.txt from trajectory timestamps and the filenames a render would produce.

    Identical to main.py's prepare_image_list, so a dataset generated here is
    complete for downstream tooling even though no images exist yet.
    """
    logger = get_logger()
    trajectory = read_camera_trajectory(str(renders_base_dir / "camera_traj.csv"))
    n_frames = int(trajectory["N"])
    frame_ids = config.frame_ids if config.frame_ids else list(range(n_frames))
    n_digits = max(4, int(math.log10(n_frames)) + 1)

    blur_enabled = str(config.setup.enable_blur).casefold() == "on"
    image_filenames = []
    for i in frame_ids:
        stem = f"{str(i).zfill(n_digits)}"
        if blur_enabled:
            stem = f"{stem}_blurred"
        image_filenames.append(f"frame_{stem}.png")

    timestamps = [float(trajectory["timestamps"][fid]) for fid in frame_ids]
    image_paths = [os.path.join("images", image_filename) for image_filename in image_filenames]
    create_image_list(str(renders_base_dir), timestamps, image_paths)
    logger.info("Prepared dataset bookkeeping for: %s", renders_base_dir)


def run_sweep(
    configs: list[SceneConfig],
    output_dir: Path,
    source_config_path: str,
    camera_from: str | None = None,
) -> list[str]:
    """Mirror of main.py's run_sweep(), minus the render call."""
    ensure_dir(output_dir)
    setup_logger(log_file=output_dir / "run.log")
    logger = get_logger()
    logger.info("Generating trajectories for %d configuration(s). Output base dir: %s", len(configs), output_dir)
    logger.info("Source config: %s", source_config_path)
    if camera_from:
        cam = configs[0].camera
        logger.info(
            "[CAMERA] borrowed from %s: f=%.1f mm, res=%s, f_px=%.1f",
            camera_from,
            cam.focal_length,
            tuple(cam.resolution),
            cam.focal_length_px,
        )

    # Preserve the source config verbatim (including sweep_parameters, which the
    # expanded per-config dumps below no longer carry).
    with open(output_dir / "sweep_configs.json", "w") as f:
        json.dump(json.load(open(source_config_path)), f, indent=2)

    all_agent_folders = []
    for i, config in enumerate(configs):
        config_prefix = f"Config_{i + 1}"

        with open(output_dir / f"{config_prefix}.json", "w") as f:
            json.dump(config.model_dump(), f, indent=2)

        if config.setup.render_frames:
            logger.warning(
                "%s has setup.render_frames=true, but this script never renders. "
                "Trajectory and GT files will be written; run main.py under Blender for images.",
                config_prefix,
            )

        agent_folders = generate_trajectories(config, output_dir, config_prefix=config_prefix)
        for agent_folder in agent_folders:
            prepare_image_list(config, Path(agent_folder))
            logger.info("[OK] %s -> %s", config_prefix, agent_folder)
            all_agent_folders.append(agent_folder)

    return all_agent_folders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SISIFOS trajectory generation without Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config_path", type=str, help="Path to a SceneConfig or TrajectoryConfig JSON")
    parser.add_argument("--sweep_config_path", type=str, help="Path to a sweep config JSON")
    parser.add_argument(
        "--camera_from",
        type=str,
        help="Borrow the camera block from another config. Use with a bare TrajectoryConfig, "
        "whose defaults would otherwise change the sampled trajectory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output base dir. Defaults to $SISIFOS_OUTPUT_DIR, else renders/<timestamp>.",
    )

    # Tolerate the `--` separator so main.py's Blender-style invocation still parses.
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    if args.config_path and args.sweep_config_path:
        raise RuntimeError(
            "Cannot specify --sweep_config_path together with --config_path. Please provide only one of these options."
        )
    config_path = args.config_path or args.sweep_config_path
    if not config_path:
        raise RuntimeError("No configuration file specified. Pass --config_path or --sweep_config_path.")

    configs = load_scene_configs(config_path)

    if args.camera_from:
        donor = load_scene_configs(args.camera_from)[0]
        for config in configs:
            config.camera = donor.camera

    _fixed = args.output_dir or os.environ.get("SISIFOS_OUTPUT_DIR")
    output_dir = Path(_fixed) if _fixed else Path("./renders") / get_timestamp_folder()

    agent_folders = run_sweep(configs, output_dir, config_path, camera_from=args.camera_from)

    # Deliberately not "[DONE] Output written to:" — the generator already logs
    # that per config, and tooling greps for it (configs/synthetic-exp/_generate-datasets.sh).
    get_logger().info("[COMPLETE] %d agent folder(s) under: %s", len(agent_folders), output_dir)


if __name__ == "__main__":
    main()
