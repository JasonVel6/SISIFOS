"""
© Dynamics and Control Systems Laboratory, Georgia Institute of Technology
SISIFOS: Specialized Illumination SImulator For Orbiting Spacecraft

This is the main file, setting up the simulation, loading the configuration
and actually rendering the data (images and annotations).

Iason Georgios Velentzas (ivelentzas3@gatech.edu)

"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from modules.config import SceneConfig, SweepConfig
from modules.io_utils import (
    create_image_list,
    ensure_dir,
    get_timestamp_folder,
    handle_gt_from_npz,
    images_to_video_blender_sequence,
)
from modules.log_utils import get_logger, setup_logger
from modules.renderer import BlenderRenderer
from modules.trajectory.generateTrajectoriesUnified import generate_trajectories_dynamical
from modules.trajectory.sampling_trajectory import (
    make_fake_frame_from_frame0,
    write_camera_trajectory_const_rotation,
    write_camera_trajectory_fib,
)
from modules.trajectory.trajectory_io import (
    get_scaled_trajectory_in_ECI,
    make_frames_from_trajectory,
    read_camera_trajectory,
)

DEFAULT_CONFIG_PATH = "/config/config_example_basic.json"


def _sanitize_folder_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in value)
    return token.strip("_") or "Unknown"


def generate_trajectories(config: SceneConfig, output_dir: Path, config_prefix: str) -> list[str]:
    model_token = _sanitize_folder_token(config.selected_model)

    if config.trajectory_type == "trajectory_generator":
        agent_folders = generate_trajectories_dynamical(
            config.trajectory,
            str(output_dir),
            config_prefix=config_prefix,
            model_name=model_token,
            camera_config=config.camera,
            save_scene_plots=config.save_scene_plots,
            scene_plot_max_frames=config.scene_plot_max_frames,
        )

    elif config.trajectory_type == "sampling_trajectory":
        agent_folder = ensure_dir(output_dir / f"{config_prefix}_{model_token}")
        agent_folders = write_camera_trajectory_fib(
            str(agent_folder),
            N=config.trajectory_sampling.num_frames,
            R_LEO=config.trajectory_sampling.R_LEO,
            R_RPO=config.trajectory_sampling.R_RPO,
            sun_az=config.trajectory_sampling.sun_az,
            sun_el=config.trajectory_sampling.sun_el,
        )

    elif config.trajectory_type == "const_rotate":
        agent_folder = ensure_dir(output_dir / f"{config_prefix}_{model_token}")
        agent_folders = write_camera_trajectory_const_rotation(
            str(agent_folder),
            R_LEO=config.trajectory_const_rotate.R_LEO,
            R_RPO=config.trajectory_const_rotate.R_RPO,
            tstep=config.trajectory_const_rotate.tstep,
            tend=config.trajectory_const_rotate.tend,
            angular_velocity=config.trajectory_const_rotate.angular_velocity,
            sun_az=config.trajectory_const_rotate.sun_az,
            sun_el=config.trajectory_const_rotate.sun_el,
        )

    elif config.trajectory_type == "filepath":
        if not config.trajectory_filepath:
            raise ValueError(
                "Trajectory type is set to 'filepath' but no trajectory_filepath is provided in the config."
            )

        # Copy the contents of the provided folder to the output directory to ensure all outputs are organized under the same base output folder
        src_folder = Path(config.trajectory_filepath)
        if not src_folder.exists() or not src_folder.is_dir():
            raise ValueError(
                f"Provided trajectory_filepath '{config.trajectory_filepath}' does not exist or is not a directory."
            )

        dest_folder = ensure_dir(output_dir / f"{config_prefix}_{model_token}")
        ensure_dir(dest_folder)
        for item in src_folder.iterdir():
            if item.is_file():
                dest = dest_folder / item.name
                if not dest.exists():
                    os.symlink(item.resolve(), dest)  # Create a symlink to avoid copying large files
            elif item.is_dir():
                dest = dest_folder / item.name
                if not dest.exists():
                    os.symlink(item.resolve(), dest)  # Create a symlink to avoid copying large folders

        agent_folders = [str(dest_folder)]
    else:
        raise ValueError(
            f"Invalid trajectory type: {config.trajectory_type}. Must be 'trajectory_generator' or 'sampling_trajectory'."
        )

    return agent_folders


def run_sisfos_with_config(config: SceneConfig, renders_base_dir: Path):
    logger = get_logger()
    renderer = BlenderRenderer(config, verbose=True)
    logger.info("%s", config.setup)
    cam, sun = renderer.setup_total()

    all_models = renderer.get_models_in_blend()
    logger.info(f"Available models in the blend: {all_models}")

    model = renderer.load_spacecraft(model_name=config.selected_model)

    all_models = renderer.get_all_models()
    logger.info(f"Models loaded in scene: {[m.name for m in all_models]}")

    trajectory_file = renders_base_dir / "camera_traj.csv"

    trajectory = read_camera_trajectory(str(trajectory_file))
    trajectory = get_scaled_trajectory_in_ECI(trajectory, earth_dist_scale_factor=config.render.earth_dist_scale_factor)
    frames = make_frames_from_trajectory(trajectory)
    logger.info("[Session] Renders output: %s/", renders_base_dir)
    frame_start_time = time.time()

    frame_ids = config.frame_ids if config.frame_ids else list(range(len(frames)))

    # Keep frame filenames at least 4-digit zero-padded for downstream tooling.
    N_digits = max(4, int(math.log10(len(frames))) + 1)

    gt_root = ensure_dir(renders_base_dir / "GTAnnotations")

    image_out_dir = renders_base_dir / "images_raw"
    masked_out_dir = renders_base_dir / "images"
    if str(config.setup.stars_mode).casefold() == "off":
        if str(config.setup.earth_mode).casefold() == "off":
            image_out_dir = image_out_dir / "Earth_Stars_OFF"
        image_out_dir = image_out_dir / "Stars_OFF"
    elif str(config.setup.earth_mode).casefold() == "off":
        image_out_dir = image_out_dir / "Earth_OFF"

    # Prepare GT folders
    gt_dirs = {
        "gt_npz": ensure_dir(gt_root / "NPZ"),
        "gt_depth": ensure_dir(gt_root / "Depth"),
        "gt_norm": ensure_dir(gt_root / "Normal"),
        "gt_flow": ensure_dir(gt_root / "Flow"),
        "gt_seg": ensure_dir(gt_root / "Seg"),
    }

    total = len(frame_ids)
    logger.info("Enabling blur is: %s", config.setup.enable_blur)

    avg_frame_time = 0.0

    avg_frame_time = 0.0

    with tqdm(total=total, desc=f"Rendering {model.name}") as pbar:
        image_filenames = []
        for i in frame_ids:
            frame_start_time = time.time()
            fr = frames[i]
            fake_fr2 = None
            if str(config.setup.enable_blur).casefold() == "on":
                fake_fr2 = make_fake_frame_from_frame0(
                    fr,
                    seed=12345 + i,
                    cam_dir_max_deg=0.6 * config.setup.blur_motion_factor,
                    cam_radius_scale_sigma=0.01 * config.setup.blur_motion_factor,
                    target_rot_max_deg=1.0 * config.setup.blur_motion_factor,
                    force_camera_lookat=True,
                )

            if str(config.setup.enable_blur).casefold() == "on" and fake_fr2 is not None:
                fps = renderer.scene.render.fps / renderer.scene.render.fps_base
                shutter_frames = config.camera.exposure_time_s * fps * config.setup.blur_shutter_factor
                image_filename = renderer.render_frame_motion_blur_traj(
                    cam,
                    model,
                    sun,
                    fr,
                    fake_fr2,
                    i,
                    shutter_frames,
                    image_out_dir,
                    config.camera.exposure_time_s,
                    N_digits,
                )
            else:
                image_filename = renderer.render_frame_v2(
                    cam, model, sun, fr, i, image_out_dir, config.camera.exposure_time_s, N_digits
                )

            image_filenames.append(image_filename)
            pbar.update(1)

            # Post-process NPZ
            target_dist = float(np.linalg.norm(fr["p_G_I"] - fr["p_C_I"]))
            npz_src = Path(os.path.join(image_out_dir, f"{i:04d}.npz"))
            if npz_src.exists():
                handle_gt_from_npz(
                    npz_src,
                    gt_dirs["gt_npz"],
                    gt_dirs["gt_depth"],
                    gt_dirs["gt_norm"],
                    gt_dirs["gt_flow"],
                    gt_dirs["gt_seg"],
                    target_dist,
                    raw_image_filename=image_filename,
                    raw_images_dir=str(image_out_dir),
                    masked_images_dir=str(masked_out_dir),
                )

            current_frame_time = time.time() - frame_start_time
            avg_frame_time = (avg_frame_time * i + current_frame_time) / (i + 1)
            time_remaining_estimate = avg_frame_time * (total - i - 1)
            time_delta_str = str(datetime.timedelta(seconds=int(time_remaining_estimate)))
            logger.info("Generated frame %d in %.2f seconds. Output: %s", i, current_frame_time, image_filename)
            logger.info("Average frame time: %.2f seconds.", avg_frame_time)
            logger.info("Estimated time remaining: %s", time_delta_str)
            logger.info(
                "Estimated time of completion: %s",
                datetime.datetime.now() + datetime.timedelta(seconds=int(time_remaining_estimate)),
            )

    # End of frames loop
    timestamps = [float(trajectory["t"][fid]) for fid in frame_ids]
    image_paths = [os.path.join("images", image_filename) for image_filename in image_filenames]
    create_image_list(str(renders_base_dir), timestamps, image_paths)

    logger.info("Finished rendering frames for %s. Output directory: %s", model.name, renders_base_dir)
    if config.setup.generate_video:
        logger.info("Saving video")
        images_to_video_blender_sequence(
            image_dir=image_out_dir,
            image_filenames=image_filenames,
            output_path=renders_base_dir / "frames.mp4",
            fps=config.setup.video_fps,
        )


def run_sweep(sweep_config: SweepConfig):
    configs = sweep_config.generate_sweep_configs()

    output_dir = Path("./renders") / get_timestamp_folder()
    ensure_dir(output_dir)

    setup_logger(log_file=output_dir / "run.log")
    logger = get_logger()
    logger.info("Running sweep with %d configurations. Output base dir: %s", len(configs), output_dir)

    # Save the config for reproducibility
    with open(output_dir / "sweep_configs.json", "w") as f:
        payload = sweep_config.model_dump()
        json.dump(payload, f, indent=2)

    PROJECT_ROOT = Path(__file__).parent.resolve()

    render_jobs: list[tuple[SceneConfig, Path]] = []

    for i, config in enumerate(configs):
        config_prefix = f"Config_{i + 1}"

        # Save each expanded config at the sweep root for reproducibility.
        with open(output_dir / f"{config_prefix}.json", "w") as f:
            payload = config.model_dump()
            json.dump(payload, f, indent=2)

        # Ensure paths are absolute
        if not os.path.isabs(config.scene_blend_path):
            config.scene_blend_path = str(PROJECT_ROOT / config.scene_blend_path)
        if config.hdri_path and not os.path.isabs(config.hdri_path):
            config.hdri_path = str(PROJECT_ROOT / config.hdri_path)
        for _obj_name, obj_cfg in config.objects.items():
            if obj_cfg.blend_path and not os.path.isabs(obj_cfg.blend_path):
                obj_cfg.blend_path = str(PROJECT_ROOT / obj_cfg.blend_path)

        agent_folders = generate_trajectories(config, output_dir, config_prefix=config_prefix)
        render_jobs.extend((config, Path(agent_folder)) for agent_folder in agent_folders)

    for config, agent_folder in render_jobs:
        run_sisfos_with_config(config, agent_folder)


if __name__ == "__main__":
    # Handle Blender's command-line arguments (-b, -q, -P script.py)
    # Blender passes arguments after '--' to the Python script
    config_path = "./config.json"  # Default

    # Look for arguments after '--' (Blender convention)
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]  # Arguments after '--'
    else:
        raise RuntimeError(
            "No configuration file specified. Please provide a config JSON path after '--' when running the script."
        )

    parser = argparse.ArgumentParser(description="SISIFOS Parser")

    parser.add_argument("--config_path", type=str, help="Path to the base configuration JSON file")
    parser.add_argument("--sweep_config_path", type=str, help="Path to the sweep configuration JSON file (optional)")

    args = parser.parse_args(argv)

    if args.sweep_config_path:
        if args.config_path:
            raise RuntimeError(
                "Cannot specify --sweep_config_path together with --config_path. Please provide only one of these options."
            )

        sweep_config_json = json.load(open(args.sweep_config_path))
        sweep_config = SweepConfig.model_validate(sweep_config_json)

    elif args.config_path:
        base_config_json = json.load(open(args.config_path))

        sweep_config_json = {}

        sweep_config_json = {"base_config": base_config_json, "sweep_parameters": sweep_config_json}
        sweep_config = SweepConfig.model_validate(sweep_config_json)

    else:
        config_json = json.load(open(DEFAULT_CONFIG_PATH))
        sweep_config_json = {"base_config": config_json, "sweep_parameters": {}}
        sweep_config = SweepConfig.model_validate(sweep_config_json)

    run_sweep(sweep_config)
