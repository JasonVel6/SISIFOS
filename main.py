"""
© Dynamics and Control Systems Laboratory, Georgia Institute of Technology
SISIFOS: Specialized Illumination SImulator For Orbiting Spacecraft

This is the main file, setting up the simulation, loading the configuration
and actually rendering the data (images and annotations).

Iason Georgios Velentzas (ivelentzas3@gatech.edu)

"""

import argparse
import copy
import datetime
import json
import math
import os
import shutil
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
from modules.trajectory.seed_screening import evaluate_seed_candidate, write_screening_report
from modules.trajectory.trajectory_io import (
    get_scaled_trajectory_in_ECI,
    make_frames_from_trajectory,
    read_camera_trajectory,
)

DEFAULT_CONFIG_PATH = "/config/config_example_basic.json"


def _sanitize_folder_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in value)
    return token.strip("_") or "Unknown"


def _resolve_frame_ids(config: SceneConfig, num_frames: int) -> list[int]:
    if config.frame_ids is not None:
        return config.frame_ids

    start_frame = config.from_frame_id or 0
    if start_frame < 0:
        raise ValueError(f"from_frame_id must be non-negative, got {start_frame}")
    if start_frame >= num_frames:
        raise ValueError(f"from_frame_id={start_frame} is out of range for trajectory with {num_frames} frames")

    return list(range(start_frame, num_frames))


def resolve_config_asset_paths(config: SceneConfig, project_root: Path | None = None) -> SceneConfig:
    project_root = project_root or Path(__file__).parent.resolve()

    if not os.path.isabs(config.scene_blend_path):
        config.scene_blend_path = str(project_root / config.scene_blend_path)
    if config.hdri_path and not os.path.isabs(config.hdri_path):
        config.hdri_path = str(project_root / config.hdri_path)
    for _obj_name, obj_cfg in config.objects.items():
        if obj_cfg.blend_path and not os.path.isabs(obj_cfg.blend_path):
            obj_cfg.blend_path = str(project_root / obj_cfg.blend_path)

    return config


def get_render_output_dirs(config: SceneConfig, renders_base_dir: Path) -> tuple[Path, Path]:
    image_out_dir = renders_base_dir / "images_raw"
    masked_out_dir = renders_base_dir / "images"
    if str(config.setup.stars_mode).casefold() == "off":
        if str(config.setup.earth_mode).casefold() == "off":
            image_out_dir = image_out_dir / "Earth_Stars_OFF"
        image_out_dir = image_out_dir / "Stars_OFF"
    elif str(config.setup.earth_mode).casefold() == "off":
        image_out_dir = image_out_dir / "Earth_OFF"

    return image_out_dir, masked_out_dir


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


def _derive_attempt_seed(base_seed: int, attempt_idx: int) -> int:
    if attempt_idx == 0:
        return int(base_seed) & 0x7FFFFFFF
    seed_seq = np.random.SeedSequence(base_seed)
    child_state = seed_seq.generate_state(attempt_idx + 1, dtype=np.uint32)
    return int(child_state[attempt_idx]) & 0x7FFFFFFF


def _resolve_screening_seed(config: SceneConfig) -> int:
    if config.trajectory.seed is not None:
        return int(config.trajectory.seed) & 0x7FFFFFFF
    return int(time.time_ns()) & 0x7FFFFFFF


def _promote_candidate_outputs(staging_dir: Path, output_dir: Path, candidate_prefix: str, final_prefix: str) -> None:
    trajectory_json = staging_dir / f"{candidate_prefix}_trajectory.json"
    if trajectory_json.exists():
        shutil.move(str(trajectory_json), str(output_dir / f"{final_prefix}_trajectory.json"))

    for path in staging_dir.glob(f"{candidate_prefix}*"):
        if path.name == f"{candidate_prefix}_trajectory.json":
            continue
        destination = output_dir / path.name.replace(candidate_prefix, final_prefix, 1)
        if destination.exists():
            raise FileExistsError(f"Cannot promote candidate output '{destination}'; destination already exists.")
        shutil.move(str(path), str(destination))


def _generate_screened_trajectories(
    config: SceneConfig, output_dir: Path, config_prefix: str, screening_root: Path
) -> tuple[list[Path], dict]:
    logger = get_logger()
    screening_cfg = config.screening
    attempts_payload: list[dict] = []

    if not screening_cfg.enabled:
        agent_folders = [Path(folder) for folder in generate_trajectories(config, output_dir, config_prefix=config_prefix)]
        return agent_folders, {"status": "disabled", "attempts": attempts_payload}

    if config.trajectory_type != "trajectory_generator":
        raise ValueError("Seed screening currently supports only trajectory_generator runs.")

    max_attempts = max(int(screening_cfg.max_attempts), 1)
    if not screening_cfg.resample_on_reject:
        max_attempts = 1
    base_seed = _resolve_screening_seed(config)
    last_report = None

    for attempt_idx in range(max_attempts):
        attempt_config = copy.deepcopy(config)
        attempt_config.trajectory.seed = _derive_attempt_seed(base_seed, attempt_idx)
        if attempt_idx > 0 and screening_cfg.resample_on_reject:
            attempt_config.trajectory.illumination_seed = None

        candidate_prefix = f"{config_prefix}_attempt_{attempt_idx + 1}"
        agent_folders = [Path(folder) for folder in generate_trajectories(attempt_config, screening_root, config_prefix=candidate_prefix)]
        report = evaluate_seed_candidate(agent_folders, attempt_config, config_prefix=config_prefix)
        report.attempts_used = attempt_idx + 1
        last_report = report

        attempt_record = report.model_dump()
        attempt_record["attempt_idx"] = attempt_idx + 1
        attempts_payload.append(attempt_record)

        report_path = screening_root / f"{candidate_prefix}_screening.json"
        write_screening_report(report, report_path)

        if report.verdict == "ACCEPT":
            config.trajectory.seed = attempt_config.trajectory.seed
            config.trajectory.illumination_seed = attempt_config.trajectory.illumination_seed
            _promote_candidate_outputs(screening_root, output_dir, candidate_prefix, config_prefix)
            final_agent_folders = [
                Path(str(agent_folder).replace(str(screening_root / candidate_prefix), str(output_dir / config_prefix)))
                for agent_folder in agent_folders
            ]
            logger.info(
                "Accepted seed %s for %s after %d attempt(s).",
                config.trajectory.seed,
                config_prefix,
                attempt_idx + 1,
            )
            return final_agent_folders, {
                "status": "accepted",
                "base_seed": base_seed,
                "attempts": attempts_payload,
                "accepted_seed": config.trajectory.seed,
            }

        logger.warning(
            "Rejected seed %s for %s on attempt %d/%d: %s",
            attempt_config.trajectory.seed,
            config_prefix,
            attempt_idx + 1,
            max_attempts,
            "; ".join(report.diagnosis) if report.diagnosis else "screening checks failed",
        )

    if last_report is not None:
        raise ValueError(
            f"{config_prefix} failed screening after {max_attempts} attempts. "
            f"Last rejection reasons: {'; '.join(last_report.diagnosis) if last_report.diagnosis else 'unknown'}"
        )
    raise ValueError(f"{config_prefix} failed screening before any candidate report was produced.")


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

    frame_ids = _resolve_frame_ids(config, len(frames))

    # Keep frame filenames at least 4-digit zero-padded for downstream tooling.
    N_digits = max(4, int(math.log10(len(frames))) + 1)

    gt_root = ensure_dir(renders_base_dir / "GTAnnotations")

    image_out_dir, masked_out_dir = get_render_output_dirs(config, renders_base_dir)

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

    with tqdm(total=total, desc=f"Rendering {model.name}") as pbar:
        image_filenames = []
        for rendered_idx, i in enumerate(frame_ids):
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
            avg_frame_time = (avg_frame_time * rendered_idx + current_frame_time) / (rendered_idx + 1)
            time_remaining_estimate = avg_frame_time * (total - rendered_idx - 1)
            time_delta_str = str(datetime.timedelta(seconds=int(time_remaining_estimate)))

            if rendered_idx == 0 or rendered_idx % 5 == 0 or rendered_idx == total - 1:
                logger.info("============================================================================================")
                logger.info(f"Finished rendering frame {i} ({rendered_idx + 1}/{total}) in {current_frame_time:.2f} seconds.")
                logger.info(f"Average frame time so far: {avg_frame_time:.2f} seconds.")
                logger.info(f"Estimated time remaining: {time_delta_str}")
                logger.info(f"Estimated time of completion: {datetime.datetime.now() + datetime.timedelta(seconds=int(time_remaining_estimate))}")

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
    screening_root = ensure_dir(output_dir / "_screening")

    render_jobs: list[tuple[SceneConfig, Path]] = []
    screening_summary: dict[str, dict] = {}

    for i, config in enumerate(configs):
        config_prefix = f"Config_{i + 1}"

        # Save each expanded config at the sweep root for reproducibility.
        with open(output_dir / f"{config_prefix}.json", "w") as f:
            payload = config.model_dump()
            json.dump(payload, f, indent=2)

        resolve_config_asset_paths(config, PROJECT_ROOT)

        agent_folders, screening_info = _generate_screened_trajectories(
            config, output_dir, config_prefix=config_prefix, screening_root=screening_root
        )
        screening_summary[config_prefix] = screening_info

        # Save each expanded config after seed screening so the accepted seed is recorded.
        with open(output_dir / f"{config_prefix}.json", "w") as f:
            payload = config.model_dump()
            json.dump(payload, f, indent=2)

        render_jobs.extend((config, Path(agent_folder)) for agent_folder in agent_folders)

    with open(output_dir / "screening_summary.json", "w") as f:
        json.dump(screening_summary, f, indent=2)

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
