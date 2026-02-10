"""
© Dynamics and Control Systems Laboratory, Georgia Institute of Technology
SISIFOS: Specialized Illumination SImulator For Orbiting Spacecraft

This is the main file, setting up the simulation, loading the configuration
and actually rendering the data (images and annotations).

Iason Georgios Velentzas (ivelentzas3@gatech.edu)

"""

import argparse
import sys
import os
import math
from pathlib import Path
from tqdm import tqdm
import argparse
import json

sys.path.append(os.getcwd())
from modules.config import SceneConfig, SweepConfig
from modules.renderer import BlenderRenderer
from modules.io_utils import ensure_dir, get_timestamp_folder, format_R_RPO, handle_gt_from_npz, vprint
from modules.trajectory.sampling_trajectory import (
    write_camera_trajectory_fib, 
    sun_sweep_90, make_fake_frame_from_frame0
)
from modules.trajectory.trajectory_io import (
    read_camera_trajectory_to_frames
)
from modules.trajectory.generateTrajectoriesUnified import generate_trajectories_dynamical

DEFAULT_CONFIG_PATH = "/config/config_example_basic.json"

def generate_trajectories(config: SceneConfig, output_dir: Path) -> list[str]:
        # Generate the trajectory TODO at some point maybe we can pass a path but the trajectory generator is quite fast for now
    if config.trajectoty_type == "trajectory_generator":
        agent_folders = generate_trajectories_dynamical(config.trajectory, str(output_dir))
        
    elif config.trajectoty_type == "sampling_trajectory":
        raise NotImplementedError("Sampling-based trajectory generation is not yet implemented. Please use 'trajectory_generator' or implement the sampling-based generator.")

        timestamp = get_timestamp_folder()
        renders_base_dir = PROJECT_ROOT / "renders" / timestamp
        gt_path = renders_base_dir / "camera_traj.txt"
        ensure_dir(gt_path.parent)

        if not gt_path.exists(): # why would it exist? we generate this file as a timestamp essentially getting rid of this possibility
            write_camera_trajectory_v2(
                str(gt_path),
                N=config.setup.num_frames,
                R_LEO=config.setup.R_LEO,
                R_RPO=config.setup.R_RPO,
        )
    else:
        raise ValueError(f"Invalid trajectory type: {config.trajectoty_type}. Must be 'trajectory_generator' or 'sampling_trajectory'.")
    
    return agent_folders

def run_sisfos_with_config(config: SceneConfig, renders_base_dir: Path):
    renderer = BlenderRenderer(config, verbose=True)
    print(config.setup)
    cam, sun = renderer.setup_total()

    trajectory_file = renders_base_dir / "camera_traj.txt"
    
    frames = read_camera_trajectory_to_frames(str(trajectory_file))
    print(f"[Session] Renders output: {renders_base_dir}/")
    
    models = renderer.select_models_to_render()
    vprint(f"Rendering {len(models)} models: {[m.name for m in models]}", True)
    
    frame_ids = config.frame_ids if config.frame_ids else list(range(len(frames)))
    res_x, res_y = config.camera.resolution
    
    # Setup Sweeps
    # TODO we will need to make this able to do in a config file
    exp_sweep_map = config.setup.sweep_exposure if config.setup.sweep_exposure else {"00": config.setup.t_ref_s}
    sun_sweep_map = config.setup.sweep_sun_az_el if config.setup.sweep_sun_az_el else sun_sweep_90()
    
    N_digits = int(math.log10(len(frames))) + 1
    N_azel_keys = int(math.log10(len(sun_sweep_map))) + 1
    N_exp_keys = int(math.log10(len(exp_sweep_map))) + 1

    for model in models:
        model_folder_name = f"{model.name}"
        model_out_dir = ensure_dir(renders_base_dir / model_folder_name)
        gt_root = ensure_dir(model_out_dir / "GTAnnotations")

        renderer.hide_all_except(model, models)
        model_out_dir = model_out_dir / "FULL"
        if str(config.setup.stars_mode).casefold() == "off":
            if str(config.setup.earth_mode).casefold() == "off":
                model_out_dir = model_out_dir / "Earth_Stars_OFF"
            model_out_dir = model_out_dir / "Stars_OFF"
        elif str(config.setup.earth_mode).casefold() == "off":
            model_out_dir = model_out_dir / "Earth_OFF"

        # Prepare GT folders
        gt_dirs = {
            "gt_npz": ensure_dir(gt_root / "NPZ"),
            "gt_depth": ensure_dir(gt_root / "Depth"),
            "gt_norm": ensure_dir(gt_root / "Normal"),
            "gt_flow": ensure_dir(gt_root / "Flow"),
            "gt_seg": ensure_dir(gt_root / "Seg")
        }
        
        if config.model_rotation_z_deg != 0:
            renderer.rotate_z(model, config.model_rotation_z_deg)

        total = len(frame_ids) * len(exp_sweep_map) * len(sun_sweep_map)/2
        print("Enabling blur is: ", config.setup.enable_blur)
        
        with tqdm(total=total, desc=f"Rendering {model.name}") as pbar:
            for i in frame_ids:
                fr = frames[i]
                fake_fr2 = None
                if str(config.setup.enable_blur).casefold()=="on":
                    fake_fr2 = make_fake_frame_from_frame0(
                        fr, seed=12345 + i,
                        cam_dir_max_deg=0.6*config.setup.blur_motion_factor,
                        cam_radius_scale_sigma=0.01*config.setup.blur_motion_factor,
                        target_rot_max_deg=1.0*config.setup.blur_motion_factor,
                        force_camera_lookat=True,
                    )
                
                for sweep_key_exp, exp_value in exp_sweep_map.items():
                    current_output_dir = model_out_dir / f"exp_{str(sweep_key_exp).zfill(N_exp_keys)}"
                    ensure_dir(current_output_dir)
                    
                    for sweep_key_azel, azel in sun_sweep_map.items():
                        if int(sweep_key_azel) < 4:
                            if str(config.setup.enable_blur).casefold()=="on" and fake_fr2 is not None:
                                fps = renderer.scene.render.fps / renderer.scene.render.fps_base
                                shutter_frames = exp_value * fps * config.setup.blur_shutter_factor
                                renderer.render_frame_motion_blur_traj(
                                    cam, model, sun, fr, fake_fr2, i, shutter_frames,
                                    current_output_dir, exp_value, azel, sweep_key_azel, N_azel_keys, N_digits
                                )
                            else:
                                renderer.render_frame_v2(
                                    cam, model, sun, fr, i, current_output_dir, exp_value,
                                    azel, sweep_key_azel, N_azel_keys, N_digits
                                )
                            pbar.update(1)

                    # Post-process NPZ
                    npz_src = Path(os.path.join(current_output_dir, f'{i:04d}.npz'))
                    if npz_src.exists():
                        handle_gt_from_npz(
                            npz_src,
                            gt_dirs["gt_npz"], gt_dirs["gt_depth"], gt_dirs["gt_norm"],
                            gt_dirs["gt_flow"], gt_dirs["gt_seg"],
                            config.setup.R_RPO
                        )

def run_sweep(sweep_config: SweepConfig):
    configs = sweep_config.generate_sweep_configs()

    output_dir = Path("./renders") / get_timestamp_folder()
    ensure_dir(output_dir)
    print(f"Running sweep with {len(configs)} configurations. Output base dir: {output_dir}")

    # Save the config for reproducibility
    with open(output_dir / "sweep_configs.json", 'w') as f:
        payload = sweep_config.model_dump()
        json.dump(payload, f, indent=2)

    for i, config in enumerate(configs):
        trial_output_dir = output_dir / f"sweep_{i+1}"
        ensure_dir(trial_output_dir)

        # Save the config used for this trail
        with open(trial_output_dir / "config.json", 'w') as f:
            payload = config.model_dump()
            json.dump(payload, f, indent=2)

            # TODO we may want to also save the trajectory generator config but this is a task for later

            PROJECT_ROOT = Path(__file__).parent.resolve()

        # Ensure paths are absolute
        if not os.path.isabs(config.scene_blend_path):
            config.scene_blend_path = str(PROJECT_ROOT / config.scene_blend_path)
        if config.hdri_path and not os.path.isabs(config.hdri_path):
            config.hdri_path = str(PROJECT_ROOT / config.hdri_path)
        for obj_name, obj_cfg in config.objects.items():
            if obj_cfg.blend_path and not os.path.isabs(obj_cfg.blend_path):
                obj_cfg.blend_path = str(PROJECT_ROOT / obj_cfg.blend_path)

        agent_folders = generate_trajectories(config, trial_output_dir)
        for agent_folder in agent_folders:
            run_sisfos_with_config(config, Path(agent_folder))

if __name__ == "__main__":
    # Handle Blender's command-line arguments (-b, -q, -P script.py)
    # Blender passes arguments after '--' to the Python script
    config_path = "./config.json"  # Default
    
    # Look for arguments after '--' (Blender convention)
    # TODO lets fix this using actual argparse for readability and robustness
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]  # Arguments after '--'
    else:
        raise RuntimeError("No configuration file specified. Please provide a config JSON path after '--' when running the script.")

    parser = argparse.ArgumentParser(description="SISIFOS Parser")

    parser.add_argument('--config_path', type=str, help='Path to the base configuration JSON file')
    parser.add_argument('--sweep_config_path', type=str, help='Path to the sweep configuration JSON file (optional)')

    args = parser.parse_args(argv)

    if args.sweep_config_path:

        if args.config_path or args.sweep_config_path:
            raise RuntimeError("Cannot specify --sweep_config_path together with --config_path. Please provide only one of these options.")

        print(f"[SISFOS] Loading sweep config from: {args.sweep_config_path}")
        sweep_config_json = json.load(open(args.sweep_config_path, 'r'))
        sweep_config = SweepConfig.model_validate(sweep_config_json)

    elif args.config_path:
        print(f"[SISFOS] Loading config from: {args.config_path}")
        base_config_json = json.load(open(args.config_path, 'r'))

        sweep_config_json = {}

        sweep_config_json = {"base_config": base_config_json, "sweep_parameters": sweep_config_json}
        sweep_config = SweepConfig.model_validate(sweep_config_json)

    else:
        config_json = json.load(open(DEFAULT_CONFIG_PATH, 'r'))
        sweep_config_json = {"base_config": config_json, "sweep_parameters": {}}
        sweep_config = SweepConfig.model_validate(sweep_config_json)

    run_sweep(sweep_config)
