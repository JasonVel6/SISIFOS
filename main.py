"""
© Dynamics and Control Systems Laboratory, Georgia Institute of Technology
SISIFOS: Specialized Illumination SImulator For Orbiting Spacecraft

This is the main file, setting up the simulation, loading the configuration
and actually rendering the data (images and annotations).

Iason Georgios Velentzas (ivelentzas3@gatech.edu)

"""

import sys
import os
import math
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())
from modules.config import SceneConfig
from modules.renderer import BlenderRenderer
from modules.io_utils import ensure_dir, get_timestamp_folder, format_R_RPO, handle_gt_from_npz, vprint
from modules.trajectory import (
    write_camera_trajectory_v2, load_camera_trajectory_v2, 
    sun_sweep_90, make_fake_frame_from_frame0
)

SUN_AZ_MAN = [0]
SUN_EL_MAN = [-90, -45, 0, 45, 90]

def main(config_path: str):
    PROJECT_ROOT = Path(__file__).parent.resolve()
    
    config = SceneConfig.from_json(config_path)
    if not os.path.isabs(config.scene_blend_path):
        config.scene_blend_path = str(PROJECT_ROOT / config.scene_blend_path)
    if config.hdri_path and not os.path.isabs(config.hdri_path):
        config.hdri_path = str(PROJECT_ROOT / config.hdri_path)
    for obj_name, obj_cfg in config.objects.items():
        if obj_cfg.blend_path and not os.path.isabs(obj_cfg.blend_path):
            obj_cfg.blend_path = str(PROJECT_ROOT / obj_cfg.blend_path)
    
    renderer = BlenderRenderer(config, verbose=True)
    print(config.setup)
    cam, sun = renderer.setup_total()

    timestamp = get_timestamp_folder()
    renders_base_dir = Path("./renders") / timestamp
    gt_path = renders_base_dir / "camera_traj.txt"
    ensure_dir(gt_path.parent)

    if not gt_path.exists():
        write_camera_trajectory_v2(
            str(gt_path),
            N=config.setup.num_frames,
            R_LEO=config.setup.R_LEO,
            R_RPO=config.setup.R_RPO,
        )
    
    frames = load_camera_trajectory_v2(str(gt_path))
    print(f"\n[Session] Timestamp: {timestamp}")
    print(f"[Session] Renders output: {renders_base_dir}/")
    
    models = renderer.select_models_to_render()
    vprint(f"Rendering {len(models)} models: {[m.name for m in models]}", True)
    
    frame_ids = config.frame_ids if config.frame_ids else list(range(len(frames)))
    R_RPO_tag = format_R_RPO(config.setup.R_RPO)
    res_x, res_y = config.camera.resolution
    
    # Setup Sweeps
    exp_sweep_map = config.setup.sweep_exposure if config.setup.sweep_exposure else {"00": config.setup.t_ref_s}
    sun_sweep_map = config.setup.sweep_sun_az_el if config.setup.sweep_sun_az_el else sun_sweep_90()
    
    N_digits = int(math.log10(len(frames))) + 1
    N_azel_keys = int(math.log10(len(sun_sweep_map))) + 1
    N_exp_keys = int(math.log10(len(exp_sweep_map))) + 1

    for model in models:
        subfolder = f"_{R_RPO_tag}_RESX{res_x}_RESY{res_y}"
        model_out_dir = ensure_dir(renders_base_dir / (model.name + subfolder))
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

if __name__ == "__main__":
    # Handle Blender's command-line arguments (-b, -q, -P script.py)
    # Blender passes arguments after '--' to the Python script
    config_path = "./config.json"  # Default
    
    # Look for arguments after '--' (Blender convention)
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    else:
        # Fallback: look for .json file in sys.argv
        for arg in sys.argv[1:]:
            if arg.endswith('.json'):
                config_path = arg
                break
    
    print(f"[Renderer] Loading config from: {config_path}")
    main(config_path)