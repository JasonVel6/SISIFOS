"""Pipeline render test using hardcoded example config values.

Renders a single frame named frame_000_sun_02.png using a Blend file
"""

import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.config import SceneConfig, ObjectConfig, CameraConfig, RenderConfig, SetupConfig
from modules.renderer import BlenderRenderer
from modules.io_utils import ensure_dir, format_R_RPO, get_timestamp_folder
from modules.trajectory import write_camera_trajectory_v2, load_camera_trajectory_v2, sun_sweep_90
from modules.blender_utils import get_world_bounds, append_blend_objects, scale_object_by_factor


def _pick_blend_name() -> str:
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return "Themis_wip_packed"


def _build_hardcoded_config(blend_name: str) -> SceneConfig:
    blend_path = PROJECT_ROOT / "assets" / f"{blend_name}.blend"
    return SceneConfig(
        scene_blend_path=str(blend_path),
        hdri_path=str(PROJECT_ROOT / "assets" / "starmap_2020_16k.exr"),
        objects={
            "Earth": ObjectConfig(
                name="Earth",
                blend_path=str(PROJECT_ROOT / "assets" / "Earth.blend"),
                position=[0.0, 0.0, 0.0],
                rotation_euler_deg=[0.0, 0.0, 0.0],
                scale=1.0,
                hide_render=False,
                extra_scale=10.0,
            ),
        },
        camera=CameraConfig(
            focal_length=170.0,
            clip_start=0.00001,
            clip_end=5000000.0,
            resolution=(2048, 2048),
        ),
        render=RenderConfig(
            engine="CYCLES",
            samples=16,
            bg_color=(0.0, 0.0, 0.0, 1.0),
            motion_blur=0.0,
            noise_strength=0.0,
        ),
        setup=SetupConfig(
            num_frames=1,
            R_RPO=36.0,
            R_LEO=8000.0,
            sweep_exposure=None,
            sweep_sun_az_el=None,
            earth_mode="ON",
            stars_mode="ON",
            t_ref_s=0.01666667,
            sun_az_ref=0.0,
            sun_el_ref=90.0,
            enable_blur="OFF",
            blur_shutter_factor=8.8,
            blur_motion_factor=8.8,
            enable_glare="OFF",
            glare_threshold=0.95,
            glare_size=6,
        ),
        save_depth=False,
        save_normals=False,
        save_optical_flow=False,
        save_segmentation=False,
        save_obj_poses=False,
        frame_ids=[0],
        selected_models=[],
        model_rotation_z_deg=45.0,
    )


def _gather_satellite_parts(exclude_names):
    return [
        o for o in bpy.data.objects
        if o.type == "MESH" and o.name not in exclude_names
    ]


def _hide_non_targets(target, keep_names):
    keep_set = set(keep_names)
    keep_set.add(target.name)
    for child in target.children_recursive:
        keep_set.add(child.name)

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.name in keep_set:
            obj.hide_render = False
            continue
        obj.hide_render = True


def _parent_keep_transform(child, parent):
    child.matrix_world = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def _ensure_camera(scene):
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
    return cam


def _ensure_sun(scene):
    sun = bpy.data.objects.get("Sun")
    if sun is None:
        sun_data = bpy.data.lights.new(name="Sun", type="SUN")
        sun = bpy.data.objects.new("Sun", sun_data)
        scene.collection.objects.link(sun)
    return sun


def _setup_world_hdri(scene, hdri_path: Path):
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    if not hdri_path.exists():
        bg = nodes.new(type="ShaderNodeBackground")
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs[1].default_value = 1.0
        out = nodes.new(type="ShaderNodeOutputWorld")
        links.new(bg.outputs["Background"], out.inputs["Surface"])
        print(f"[World] HDRI not found, using black background: {hdri_path}")
        return

    img = bpy.data.images.load(str(hdri_path), check_existing=True)
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = img
    env_tex.image.colorspace_settings.name = "Non-Color"
    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs[1].default_value = 1.0
    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    print(f"[World] Loaded HDRI: {hdri_path}")


def _setup_scene(renderer: BlenderRenderer, config: SceneConfig):
    bpy.ops.wm.open_mainfile(filepath=config.scene_blend_path)
    scene = bpy.context.scene

    scene.render.engine = config.render.engine
    scene.cycles.samples = config.render.samples
    scene.render.resolution_x, scene.render.resolution_y = config.camera.resolution
    scene.view_settings.exposure = 0.0
    scene.render.image_settings.file_format = "PNG"

    renderer.scene = scene
    renderer.world = scene.world

    _setup_world_hdri(scene, Path(config.hdri_path))

    cam = _ensure_camera(scene)
    cam.rotation_mode = "QUATERNION"
    cam.data.lens = config.camera.focal_length
    cam.data.clip_start = config.camera.clip_start
    cam.data.clip_end = config.camera.clip_end

    sun = _ensure_sun(scene)
    sun.data.energy = 10.0

    if config.objects.get("Earth") and config.objects["Earth"].blend_path:
        if bpy.data.objects.get("Earth") is None:
            append_blend_objects(config.objects["Earth"].blend_path)
        earth = bpy.data.objects.get("Earth")
        clouds = bpy.data.objects.get("Clouds")
        atmo = bpy.data.objects.get("Atmo")
        if earth:
            scale_object_by_factor(earth, config.objects["Earth"].extra_scale)
        if clouds:
            scale_object_by_factor(clouds, config.objects["Earth"].extra_scale)
        if atmo:
            scale_object_by_factor(atmo, config.objects["Earth"].extra_scale)

    return cam, sun


def main() -> None:
    blend_name = _pick_blend_name()
    config = _build_hardcoded_config(blend_name)
    if not os.path.isfile(config.scene_blend_path):
        raise FileNotFoundError(f"Blend file not found: {config.scene_blend_path}")

    renderer = BlenderRenderer(config, verbose=True)
    cam, sun = _setup_scene(renderer, config)

    sun_sweep_map = sun_sweep_90()
    if "02" not in sun_sweep_map:
        raise RuntimeError("sun_sweep_90 did not produce key '02'.")

    output_root = ensure_dir(PROJECT_ROOT / "renders" / get_timestamp_folder())
    traj_path = output_root / "camera_traj.txt"
    write_camera_trajectory_v2(
        str(traj_path),
        N=1,
        R_LEO=config.setup.R_LEO,
        R_RPO=config.setup.R_RPO,
        verbose=True,
    )

    frames = load_camera_trajectory_v2(str(traj_path))
    if not frames:
        raise RuntimeError("No frames loaded from trajectory.")

    satellite_parts = _gather_satellite_parts({"Earth", "Clouds", "Atmo", "Sun", "Camera"})
    if not satellite_parts:
        raise RuntimeError("No spacecraft mesh parts found in scene.")

    def _size(o):
        min_c, max_c = get_world_bounds(o)
        dims = max_c - min_c
        return max(dims.x, dims.y, dims.z)

    target = max(satellite_parts, key=_size)
    for part in satellite_parts:
        if part == target:
            continue
        _parent_keep_transform(part, target)

    _hide_non_targets(target, {"Earth", "Clouds", "Atmo"})
    if config.model_rotation_z_deg != 0:
        renderer.rotate_z(target, config.model_rotation_z_deg)

    rpo_tag = format_R_RPO(config.setup.R_RPO)
    model_out_dir = ensure_dir(output_root / f"{blend_name}_{rpo_tag}" / "FULL")

    renderer.render_frame_v2(
        cam=cam,
        model=target,
        sun=sun,
        frame_dict=frames[0],
        frame_id=0,
        output_dir=model_out_dir,
        exposure_time_s=config.setup.t_ref_s,
        azel=sun_sweep_map["02"],
        sun_key="02",
        N_azel_keys=2,
        N_digits=3,
    )

    print(f"Render saved under: {model_out_dir}")


if __name__ == "__main__":
    main()