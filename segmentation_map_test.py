"""Generate segmentation maps along a Fibonacci camera trajectory.

Usage:
  blender -b -P segmentation_map_test.py -- Themis_wip_packed
"""

import sys
import shutil
from pathlib import Path

import bpy
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.trajectory import write_camera_trajectory_v2, load_camera_trajectory_v2
from modules.io_utils import ensure_dir, get_timestamp_folder
from modules.blender_utils import get_world_bounds, set_sun_direction
import modules.addon_ground_truth_generation as vision_addon


NUM_FRAMES = 50
R_LEO = 8000.0
R_RPO = 36.0
SUN_AZ_DEG = 0.0
SUN_EL_DEG = 90.0
HIGH_CONTRAST_PALETTE = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0),
    (0.5, 1.0, 0.0),
    (0.0, 0.5, 1.0),
    (1.0, 0.0, 0.5),
    (0.5, 0.0, 1.0),
]


def _pick_blend_name() -> str:
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return "Themis_wip_packed"


def _resolve_blend_path(name: str) -> Path:
    name_path = Path(name)
    if name_path.is_absolute() and name_path.exists():
        return name_path
    if name.lower().endswith(".blend"):
        candidate = PROJECT_ROOT / "assets" / name
        if candidate.exists():
            return candidate
        return name_path
    return PROJECT_ROOT / "assets" / f"{name}.blend"


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


def _setup_scene(blend_path: Path):
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.use_persistent_data = True
    scene.cycles.device = "GPU"

    cam = _ensure_camera(scene)
    cam.rotation_mode = "QUATERNION"
    cam.data.lens = 170.0
    cam.data.clip_start = 0.00001
    cam.data.clip_end = 5000000.0

    sun = _ensure_sun(scene)
    sun.data.energy = 10.0

    return scene, cam, sun


def _configure_segmentation(scene):
    vision_addon.register()
    vb = scene.vision_blender
    vb.bool_save_gt_data = True
    vb.bool_save_segmentation_masks = True
    vb.bool_save_depth = False
    vb.bool_save_normals = False
    vb.bool_save_cam_param = False
    vb.bool_save_opt_flow = False
    vb.bool_save_obj_poses = False


def _color_for_id(idx: int, color_map: dict[int, tuple[float, float, float]]):
    if idx in color_map:
        return color_map[idx]
    if idx == 0:
        return HIGH_CONTRAST_PALETTE[0]
    palette = HIGH_CONTRAST_PALETTE
    return palette[1 + ((idx - 1) % (len(palette) - 1))]


def _format_props(props) -> str:
    if not isinstance(props, tuple):
        return str(props)
    if len(props) != 4:
        return str(props)
    return "(" + ", ".join(f"{v:.6f}" for v in props) + ")"


def _log_material_colors(color_map: dict[int, tuple[float, float, float]]):
    materials = sorted(bpy.data.materials, key=lambda m: m.pass_index)
    print("[Segmentation] Material color map:")
    for material in materials:
        key = vision_addon.get_material_property_key(material, precision=6)
        color = _color_for_id(int(material.pass_index), color_map)
        print(
            f"  idx={material.pass_index:3d} color={color} name={material.name} props={_format_props(key)}"
        )


def _rgb_to_hex(color: tuple[float, float, float]) -> str:
    r = max(0, min(255, int(round(color[0] * 255))))
    g = max(0, min(255, int(round(color[1] * 255))))
    b = max(0, min(255, int(round(color[2] * 255))))
    return f"#{r:02X}{g:02X}{b:02X}"


def _write_material_color_log(path: Path, color_map: dict[int, tuple[float, float, float]]):
    materials = sorted(bpy.data.materials, key=lambda m: m.pass_index)
    lines = ["idx\tname\tproperties\thex"]
    for material in materials:
        key = vision_addon.get_material_property_key(material, precision=6)
        color = _color_for_id(int(material.pass_index), color_map)
        hex_color = _rgb_to_hex(color)
        lines.append(
            f"{material.pass_index}\t{material.name}\t{_format_props(key)}\t{hex_color}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_material_color_map() -> dict[int, tuple[float, float, float]]:
    vision_addon.assign_material_indexes_by_properties(precision=6)
    color_map = {}
    for material in bpy.data.materials:
        idx = int(material.pass_index)
        color_map[idx] = _color_for_id(idx, color_map)
    return color_map


def _build_collection_color_map() -> tuple[dict[str, int], dict[int, tuple[float, float, float]]]:
    collection_to_index = vision_addon.assign_object_indexes_by_collection()
    color_map = {}
    for idx in sorted(collection_to_index.values()):
        color_map[idx] = _color_for_id(idx, color_map)
    return collection_to_index, color_map


def _write_collection_color_log(
    path: Path,
    collection_to_index: dict[str, int],
    color_map: dict[int, tuple[float, float, float]],
):
    lines = ["idx\tname\thex"]
    for name, idx in sorted(collection_to_index.items(), key=lambda item: item[1]):
        color = _color_for_id(int(idx), color_map)
        hex_color = _rgb_to_hex(color)
        lines.append(f"{idx}\t{name}\t{hex_color}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_segmentation_from_npz(
    npz_src: Path,
    gt_npz_dir: Path,
    gt_seg_dir: Path,
    color_map,
    seg_key: str = "segmentation_masks",
    use_id_color: bool = False,
    seg_suffix: str = "Seg",
):
    npz_src = Path(npz_src)
    gt_npz_dir.mkdir(parents=True, exist_ok=True)
    gt_seg_dir.mkdir(parents=True, exist_ok=True)

    npz_dst = gt_npz_dir / npz_src.name
    if npz_dst.exists():
        npz_src = npz_dst
    elif npz_dst.resolve() != npz_src.resolve():
        try:
            npz_src.replace(npz_dst)
        except Exception:
            shutil.copy2(npz_src, npz_dst)
            npz_src.unlink(missing_ok=True)
        npz_src = npz_dst

    data = np.load(npz_src, allow_pickle=True)
    if seg_key not in data:
        return None

    base = npz_src.stem
    seg = np.rint(data[seg_key]).astype(np.int32)
    if use_id_color:
        rgb = _id_to_color(seg)
    else:
        max_id = int(seg.max()) if seg.size else 0
        colors = np.zeros((max_id + 1, 3), dtype=np.float32)
        for idx in range(max_id + 1):
            colors[idx] = _color_for_id(idx, color_map)
        rgb = colors[seg]
    plt.imsave(str(gt_seg_dir / f"{base}_{seg_suffix}.png"), rgb)
    return np.unique(seg)


def main() -> None:
    blend_name = _pick_blend_name()
    blend_path = _resolve_blend_path(blend_name)
    if not blend_path.exists():
        raise FileNotFoundError(f"Blend file not found: {blend_path}")

    scene, cam, sun = _setup_scene(blend_path)
    _configure_segmentation(scene)

    output_root = ensure_dir(PROJECT_ROOT / "renders" / get_timestamp_folder() / f"{blend_path.stem}_SEG")
    material_color_map = _build_material_color_map()
    _log_material_colors(material_color_map)
    _write_material_color_log(output_root / "material_color_log.txt", material_color_map)
    collection_to_index, collection_color_map = _build_collection_color_map()
    _write_collection_color_log(
        output_root / "collection_color_log.txt",
        collection_to_index,
        collection_color_map,
    )
    traj_path = output_root / "camera_traj.txt"
    write_camera_trajectory_v2(
        str(traj_path),
        N=NUM_FRAMES,
        R_LEO=R_LEO,
        R_RPO=R_RPO,
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

    _hide_non_targets(target, set())

    gt_root = ensure_dir(output_root / "GTAnnotations")
    gt_dirs = {
        "gt_npz": ensure_dir(gt_root / "NPZ"),
        "gt_depth": ensure_dir(gt_root / "Depth"),
        "gt_norm": ensure_dir(gt_root / "Normal"),
        "gt_flow": ensure_dir(gt_root / "Flow"),
        "gt_seg": ensure_dir(gt_root / "Seg"),
    }
    npz_tmp_dir = ensure_dir(output_root / "_npz_tmp")

    for i, fr in enumerate(frames):
        target.location = fr["p_G_I"]
        target.rotation_mode = "QUATERNION"
        target.rotation_quaternion = fr["q_I_G"]

        cam.location = fr["p_C_I"]
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = fr["q_I_C"]

        set_sun_direction(sun, SUN_AZ_DEG, SUN_EL_DEG)
        scene.frame_set(i)
        scene.render.filepath = str(npz_tmp_dir / f"frame_{i:04d}")
        bpy.ops.render.render(write_still=False)

        npz_src = npz_tmp_dir / f"{i:04d}.npz"
        if npz_src.exists():
            _save_segmentation_from_npz(
                npz_src,
                gt_dirs["gt_npz"],
                gt_dirs["gt_seg"],
                material_color_map,
                seg_suffix="SegMaterial",
            )
            _save_segmentation_from_npz(
                npz_src,
                gt_dirs["gt_npz"],
                gt_dirs["gt_seg"],
                collection_color_map,
                seg_key="segmentation_masks_collection",
                seg_suffix="SegCollection",
            )


if __name__ == "__main__":
    main()
