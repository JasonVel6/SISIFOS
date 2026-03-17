import importlib.util
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Quaternion

from .blender_utils import (
    append_blend_objects,
    append_blend_objects_filtered,
    clear_anim,
    keyframe_pose,
    list_blend_object_names,
    remove_objects_from_scene,
    scale_object_by_factor,
    set_sun_direction,
)
from .config import SceneConfig
from .log_utils import get_logger


class BlenderRenderer:
    """Main renderer class for image generation."""

    def __init__(self, config: SceneConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.scene = bpy.context.scene
        self.world = self.scene.world
        self._target_blend_object_names = None
        self.logger = get_logger()

        # Calculate rotation quaternions for spacecraft models based on config defaults
        if not config.model_rotation_A_model_euler:
            raise ValueError("model_rotation_A_model_euler must be provided in config for each spacecraft")
        euler_A_model = Euler(
            (
                math.radians(float(config.model_rotation_A_model_euler[0])),
                math.radians(float(config.model_rotation_A_model_euler[1])),
                math.radians(float(config.model_rotation_A_model_euler[2])),
            ),
            "XYZ",
        )
        self.quat_A_model = euler_A_model.to_quaternion()
        self._log_info(
            "Initialized BlenderRenderer with model rotation (Euler XYZ in degrees): %s",
            config.model_rotation_A_model_euler,
        )
        self._log_info(
            "Calculated model rotation quaternion: (w: %.6f, x: %.6f, y: %.6f, z: %.6f)",
            self.quat_A_model.w,
            self.quat_A_model.x,
            self.quat_A_model.y,
            self.quat_A_model.z,
        )

    def _log_info(self, message: str, *args):
        if self.verbose:
            self.logger.info(message, *args)

    def _set_pass_index_recursive(self, root_name: str, pass_index: int) -> None:
        root = bpy.data.objects.get(root_name)
        if root is None:
            self._log_info("Segmentation pass index skipped; object '%s' not found", root_name)
            return

        root.pass_index = pass_index
        for child in root.children_recursive:
            child.pass_index = pass_index

        self._log_info(
            "Segmentation pass_index=%d assigned to '%s' (+ %d children)",
            pass_index,
            root_name,
            len(list(root.children_recursive)),
        )

    def setup_total(self):
        self._log_info("Loading scene: %s", self.config.scene_blend_path)
        bpy.ops.wm.open_mainfile(filepath=self.config.scene_blend_path)

        self.scene = bpy.context.scene
        self.world = self.scene.world
        self.scene.render.engine = self.config.render.engine
        self.scene.cycles.samples = self.config.render.samples
        self.scene.render.resolution_x, self.scene.render.resolution_y = self.config.camera.resolution

        # Enable GPU rendering using the best available Cycles backend.
        if self.config.render.engine == "CYCLES":
            try:
                # Reuse scene/mesh data across frames for faster sequence renders.
                self.scene.render.use_persistent_data = True
                prefs = bpy.context.preferences.addons["cycles"].preferences
                # Prefer OptiX, fallback to CUDA.
                selected_backend = None
                gpu_found = False
                for backend in ("OPTIX", "CUDA"):
                    try:
                        # Refresh twice; Blender 4.x sometimes needs a second
                        # get_devices() after open_mainfile to populate devices.
                        prefs.compute_device_type = backend
                        prefs.get_devices()
                        prefs.compute_device_type = backend
                        prefs.get_devices()
                    except Exception:
                        continue

                    for device in prefs.devices:
                        self._log_info(
                            "  [GPU probe] device: %s, type: %s, use: %s",
                            device.name,
                            device.type,
                            device.use,
                        )

                    backend_devices = [d for d in prefs.devices if d.type == backend]
                    if backend_devices:
                        for device in prefs.devices:
                            device.use = device.type == backend
                        selected_backend = backend
                        gpu_found = True
                        break

                if gpu_found:
                    self.scene.cycles.device = "GPU"
                    self._log_info("Cycles rendering on GPU (%s)", selected_backend)
                else:
                    self._log_info("No OPTIX/CUDA GPU found, using CPU rendering")

                # Confirm final state
                self._log_info("  [GPU confirm] scene.cycles.device = %s", self.scene.cycles.device)
                self._log_info("  [GPU confirm] compute_device_type = %s", prefs.compute_device_type)
                for device in prefs.devices:
                    self._log_info(
                        "  [GPU confirm] %s: type=%s, use=%s",
                        device.name,
                        device.type,
                        device.use,
                    )
            except Exception as e:
                self.logger.exception("GPU setup failed: %s", e)

        append_blend_objects(self.config.objects["Earth"].blend_path)

        addon_path = os.path.join(os.path.dirname(__file__), "addon_ground_truth_generation.py")

        spec = importlib.util.spec_from_file_location("vision_blender_addon", addon_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vision_blender_addon"] = mod
        spec.loader.exec_module(mod)
        mod.register()  # installs scene.vision_blender and render handlers
        vb = self.scene.vision_blender
        world = bpy.context.scene.world
        world.use_nodes = True

        # Clear existing nodes
        nodes = world.node_tree.nodes
        nodes.clear()
        links = world.node_tree.links

        if str(self.config.setup.stars_mode).casefold() == "off":
            bg = nodes.new(type="ShaderNodeBackground")
            bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # black
            bg.inputs[1].default_value = 1.0
            out = nodes.new(type="ShaderNodeOutputWorld")
            links.new(bg.outputs["Background"], out.inputs["Surface"])
        elif str(self.config.setup.stars_mode).casefold() == "on":
            if not os.path.isfile(bpy.path.abspath(self.config.hdri_path)):
                raise FileNotFoundError(bpy.path.abspath(self.config.hdri_path))

            img = bpy.data.images.load(bpy.path.abspath(self.config.hdri_path), check_existing=True)

            env_tex = nodes.new("ShaderNodeTexEnvironment")
            env_tex.image = img
            env_tex.image.colorspace_settings.name = "Non-Color"
            background = nodes.new(type="ShaderNodeBackground")
            background.inputs[1].default_value = 1.0  # Strength (increase if needed)
            output = nodes.new(type="ShaderNodeOutputWorld")
            links.new(env_tex.outputs["Color"], background.inputs["Color"])
            links.new(background.outputs["Background"], output.inputs["Surface"])
            self.logger.info("Loaded stars HDRI: %s", self.config.hdri_path)

        def set_earth_visibility(enable: bool):
            for name in ["Earth", "Clouds", "Atmo"]:
                obj = bpy.data.objects.get(name)
                if obj:
                    obj.hide_render = not enable

        if str(self.config.setup.earth_mode).casefold() == "off":
            set_earth_visibility(False)

        ## GLARAKI
        self.scene.use_nodes = True
        c_tree = self.scene.node_tree
        c_nodes = c_tree.nodes
        c_links = c_tree.links
        c_nodes.clear()

        rl = c_nodes.new("CompositorNodeRLayers")
        rl.location = (-300, 0)

        comp = c_nodes.new("CompositorNodeComposite")
        comp.location = (300, 0)

        glare = c_nodes.new("CompositorNodeGlare")
        if str(self.config.setup.enable_glare).casefold() == "on":
            glare.location = (0, 0)
            glare.glare_type = "FOG_GLOW"
            glare.quality = "HIGH"
            glare.threshold = self.config.setup.glare_threshold
            glare.mix = 0.5
            glare.size = self.config.setup.glare_size

            c_links.new(rl.outputs["Image"], glare.inputs["Image"])
            c_links.new(glare.outputs["Image"], comp.inputs["Image"])
        else:
            c_links.new(rl.outputs["Image"], comp.inputs["Image"])
        vb = self.scene.vision_blender
        vb.bool_save_depth = self.config.save_depth
        vb.bool_save_normals = self.config.save_normals
        vb.bool_save_cam_param = True
        vb.bool_save_opt_flow = self.config.save_optical_flow
        vb.bool_save_segmentation_masks = self.config.save_segmentation
        vb.bool_save_obj_poses = self.config.save_obj_poses
        vb.bool_save_gt_data = any(
            [
                vb.bool_save_depth,
                vb.bool_save_normals,
                vb.bool_save_cam_param,
                vb.bool_save_opt_flow,
                vb.bool_save_segmentation_masks,
                vb.bool_save_obj_poses,
            ]
        )

        # Cycles' IndexOB pass reads pass_index per object, not via parent inheritance.
        self._set_pass_index_recursive("Target", 1)
        for pass_index, object_name in enumerate(["Earth", "Clouds", "Atmo"], start=2):
            self._set_pass_index_recursive(object_name, pass_index)

        self._log_info("Vision Blender addon configured")
        cam = bpy.data.objects.get("Camera")
        cam.rotation_mode = "QUATERNION"
        cam.data.lens = self.config.camera.focal_length
        cam.data.sensor_width = self.config.camera.sensor_width
        cam.data.clip_start = self.config.camera.clip_start
        cam.data.clip_end = self.config.camera.clip_end

        earth = bpy.data.objects["Earth"]
        clouds = bpy.data.objects["Clouds"]
        atmo = bpy.data.objects["Atmo"]
        sun = bpy.data.objects["Sun"]  # or create one
        bpy.context.view_layer.update()
        sun.data.energy = 10.0
        scale_object_by_factor(earth, self.config.objects["Earth"].scale_factor)
        scale_object_by_factor(clouds, self.config.objects["Clouds"].scale_factor)
        scale_object_by_factor(atmo, self.config.objects["Atmo"].scale_factor)
        return cam, sun

    def _get_target_blend_object_names(self) -> list[str]:
        """Cache object-name scan of target blend to avoid repeated library loads."""
        if self._target_blend_object_names is None:
            blend_path = self.config.objects["Target"].blend_path
            self._target_blend_object_names = list_blend_object_names(blend_path)
        return self._target_blend_object_names

    def _remove_existing_spacecraft(self) -> None:
        """Remove RF_* roots (and descendants) already present in the current scene."""
        roots = [o for o in bpy.data.objects if o.parent is None and o.name.startswith("RF_")]
        if not roots:
            return
        to_remove = set()
        for root in roots:
            to_remove.add(root)
            to_remove.update(root.children_recursive)
        remove_objects_from_scene(list(to_remove))

    def load_spacecraft(self, model_name: str) -> bpy.types.Object:
        """Load a single spacecraft (root + descendants) into the scene and return the root."""
        blend_path = self.config.objects["Target"].blend_path
        all_names = self._get_target_blend_object_names()
        rf_names = {n for n in all_names if n.startswith("RF_")}
        if model_name not in rf_names:
            raise ValueError(
                f"Selected model '{model_name}' is not a valid RF_* root in '{blend_path}'. "
                f"Available: {sorted(rf_names)}"
            )

        # Ensure scene starts with zero spacecraft roots.
        self._remove_existing_spacecraft()
        names_to_load = [model_name] + [n for n in all_names if not n.startswith("RF_")]
        loaded_objs = append_blend_objects_filtered(blend_path, names_to_load)

        root = bpy.data.objects.get(model_name)
        if root is None:
            raise RuntimeError(f"Spacecraft root '{model_name}' not found after append")

        # Keep only root and its actual descendants; remove orphans
        keep = set([root] + list(root.children_recursive))
        orphans = [o for o in loaded_objs if o not in keep]
        if orphans:
            remove_objects_from_scene(orphans)

        # Hard guard: keep exactly one RF_* root in scene.
        rf_roots_in_scene = self.get_all_models()
        extras = [o for o in rf_roots_in_scene if o.name != model_name]
        if extras:
            to_remove = set()
            for extra in extras:
                to_remove.add(extra)
                to_remove.update(extra.children_recursive)
            remove_objects_from_scene(list(to_remove))
            rf_roots_in_scene = self.get_all_models()
        if len(rf_roots_in_scene) != 1 or rf_roots_in_scene[0].name != model_name:
            raise RuntimeError(
                f"Expected exactly one loaded spacecraft root '{model_name}', found: "
                f"{[o.name for o in rf_roots_in_scene]}"
            )

        self._log_info("Loaded spacecraft '%s' (%d objects)", model_name, 1 + len(list(root.children_recursive)))
        return root

    def get_all_models(self) -> list[bpy.types.Object]:
        """Get all RF_* models."""
        return [o for o in bpy.data.objects if o.parent is None and o.name.startswith("RF_")]

    def select_model_to_render(self) -> bpy.types.Object:
        """Select currently loaded model by name, else load it."""
        for model in self.get_all_models():
            if model.name == self.config.selected_model:
                return model
        return self.load_spacecraft(self.config.selected_model)

    def hide_all_except(self, target_root: bpy.types.Object, all_roots: list[bpy.types.Object]) -> None:
        """Hide all RF_* roots except the target root (and their descendants)."""
        for root in all_roots:
            hide = root != target_root
            root.hide_render = hide
            for child in root.children_recursive:
                child.hide_render = hide
        bpy.context.view_layer.update()

    def rotate_z(self, obj: bpy.types.Object, deg: float) -> None:
        """Apply an additional local-Z rotation to the model."""
        obj.rotation_mode = "QUATERNION"
        q_rot = Quaternion((0, 0, 1), math.radians(deg))
        obj.rotation_quaternion = q_rot @ obj.rotation_quaternion
        bpy.context.view_layer.update()

    def render_frame_v2(
        self,
        cam: bpy.types.Object,
        model: bpy.types.Object,
        sun: bpy.types.Object,
        frame_dict: dict,
        frame_id: int,
        output_dir: Path,
        exposure_time_s: float,
        N_digits: int,
    ) -> str:
        """
        Render single frame using INERTIAL FRAME trajectory data.

        frame_dict contains:
          p_G_I: position of target in inertial frame
          q_I_G: orientation of target relative to inertial
          p_C_I: position of camera in inertial frame
          q_I_C: orientation of camera relative to inertial
          sun_az, sun_el: sun angles

        Placement strategy:
          - Earth/atmosphere: fixed at origin (0,0,0) with fixed orientation
          - Target (model): p_G_I position, q_I_G orientation
          - Camera: p_C_I position, q_I_C orientation (or look-at)
        """
        p_G_I = frame_dict["p_G_I"]
        q_I_G = frame_dict["q_I_G"]
        p_C_I = frame_dict["p_C_I"]
        q_I_C = frame_dict["q_I_C"]
        sun_az = frame_dict["sun_az"]
        sun_el = frame_dict["sun_el"]

        # Apply poses
        model.rotation_mode = "QUATERNION"
        cam.rotation_mode = "QUATERNION"

        # Target (model) pose in inertial frame
        model.location = p_G_I
        model.rotation_quaternion = Quaternion(q_I_G) @ self.quat_A_model
        self.logger.info(model.rotation_quaternion)

        # Camera pose in inertial frame
        cam.location = p_C_I

        # Camera orientation: look-at target OR use q_I_C directly
        # Option 1: Use stored orientation
        cam.rotation_quaternion = q_I_C

        # Option 2: Enforce look-at (uncomment to use)
        # direction = (model.location - cam.location).normalized()
        # quat = direction.to_track_quat('-Z', 'Y')
        # cam.rotation_quaternion = quat
        set_sun_direction(sun, sun_az, sun_el)
        bpy.context.view_layer.update()

        # Debug: Log poses before rendering
        # print("\n" + "="*80)
        # print(f"[Frame {str(frame_id).zfill(N_digits)}] Rendering with exposure {exposure_time_s*1e6:.1f}µs")
        # print("="*80)

        # Model pose
        # print(f"\n[Model] {model.name} (in inertial frame)")
        # print(f"  Position:    ({p_G_I.x:12.6f}, {p_G_I.y:12.6f}, {p_G_I.z:12.6f})")
        # print(f"  Rotation Q:  ({q_I_G.w:8.6f}, {q_I_G.x:8.6f}, {q_I_G.y:8.6f}, {q_I_G.z:8.6f})")
        # print(f"  Distance from origin: {p_G_I.length:.6f} m")

        # Camera pose
        # print(f"\n[Camera] {cam.name} (in inertial frame)")
        # print(f"  Position:    ({p_C_I.x:12.6f}, {p_C_I.y:12.6f}, {p_C_I.z:12.6f})")
        # print(f"  Rotation Q:  ({q_I_C.w:8.6f}, {q_I_C.x:8.6f}, {q_I_C.y:8.6f}, {q_I_C.z:8.6f})")
        # print(f"  Focal length: {cam.data.lens:.2f} mm")

        # Trajectory info
        # print(f"\n[Trajectory Frame {frame_id}] (Inertial Frame Reference)")
        # print(f"  p_G_I (target pos in I):  ({p_G_I.x:12.6f}, {p_G_I.y:12.6f}, {p_G_I.z:12.6f})")
        # print(f"  q_I_G (target orient):    ({q_I_G.w:8.6f}, {q_I_G.x:8.6f}, {q_I_G.y:8.6f}, {q_I_G.z:8.6f})")
        # print(f"  p_C_I (camera pos in I):  ({p_C_I.x:12.6f}, {p_C_I.y:12.6f}, {p_C_I.z:12.6f})")
        # print(f"  q_I_C (camera orient):    ({q_I_C.w:8.6f}, {q_I_C.x:8.6f}, {q_I_C.y:8.6f}, {q_I_C.z:8.6f})")
        # print(f"  Sun azimuth: {sun_az:7.2f}°, elevation: {sun_el:7.2f}°")

        # Render settings
        # print(f"\n[Render Settings]")
        # print(f"  Output:      {self.scene.render.filepath}")
        # print(f"  Resolution:  {self.scene.render.resolution_x}x{self.scene.render.resolution_y}")
        # print(f"  Engine:      {self.scene.render.engine}")
        # if self.scene.render.engine == 'CYCLES':
        #     print(f"  Samples:     {self.scene.cycles.samples}")
        # print("="*80 + "\n")

        # Set exposure
        base_ev = self.scene.view_settings.exposure
        ev_shift = math.log(exposure_time_s / self.config.setup.t_ref_s, 2.0)
        self.scene.view_settings.exposure = base_ev + ev_shift

        # Render
        # exp_tag = f"{int(round(exposure_time_s * 1e6)):08d}us"
        # stem = f"{frame_id:04d}_{exp_tag}_{sun_tag}_{mode_suffix}"
        stem = f"{str(frame_id).zfill(N_digits)}"

        self.scene.render.filepath = str(output_dir.resolve() / f"frame_{stem}")
        self.scene.frame_set(frame_id)
        bpy.ops.render.render(write_still=True)

        # Restore exposure
        self.scene.view_settings.exposure = base_ev

        return f"frame_{stem}.png"

    def render_frame_motion_blur_traj(
        self,
        cam: bpy.types.Object,
        model: bpy.types.Object,
        sun: bpy.types.Object,
        frame_dict1: dict,
        frame_dict2: dict,
        frame_id1: int,
        shutter: float,
        output_dir: Path,
        exposure_time_s: float,
        N_digits: int,
    ) -> str:
        clear_anim(cam)
        clear_anim(model)
        self.scene.frame_start = frame_id1
        self.scene.frame_end = frame_id1 + 1
        self.scene.frame_set(frame_id1)
        sun_az = frame_dict1["sun_az"]
        sun_el = frame_dict1["sun_el"]
        set_sun_direction(sun, sun_az, sun_el)
        bpy.context.view_layer.update()
        base_ev = self.scene.view_settings.exposure
        ev_shift = math.log(exposure_time_s / self.config.setup.t_ref_s, 2.0)
        self.scene.view_settings.exposure = base_ev + ev_shift

        model.rotation_mode = "QUATERNION"
        cam.rotation_mode = "QUATERNION"
        p_G_I = frame_dict1["p_G_I"]
        q_I_G = frame_dict1["q_I_G"]
        p_C_I = frame_dict1["p_C_I"]
        q_I_C = frame_dict1["q_I_C"]
        model.location = p_G_I
        model.rotation_quaternion = Quaternion(q_I_G) @ self.quat_A_model
        cam.location = p_C_I
        cam.rotation_quaternion = q_I_C
        bpy.context.view_layer.update()
        keyframe_pose(model, frame_id1)
        keyframe_pose(cam, frame_id1)
        self.scene.frame_set(frame_id1 + 1)
        p_G_I = frame_dict2["p_G_I"]
        q_I_G = frame_dict2["q_I_G"]
        p_C_I = frame_dict2["p_C_I"]
        q_I_C = frame_dict2["q_I_C"]
        model.location = p_G_I
        model.rotation_quaternion = Quaternion(q_I_G) @ self.quat_A_model
        cam.location = p_C_I
        cam.rotation_quaternion = q_I_C
        bpy.context.view_layer.update()
        keyframe_pose(model, frame_id1 + 1)
        keyframe_pose(cam, frame_id1 + 1)
        self.scene.render.use_motion_blur = True
        self.scene.render.motion_blur_shutter = float(shutter)
        cy = bpy.context.scene.cycles
        if hasattr(cy, "motion_blur_position"):
            cy.motion_blur_position = "START"
        self.scene.frame_set(frame_id1)
        bpy.context.view_layer.update()
        # stem = f"{frame_id:04d}_{exp_tag}_{sun_tag}_{mode_suffix}"
        stem = f"{str(frame_id1).zfill(N_digits)}_blurred"
        self.scene.render.filepath = str(output_dir.resolve() / f"frame_{stem}")
        bpy.ops.render.render(write_still=True)

        return f"frame_{stem}.png"

    def render_animation(
        self,
        cam: bpy.types.Object,
        model: bpy.types.Object,
        sun: bpy.types.Object,
        frames: list[dict],
        frame_ids: list[int],
        output_dir: Path,
        exposure_time_s: float,
        N_digits: int,
    ) -> list[str]:
        """Keyframe the whole trajectory and render every *frame_id*.

        All frames are keyframed up front so Cycles can sample real
        inter-frame motion for physically correct motion blur.
        Frames are rendered individually to preserve per-frame GT
        extraction via the vision_blender addon.

        Returns a list of output PNG filenames.
        """
        if not frame_ids:
            raise ValueError("frame_ids cannot be empty")

        # Clear existing animation data
        clear_anim(cam)
        clear_anim(model)
        clear_anim(sun)

        # Rotation modes
        model.rotation_mode = "QUATERNION"
        cam.rotation_mode = "QUATERNION"
        sun.rotation_mode = "QUATERNION"

        # Keyframe every frame
        for fid in frame_ids:
            fdata = frames[fid]
            self.scene.frame_set(fid)

            model.location = fdata["p_G_I"]
            model.rotation_quaternion = Quaternion(tuple(fdata["q_I_G"])) @ self.quat_A_model

            cam.location = fdata["p_C_I"]
            cam.rotation_quaternion = Quaternion(tuple(fdata["q_I_C"]))

            bpy.context.view_layer.update()

            keyframe_pose(model, fid)
            keyframe_pose(cam, fid)
            set_sun_direction(sun, fdata["sun_az"], fdata["sun_el"])
            sun.keyframe_insert(data_path="rotation_quaternion", frame=fid)

        # Frame range
        self.scene.frame_start = min(frame_ids)
        self.scene.frame_end = max(frame_ids)

        # Per-frame exposure handler
        base_ev = self.scene.view_settings.exposure
        ev_shift = math.log(exposure_time_s / self.config.setup.t_ref_s, 2.0)

        def _exposure_handler(scene, depsgraph=None):
            scene.view_settings.exposure = base_ev + ev_shift

        bpy.app.handlers.frame_change_pre.append(_exposure_handler)

        # Motion blur
        enable_blur = str(self.config.setup.enable_blur).casefold()
        if enable_blur == "on":
            self.scene.render.use_motion_blur = True
            fps = self.scene.render.fps / self.scene.render.fps_base
            shutter_frames = self.config.camera.exposure_time_s * fps * self.config.setup.blur_shutter_factor
            self.scene.render.motion_blur_shutter = float(shutter_frames)
        else:
            self.scene.render.use_motion_blur = False

        #  Output settings
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.use_file_extension = True

        try:
            # Render frame-by-frame with keyframed data in place.
            for fid in frame_ids:
                stem = str(fid).zfill(N_digits)
                self.scene.render.filepath = str(output_dir / f"frame_{stem}")
                self.scene.frame_set(fid)
                bpy.ops.render.render(write_still=True)
        finally:
            if _exposure_handler in bpy.app.handlers.frame_change_pre:
                bpy.app.handlers.frame_change_pre.remove(_exposure_handler)
            self.scene.view_settings.exposure = base_ev

        # Build output filename list
        return [f"frame_{str(fid).zfill(N_digits)}.png" for fid in frame_ids]
