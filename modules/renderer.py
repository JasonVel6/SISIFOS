import importlib.util
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion

from .blender_utils import append_blend_objects, clear_anim, keyframe_pose, scale_object_by_factor, set_sun_direction
from .config import SceneConfig
from .io_utils import vprint


class BlenderRenderer:
    """Main renderer class for image generation."""

    def __init__(self, config: SceneConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.scene = bpy.context.scene
        self.world = self.scene.world

    def setup_total(self):
        vprint(f"Loading scene: {self.config.scene_blend_path}", self.verbose)
        bpy.ops.wm.open_mainfile(filepath=self.config.scene_blend_path)

        self.scene = bpy.context.scene
        self.world = self.scene.world
        self.scene.render.engine = self.config.render.engine
        self.scene.cycles.samples = self.config.render.samples
        self.scene.render.resolution_x, self.scene.render.resolution_y = self.config.camera.resolution

        new_objects = append_blend_objects(self.config.objects["Earth"].blend_path)
        new_objects2 = append_blend_objects(self.config.objects["Target"].blend_path)

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
            print("Loaded stars HDRI:", self.config.hdri_path)

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
        vb.bool_save_gt_data = True
        vb.bool_save_depth = True
        vb.bool_save_normals = True
        vb.bool_save_cam_param = True
        vb.bool_save_opt_flow = True  # needs Cycles' Vector pass
        vb.bool_save_segmentation_masks = True  # needs object pass_index > 0
        vb.bool_save_obj_poses = True

        vprint("Vision Blender addon configured", self.verbose)
        cam = bpy.data.objects.get("Camera")
        cam.rotation_mode = "QUATERNION"
        cam.data.lens = self.config.camera.focal_length
        cam.data.sensor_width = self.config.camera.sensor_width
        cam.data.clip_start = self.config.camera.clip_start
        cam.data.clip_end = self.config.camera.clip_end

        earth = bpy.data.objects["Earth"]
        clouds = bpy.data.objects["Clouds"]
        atmo = bpy.data.objects["Atmo"]
        target = bpy.data.objects["Target"]
        sun = bpy.data.objects["Sun"]  # or create one
        bpy.context.view_layer.update()
        sun.data.energy = 10.0
        scale_object_by_factor(earth, self.config.objects["Earth"].scale_factor)
        scale_object_by_factor(clouds, self.config.objects["Clouds"].scale_factor)
        scale_object_by_factor(atmo, self.config.objects["Atmo"].scale_factor)
        return cam, sun

    def select_model_to_render(self) -> bpy.types.Object:
        """Get RF_* model to render."""

        models = [o for o in bpy.data.objects if o.parent is None and o.name.startswith("RF_")]
        selected_model = None
        for model in models:
            if model.name == self.config.selected_model:
                selected_model = model

        if not selected_model:
            raise ValueError(f"Selected model '{self.config.selected_model}' not found among RF_* objects.")

        return selected_model

    def get_all_models(self) -> list[bpy.types.Object]:
        """Get all RF_* models."""
        return [o for o in bpy.data.objects if o.parent is None and o.name.startswith("RF_")]

    def hide_all_except(self, target_root, all_roots):
        """Hide all models except target."""
        for r in all_roots:
            hide = r != target_root
            r.hide_render = hide
            for c in r.children_recursive:
                c.hide_render = hide
        bpy.context.view_layer.update()

    def rotate_z(self, obj, deg: float):
        """Rotate object around local Z axis."""
        obj.rotation_mode = "QUATERNION"
        q_rot = Quaternion((0, 0, 1), math.radians(deg))
        obj.rotation_quaternion = q_rot @ obj.rotation_quaternion
        bpy.context.view_layer.update()

    @staticmethod
    def _are_contiguous(frame_ids: list[int]) -> bool:
        """Return True if *frame_ids* form a contiguous range min … max."""
        if not frame_ids:
            return False
        return list(range(min(frame_ids), max(frame_ids) + 1)) == sorted(frame_ids)

    def _keyframe_sun_direction(self, sun: bpy.types.Object, sun_az: float, sun_el: float, frame: int) -> None:
        """Set the sun direction and insert a keyframe for it."""
        set_sun_direction(sun, sun_az, sun_el)
        sun.keyframe_insert(data_path="rotation_quaternion", frame=frame)

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
            model.rotation_quaternion = Quaternion(tuple(fdata["q_I_G"]))

            cam.location = fdata["p_C_I"]
            cam.rotation_quaternion = Quaternion(tuple(fdata["q_I_C"]))

            bpy.context.view_layer.update()

            keyframe_pose(model, fid)
            keyframe_pose(cam, fid)
            self._keyframe_sun_direction(sun, fdata["sun_az"], fdata["sun_el"], fid)

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
