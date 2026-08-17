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

    @property
    def _beauty_ext(self) -> str:
        """File extension Blender appends for the configured beauty format."""
        return ".exr" if self.config.render.output_format == "OPEN_EXR" else ".png"

    def setup_total(self):
        self._log_info("Loading scene: %s", self.config.scene_blend_path)
        bpy.ops.wm.open_mainfile(filepath=self.config.scene_blend_path)

        self.scene = bpy.context.scene
        # Use a linear/Standard view transform instead of scene.blend's baked-in
        # AgX. AgX is a cinematic film curve that crushes midtones and rolls off
        # highlights — wrong for a radiometric / feature-extraction render, which
        # should reflect the near-linear sensor response. (2026-06-04)
        self.scene.view_settings.view_transform = "Standard"
        self.world = self.scene.world
        self.scene.render.engine = self.config.render.engine
        self.scene.cycles.samples = self.config.render.samples
        self.scene.render.resolution_x, self.scene.render.resolution_y = self.config.camera.resolution

        # Beauty-pass encoding. Default "PNG" reproduces the historical
        # display-referred output exactly; "OPEN_EXR" taps scene-linear radiance
        # BEFORE the exposure/view-transform/quantization chain above, which is
        # the only correct insertion point for a radiometric sensor model.
        # Order matters: set the format first, then the depth, because the valid
        # color_depth enum is format-dependent (PNG 8/16, OPEN_EXR 16/32).
        img_settings = self.scene.render.image_settings
        img_settings.file_format = self.config.render.output_format
        depth = self.config.render.color_depth
        if depth is None:
            depth = "32" if self.config.render.output_format == "OPEN_EXR" else "8"
        img_settings.color_depth = depth
        self._log_info(
            "Beauty pass: format=%s color_depth=%s (%s)",
            self.config.render.output_format,
            depth,
            "scene-linear, pre-tone-map" if self.config.render.output_format == "OPEN_EXR" else "display-referred",
        )

        # Pin the sampling seed rather than relying on Blender's unstated
        # default, so a re-render is reproducible by declaration.
        self.scene.cycles.seed = self.config.render.cycles_seed
        self.scene.cycles.use_animated_seed = False

        # Denoising. None leaves scene.blend's setting alone (historical
        # behaviour: OIDN ON). Radiometric masters set it False so the beauty
        # pass is un-filtered Monte Carlo radiance rather than a learned
        # estimate of it.
        if self.config.render.use_denoising is not None:
            self.scene.cycles.use_denoising = self.config.render.use_denoising
        self._log_info(
            "Cycles: samples=%d seed=%d denoising=%s",
            self.config.render.samples,
            self.config.render.cycles_seed,
            self.scene.cycles.use_denoising,
        )

        # Centered render-border crop. Cycles renders only the (crop_w x crop_h)
        # window centred on the principal point; the output PNG is exactly
        # (crop_w, crop_h). Camera intrinsics stay full-frame; the crop origin
        # is exposed via render_crop_info() for downstream coordinate mapping.
        crop_px = self.config.render.crop_to_border_px
        full_w, full_h = self.config.camera.resolution
        self._crop_origin_px = (0, 0)
        self._crop_size_px = (full_w, full_h)
        if crop_px is not None:
            crop_w, crop_h = int(crop_px[0]), int(crop_px[1])
            if crop_w <= 0 or crop_h <= 0 or crop_w > full_w or crop_h > full_h:
                raise ValueError(
                    f"crop_to_border_px {crop_px} must be positive and fit within "
                    f"resolution {self.config.camera.resolution}"
                )
            cx, cy = full_w / 2.0, full_h / 2.0
            # Nudge the max bounds by 0.25 pixels to defeat Blender's float-to-pixel
            # truncation: when border_max*W lands exactly on an integer pixel boundary,
            # IEEE 754 representation can place the result a fraction below the integer
            # and Blender drops the last column/row (we observed 255 px output for a
            # 256 px request at 5472 wide). 0.25 px is comfortably below the next pixel
            # boundary regardless of whether Blender uses floor/ceil/round internally.
            x0 = (cx - crop_w / 2.0) / full_w
            x1 = (cx + crop_w / 2.0 + 0.25) / full_w
            # Blender's border y axis is bottom-origin; vertical centring is
            # symmetric, so the same expression works regardless.
            y0 = (cy - crop_h / 2.0) / full_h
            y1 = (cy + crop_h / 2.0 + 0.25) / full_h
            self.scene.render.use_border = True
            self.scene.render.use_crop_to_border = True
            self.scene.render.border_min_x = x0
            self.scene.render.border_max_x = x1
            self.scene.render.border_min_y = y0
            self.scene.render.border_max_y = y1
            self._crop_origin_px = (int(round((cx - crop_w / 2.0))), int(round((cy - crop_h / 2.0))))
            self._crop_size_px = (crop_w, crop_h)
            self._log_info(
                "Render crop enabled: %dx%d centred at (%d, %d) in full %dx%d frame; "
                "border=(x:[%.4f, %.4f], y:[%.4f, %.4f])",
                crop_w, crop_h, int(cx), int(cy), full_w, full_h, x0, x1, y0, y1,
            )
        else:
            self.scene.render.use_border = False
            self.scene.render.use_crop_to_border = False

        # Expose crop window to the GT addon via scene custom properties so its
        # depth/normal/seg/optical-flow outputs can be sliced to match the saved
        # RGB crop instead of dumping full-sensor (5472x3648) annotations.
        self.scene["_sisifos_crop_origin_x"] = int(self._crop_origin_px[0])
        self.scene["_sisifos_crop_origin_y"] = int(self._crop_origin_px[1])
        self.scene["_sisifos_crop_size_x"] = int(self._crop_size_px[0])
        self.scene["_sisifos_crop_size_y"] = int(self._crop_size_px[1])

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
        rl.location = (-600, 0)

        comp = c_nodes.new("CompositorNodeComposite")
        comp.location = (600, 0)

        # Build the chain RL -> [PSF blur] -> [Glare] -> Composite. PSF blur is
        # placed before glare so the optical PSF is applied to the radiometric
        # image before any saturation-driven post effects.
        prev_socket = rl.outputs["Image"]

        combined_fwhm_px = self.config.camera.combined_psf_fwhm_px
        if combined_fwhm_px is not None and combined_fwhm_px > 0.0:
            sigma_px = combined_fwhm_px / 2.355
            kernel_radius = max(1, int(math.ceil(3.0 * sigma_px)))
            blur = c_nodes.new("CompositorNodeBlur")
            blur.filter_type = "GAUSS"
            blur.use_relative = False
            blur.size_x = kernel_radius
            blur.size_y = kernel_radius
            blur.use_extended_bounds = True
            blur.location = (-300, 0)
            c_links.new(prev_socket, blur.inputs["Image"])
            prev_socket = blur.outputs["Image"]
            self._log_info(
                "PSF compositor blur enabled: combined FWHM=%.3f px, sigma=%.3f px, kernel_radius=%d px",
                combined_fwhm_px,
                sigma_px,
                kernel_radius,
            )

        glare = c_nodes.new("CompositorNodeGlare")
        glare.location = (0, 0)
        if str(self.config.setup.enable_glare).casefold() == "on":
            glare.glare_type = "FOG_GLOW"
            glare.quality = "HIGH"
            glare.threshold = self.config.setup.glare_threshold
            glare.mix = 0.5
            glare.size = self.config.setup.glare_size
            c_links.new(prev_socket, glare.inputs["Image"])
            prev_socket = glare.outputs["Image"]

        c_links.new(prev_socket, comp.inputs["Image"])

        # Illumination-support passes, tapped from Render Layers DIRECTLY --
        # deliberately upstream of the PSF blur and glare wired above. The blur
        # spreads light ~1 px into geometrically shadowed pixels, so a mask
        # derived from the composited beauty pass would count that optical
        # bleed as "illuminated". These sockets carry the light actually
        # delivered to the surface.
        self._light_pass_node = None
        if self.config.render.save_light_passes:
            view_layer = self.scene.view_layers[0]
            view_layer.use_pass_diffuse_direct = True
            view_layer.use_pass_glossy_direct = True
            lp = c_nodes.new("CompositorNodeOutputFile")
            lp.name = "output_light_passes"
            lp.label = "output_light_passes"
            lp.location = (600, -400)
            lp.format.file_format = "OPEN_EXR"
            lp.format.color_depth = "32"
            lp.format.color_mode = "RGB"
            lp.file_slots.clear()
            for socket_name in ("DiffDir", "GlossDir"):
                lp.file_slots.new(socket_name)
                c_links.new(rl.outputs[socket_name], lp.inputs[socket_name])
            self._light_pass_node = lp
            self._log_info("Light passes enabled: DiffDir + GlossDir (pre-PSF, pre-glare)")

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
        sensor_w_mm = self.config.camera.effective_sensor_width_mm
        cam.data.sensor_width = sensor_w_mm
        cam.data.sensor_fit = self.config.camera.sensor_fit
        cam.data.clip_start = self.config.camera.clip_start
        cam.data.clip_end = self.config.camera.clip_end

        # Depth-of-field. Both `f_number` and `dof_focus_distance_m` must be
        # provided for Cycles to actually render defocus; setting just one is
        # ambiguous so we keep DoF off and record the f/# only via the camera
        # data block for downstream provenance.
        if self.config.camera.f_number is not None:
            cam.data.dof.aperture_fstop = float(self.config.camera.f_number)
            if self.config.camera.dof_focus_distance_m is not None:
                cam.data.dof.use_dof = True
                cam.data.dof.focus_distance = float(self.config.camera.dof_focus_distance_m)
                self._log_info(
                    "Cycles DoF enabled: aperture_fstop=%.3f, focus_distance=%.3f m",
                    cam.data.dof.aperture_fstop,
                    cam.data.dof.focus_distance,
                )
            else:
                cam.data.dof.use_dof = False
                self._log_info(
                    "f_number=%.3f recorded on camera; DoF rendering disabled "
                    "(set camera.dof_focus_distance_m to enable Cycles defocus).",
                    cam.data.dof.aperture_fstop,
                )

        # Derived camera metrics for sanity-checking the spec.
        res_x, res_y = self.config.camera.resolution
        focal_mm = self.config.camera.focal_length
        fov_x_deg = math.degrees(2.0 * math.atan((sensor_w_mm * 0.5) / focal_mm))
        sensor_h_mm = sensor_w_mm * res_y / res_x
        fov_y_deg = math.degrees(2.0 * math.atan((sensor_h_mm * 0.5) / focal_mm))
        sensor_diag_mm = math.sqrt(sensor_w_mm**2 + sensor_h_mm**2)
        fov_diag_deg = math.degrees(2.0 * math.atan((sensor_diag_mm * 0.5) / focal_mm))
        ifov_per_px_urad = (sensor_w_mm / focal_mm / res_x) * 1e6
        diff_fwhm_px = self.config.camera.diffraction_fwhm_px
        aberr_fwhm_px = self.config.camera.psf_fwhm_px
        combined_fwhm_px = self.config.camera.combined_psf_fwhm_px
        self._log_info(
            "Camera optics summary: focal=%.3f mm | sensor=%.3f x %.3f mm (diag %.3f mm) "
            "| res=%dx%d | FOV(h/v/diag)=%.3f / %.3f / %.3f deg | IFOV=%.3f urad/px",
            focal_mm,
            sensor_w_mm,
            sensor_h_mm,
            sensor_diag_mm,
            res_x,
            res_y,
            fov_x_deg,
            fov_y_deg,
            fov_diag_deg,
            ifov_per_px_urad,
        )
        self._log_info(
            "Camera PSF summary: aberration FWHM=%s px | diffraction FWHM=%s px | combined FWHM=%s px",
            f"{aberr_fwhm_px:.3f}" if aberr_fwhm_px is not None else "n/a",
            f"{diff_fwhm_px:.3f}" if diff_fwhm_px is not None else "n/a",
            f"{combined_fwhm_px:.3f}" if combined_fwhm_px is not None else "n/a",
        )

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
        """Remove RF_* roots (and descendants) and any orphan spacecraft debris.

        scene.blend can ship with stale spacecraft meshes parented under the
        bare `Target` empty (e.g. INTEGRAL_polySurface25_SPI from a prior
        session). Those meshes sit at the world origin and overflow the camera
        FOV at typical RPO ranges, masking distance-driven scaling. We clean
        them out here before any new spacecraft is appended.
        """
        to_remove = set()

        for o in bpy.data.objects:
            if o.parent is None and o.name.startswith("RF_"):
                to_remove.add(o)
                to_remove.update(o.children_recursive)

        target_empty = bpy.data.objects.get("Target")
        if target_empty is not None:
            for child in list(target_empty.children_recursive):
                to_remove.add(child)

        if to_remove:
            remove_objects_from_scene(list(to_remove))
            self._log_info("Removed %d stale spacecraft objects from scene", len(to_remove))

    def get_models_in_blend(self) -> list[str]:
        """Inspect the blend file and return RF_* root names to render (without loading)."""
        all_names = self._get_target_blend_object_names()
        rf_names = sorted([n for n in all_names if n.startswith("RF_")], key=str.lower)
        return rf_names

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

        scale_xyz = tuple(float(s) for s in self.config.model_scale_xyz)
        if scale_xyz != (1.0, 1.0, 1.0):
            if any(s <= 0.0 for s in scale_xyz):
                raise ValueError(
                    f"model_scale_xyz must be strictly positive, got {scale_xyz}"
                )
            root.scale = scale_xyz
            bpy.context.view_layer.update()
            self._log_info(
                "Applied model_scale_xyz to '%s': (%.4f, %.4f, %.4f)",
                model_name,
                *scale_xyz,
            )

        self._log_info("Loaded spacecraft '%s' (%d objects)", model_name, 1 + len(list(root.children_recursive)))
        return root

    def get_all_models(self) -> list[bpy.types.Object]:
        """Get all RF_* models."""
        return [o for o in bpy.data.objects if o.parent is None and o.name.startswith("RF_")]

    def render_crop_info(self) -> dict:
        """Crop + intrinsics metadata for downstream pixel-coordinate mapping.

        Returns a dict that fully describes how the cropped output relates to
        the full-frame camera model: full-frame resolution, crop origin/size in
        full-frame pixels, and full-frame intrinsics (focal length in mm and
        pixels). When no crop is active, crop_origin_px = (0, 0) and
        crop_size_px = full-frame resolution.
        """
        full_w, full_h = self.config.camera.resolution
        return {
            "full_frame_resolution_px": [int(full_w), int(full_h)],
            "crop_origin_px": [int(self._crop_origin_px[0]), int(self._crop_origin_px[1])],
            "crop_size_px": [int(self._crop_size_px[0]), int(self._crop_size_px[1])],
            "crop_principal_point_in_crop_px": [
                full_w / 2.0 - self._crop_origin_px[0],
                full_h / 2.0 - self._crop_origin_px[1],
            ],
            "focal_length_mm": float(self.config.camera.focal_length),
            "focal_length_px": float(self.config.camera.focal_length_px),
            "sensor_width_mm": float(self.config.camera.effective_sensor_width_mm),
            "pixel_size_um": (
                float(self.config.camera.pixel_size_um)
                if self.config.camera.pixel_size_um is not None
                else float(self.config.camera.effective_sensor_width_mm)
                * 1000.0
                / float(full_w)
            ),
        }

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
        # File Output nodes write <base_path>/<slot><frame>.exr on their own, so
        # they only need the directory; the frame number follows frame_set().
        if getattr(self, "_light_pass_node", None) is not None:
            self._light_pass_node.base_path = str(output_dir.resolve() / "LightPasses")
        self.scene.frame_set(frame_id)
        bpy.ops.render.render(write_still=True)

        # Restore exposure
        self.scene.view_settings.exposure = base_ev

        return f"frame_{stem}{self._beauty_ext}"

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
        if getattr(self, "_light_pass_node", None) is not None:
            self._light_pass_node.base_path = str(output_dir.resolve() / "LightPasses")
        bpy.ops.render.render(write_still=True)

        return f"frame_{stem}{self._beauty_ext}"
