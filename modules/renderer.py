import bpy
import os
import sys
import math
import importlib.util
from pathlib import Path
from typing import List, Dict
from mathutils import Quaternion

from .config import SceneConfig
from .io_utils import vprint
from .blender_utils import (
    append_blend_objects, scale_object_by_factor, set_sun_direction, 
    clear_anim, keyframe_pose
)

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
        if str(self.config.setup.earth_mode).casefold() =="off":
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
        if str(self.config.setup.enable_glare).casefold() =="on":
            glare.location = (0, 0)
            glare.glare_type = 'FOG_GLOW'
            glare.quality = 'HIGH'
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
        vb.bool_save_opt_flow = True               # needs Cycles' Vector pass
        vb.bool_save_segmentation_masks = True     # needs object pass_index > 0
        vb.bool_save_obj_poses = True
            
        vprint("Vision Blender addon configured", self.verbose)
        cam = bpy.data.objects.get("Camera")
        cam.rotation_mode = 'QUATERNION'
        cam.data.lens = self.config.camera.focal_length
        cam.data.clip_start = self.config.camera.clip_start
        cam.data.clip_end = self.config.camera.clip_end
        
        earth  = bpy.data.objects["Earth"]
        clouds = bpy.data.objects["Clouds"]
        atmo   = bpy.data.objects["Atmo"]
        target = bpy.data.objects["Target"]
        sun = bpy.data.objects["Sun"]  # or create one
        bpy.context.view_layer.update()
        sun.data.energy = 10.0
        # TODO this should be parameterized
        scale_object_by_factor(earth,  10)
        scale_object_by_factor(clouds, 10)
        scale_object_by_factor(atmo,   10)
        return cam, sun
    
    def select_models_to_render(self) -> List[bpy.types.Object]:
        """Get list of RF_* models to render."""
        
        models = [o for o in bpy.data.objects
                 if o.parent is None and o.name.startswith("RF_")]
        if self.config.selected_models:
            models = [m for m in models if m.name in self.config.selected_models]
        
        return sorted(models, key=lambda o: o.name.lower())
    
    def hide_all_except(self, target_root, all_roots):
        """Hide all models except target."""
        for r in all_roots:
            hide = (r != target_root)
            r.hide_render = hide
            for c in r.children_recursive:
                c.hide_render = hide
        bpy.context.view_layer.update()
    
    def rotate_z(self, obj, deg: float):
        """Rotate object around local Z axis."""
        obj.rotation_mode = 'QUATERNION'
        q_rot = Quaternion((0, 0, 1), math.radians(deg))
        obj.rotation_quaternion = q_rot @ obj.rotation_quaternion
        bpy.context.view_layer.update()
    
    def render_frame_v2(self, 
                        cam: bpy.types.Object,
                        model: bpy.types.Object,
                        sun:bpy.types.Object,
                        frame_dict: Dict,
                        frame_id: int,
                        output_dir: Path,
                        exposure_time_s: float,
                        N_digits:int) -> None:
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
        model.rotation_quaternion = q_I_G
        
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
        model_euler_deg = tuple(math.degrees(a) for a in model.rotation_euler)
        # print(f"\n[Model] {model.name} (in inertial frame)")
        # print(f"  Position:    ({p_G_I.x:12.6f}, {p_G_I.y:12.6f}, {p_G_I.z:12.6f})")
        # print(f"  Rotation Q:  ({q_I_G.w:8.6f}, {q_I_G.x:8.6f}, {q_I_G.y:8.6f}, {q_I_G.z:8.6f})")
        # print(f"  Rotation E:  ({model_euler_deg[0]:8.3f}°, {model_euler_deg[1]:8.3f}°, {model_euler_deg[2]:8.3f}°)")
        # print(f"  Distance from origin: {p_G_I.length:.6f} m")
        
        # Camera pose
        cam_euler_deg = tuple(math.degrees(a) for a in cam.rotation_euler)
        cam_to_model = (model.location - cam.location).normalized()
        distance_cam_model = (model.location - cam.location).length
        # print(f"\n[Camera] {cam.name} (in inertial frame)")
        # print(f"  Position:    ({p_C_I.x:12.6f}, {p_C_I.y:12.6f}, {p_C_I.z:12.6f})")
        # print(f"  Rotation Q:  ({q_I_C.w:8.6f}, {q_I_C.x:8.6f}, {q_I_C.y:8.6f}, {q_I_C.z:8.6f})")
        # print(f"  Rotation E:  ({cam_euler_deg[0]:8.3f}°, {cam_euler_deg[1]:8.3f}°, {cam_euler_deg[2]:8.3f}°)")
        # print(f"  Look dir:    ({cam_to_model.x:8.6f}, {cam_to_model.y:8.6f}, {cam_to_model.z:8.6f})")
        # print(f"  Distance to model: {distance_cam_model:.6f} m")
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
        #exp_tag = f"{int(round(exposure_time_s * 1e6)):08d}us"
        #stem = f"{frame_id:04d}_{exp_tag}_{sun_tag}_{mode_suffix}"
        stem = f"{str(frame_id).zfill(N_digits)}"
        
        self.scene.render.filepath = str(output_dir / f"frame_{stem}")
        self.scene.frame_set(frame_id)
        bpy.ops.render.render(write_still=True)
        
        # Restore exposure
        self.scene.view_settings.exposure = base_ev

    def render_frame_motion_blur_traj(self,cam: bpy.types.Object,
                     model: bpy.types.Object,sun:bpy.types.Object,
                     frame_dict1: Dict,  frame_dict2: Dict,
                     frame_id1: int,
                     shutter:float,
                     output_dir: Path,
                     exposure_time_s: float,
                    N_digits:int) -> None:
        clear_anim(cam)
        clear_anim(model)
        self.scene.frame_start = frame_id1
        self.scene.frame_end = frame_id1+1
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
        model.rotation_quaternion = q_I_G
        cam.location = p_C_I
        cam.rotation_quaternion = q_I_C
        bpy.context.view_layer.update()
        keyframe_pose(model, frame_id1)
        keyframe_pose(cam, frame_id1)
        self.scene.frame_set(frame_id1+1)
        p_G_I = frame_dict2["p_G_I"]
        q_I_G = frame_dict2["q_I_G"]
        p_C_I = frame_dict2["p_C_I"]
        q_I_C = frame_dict2["q_I_C"]
        model.location = p_G_I
        model.rotation_quaternion = q_I_G
        cam.location = p_C_I
        cam.rotation_quaternion = q_I_C
        bpy.context.view_layer.update()
        keyframe_pose(model, frame_id1+1)
        keyframe_pose(cam, frame_id1+1)
        self.scene.render.use_motion_blur = True
        self.scene.render.motion_blur_shutter = float(shutter)
        cy = bpy.context.scene.cycles
        if hasattr(cy, "motion_blur_position"):
            cy.motion_blur_position = 'START'  
        self.scene.frame_set(frame_id1)
        bpy.context.view_layer.update()
        #stem = f"{frame_id:04d}_{exp_tag}_{sun_tag}_{mode_suffix}"
        stem = f"{str(frame_id1).zfill(N_digits)}_blurred"
        self.scene.render.filepath = str(output_dir / f"frame_{stem}")
        bpy.ops.render.render(write_still=True)