import bpy
import math
from mathutils import Vector, Matrix, Euler, Quaternion
import numpy as np
import subprocess
import os, sys, importlib.util,glob, shutil
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from datetime import datetime
from math import degrees

# ===========================================================
# GLOBAL SETTINGS
# ===========================================================
SUN_DISTANCE = 5000
VERBOSE = True   # toggle verbosity

# Subdirectory names for CV outputs
CV_SUBFOLDERS = ["Depth", "Normal", "Flow", "Seg"]

# ===========================================================
# UTILS
# ===========================================================

def vprint(*args):
    """Print only if VERBOSE = True."""
    if VERBOSE:
        print(*args)

def images_to_video_ffmpeg(
    input_pattern,
    output_path,
    fps=24,
    crf=18,
    overwrite=True
):
    """
    Convert rendered images to a video using FFmpeg.

    input_pattern: e.g. "//renders/INE_24_11_2025/frame_%04d.png"
    output_path:   e.g. "//renders/INE_24_11_2025/output.mp4"
    fps: frames per second
    crf: quality (lower = better, default 18)
    overwrite: allow overwriting existing file
    """

    # Expand Blender paths ("//")
    abs_input = bpy.path.abspath(input_pattern)
    abs_output = bpy.path.abspath(output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    # -y means overwrite
    overwrite_flag = "-y" if overwrite else "-n"

    cmd = [
        "ffmpeg",
        overwrite_flag,
        "-framerate", str(fps),
        "-i", abs_input,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",   # safest for compatibility
        "-crf", str(crf),
        abs_output
    ]

    print("Running FFmpeg command:\n", " ".join(cmd))

    # Execute FFmpeg
    subprocess.run(cmd, check=True)
    print("Video generated:", abs_output)

def print_orientation(obj, label=None):
    print("\n=== ORIENTATION DEBUG ===")
    if label:
        print(f"Object: {label}")
    else:
        print(f"Object: {obj.name}")

    # --- Euler angles ---
    euler_rad = obj.rotation_euler
    euler_deg = tuple(math.degrees(a) for a in euler_rad)
    print(f"Euler (rad): {tuple(round(a,6) for a in euler_rad)}")
    print(f"Euler (deg): {tuple(round(a,3) for a in euler_deg)}")

    # --- Quaternion ---
    q = obj.rotation_quaternion
    print("Quaternion (w,x,y,z): ", 
          tuple(round(a,6) for a in (q.w, q.x, q.y, q.z)))

    # --- Rotation Matrix ---
    R = obj.matrix_world.to_3x3()
    print("Rotation Matrix:")
    for row in R:
        print(" ", [round(v,6) for v in row])

    # --- Camera forward/up/right ---
    # Blender camera convention: forward = -Z, up = Y
    forward = -(R @ Vector((0,0,1)))
    up      =   R @ Vector((0,1,0))
    right   =   R @ Vector((1,0,0))

    print("Forward vector: ", tuple(round(v,6) for v in forward))
    print("Up vector:      ", tuple(round(v,6) for v in up))
    print("Right vector:   ", tuple(round(v,6) for v in right))

    print("==========================\n")
# ===========================================================
# FILE LOADING
# ===========================================================
def read_gt_values(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    # Find section indices
    idx_nb    = lines.index("nbTruePts =")
    idx_tspan = lines.index("tspan =")
    idx_qGC   = lines.index("q_GC =")
    idx_rCG   = lines.index("r_CG =")
    idx_rOG   = lines.index("r_OG_G =")
    idx_saz   = lines.index("sun_az =")
    idx_sel   = lines.index("sun_el =")
    idx_qIG   = lines.index("q_IG =")

    # Read counts
    nbTruePts = int(lines[idx_nb + 1])

    # Parse arrays
    tspan  = np.array([float(x) for x in lines[idx_tspan+1 : idx_tspan+1+nbTruePts]])
    q_GC   = np.array([list(map(float, x.split())) for x in lines[idx_qGC+1 : idx_qGC+1+nbTruePts]])
    r_CG   = np.array([list(map(float, x.split())) for x in lines[idx_rCG+1 : idx_rCG+1+nbTruePts]])
    r_OG_G = np.array([list(map(float, x.split())) for x in lines[idx_rOG+1 : idx_rOG+1+nbTruePts]])
    sun_az = np.array([float(x) for x in lines[idx_saz+1 : idx_saz+1+nbTruePts]])
    sun_el = np.array([float(x) for x in lines[idx_sel+1 : idx_sel+1+nbTruePts]])
    q_IG   = np.array([list(map(float, x.split())) for x in lines[idx_qIG+1 : idx_qIG+1+nbTruePts]])

    return {
        "N": nbTruePts,
        "t": tspan,
        "q_GC": q_GC,
        "r_CG": r_CG,
        "r_OG": r_OG_G,
        "sun_az": sun_az,
        "sun_el": sun_el,
        "q_IG": q_IG
    }

def apply_camera_pose(cam, target, r_CG, q_GC, debug_frame=None):
    """
    Apply camera pose relative to target object.

    r_CG : np.array of shape (3,)
        camera position in G frame
    q_GC : quaternion [x, y, z, w] from trajectory generator
        rotation from G frame to C frame (camera frame where +Z = forward)

    Note on frame convention:
        - Trajectory generator: R_GC with columns [right, up, forward], +Z = toward target
        - Blender camera: looks along -Z, +Y = up
        - q_GC represents R_G_to_C (active rotation from G to C)
        - Blender's rotation_quaternion represents R_world_to_local
    """

    # Set camera position
    cam.location = Vector(r_CG)

    # Convert quaternion from trajectory format [x,y,z,w] to Blender format
    # Trajectory generator stores as [x, y, z, w]
    qx, qy, qz, qw = q_GC

    # From empirical comparison with to_track_quat('-Z', 'Y'):
    # q_GC raw:     (w=0.9572, x=-0.1981, y=-0.0428, z=0.2067)
    # to_track:     (w=0.2067, x=0.9572,  y=0.1981,  z=0.0428)
    #
    # The mapping is: new_w = old_z, new_x = old_w, new_y = -old_x, new_z = -old_y
    # This corresponds to a specific frame rotation between conventions
    q_blender = Quaternion((qz, qw, -qx, -qy))

    # Debug: print comparison on first few frames
    if debug_frame is not None and debug_frame < 5:
        direction = (target.location - cam.location).normalized()
        q_track = direction.to_track_quat('-Z', 'Y')
        q_raw = Quaternion((qw, qx, qy, qz))
        print(f"\n=== Frame {debug_frame} quaternion comparison ===")
        print(f"  q_GC raw:           {q_raw}")
        print(f"  to_track_quat:      {q_track}")
        print(f"  q_blender (remap):  {q_blender}")

    cam.rotation_quaternion = q_blender

    # (optional) force update
    bpy.context.view_layer.update()

def sun_direction_from_az_el(az, el):
    az = math.radians(az)
    el = math.radians(el)

    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)

    return Vector((x, y, z)).normalized()

def apply_sun_direction(sun_obj, direction):
    quat = direction.to_track_quat('-Z', 'Y')  # Sun shines along -Z
    sun_obj.rotation_euler = quat.to_euler()

def set_scale(obj, scale):
    """
    Scale an object (including EMPTY parents), propagating to all children.
    scale: float or tuple/list (3)
    """
    if isinstance(scale, (float, int)):
        obj.scale = Vector((scale, scale, scale))
    else:
        obj.scale = Vector(scale)

    vprint(f"[Scale] {obj.name} scaled to: {obj.scale}")

    # Optional: force update world matrices
    bpy.context.view_layer.update()


def normalize_bounding_box(obj, smax):
    """
    Normalize the object's world-space bounding box such that
    its largest dimension becomes 'smax'.
    
    Works for:
    - Single meshes
    - Objects with children
    - EMPTY parent with nested meshes
    
    This keeps aspect ratios intact.
    """
    # Compute current bounding box
    min_c, max_c = get_world_bounds(obj)
    dims = max_c - min_c

    max_dim = max(dims.x, dims.y, dims.z)

    if max_dim < 1e-8:
        raise RuntimeError(f"Object '{obj.name}' has zero bounding box dimension.")

    # Compute scale factor
    scale_factor = smax / max_dim

    # Apply uniform scale (correct for empty parents)
    if VERBOSE:
        print(f"[Normalize] {obj.name}:")
        print(f"  Current max dimension: {max_dim:.4f}")
        print(f"  Desired max dimension: {smax:.4f}")
        print(f"  Applying uniform scale: {scale_factor:.4f}")

    set_scale(obj, scale_factor)

    # Update world matrices
    bpy.context.view_layer.update()

    return scale_factor

def load_model(filepath, link=False, append=False):
    """
    Load OBJ/FBX/GLB files or append from a .blend file.
    """
    ext = filepath.lower().split('.')[-1]

    if ext in ["obj"]:
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext in ["fbx"]:
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in ["glb", "gltf"]:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == "blend":
        with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
            if append:
                data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise RuntimeError(f"Unsupported file type: {ext}")


# ===========================================================
# OBJECT INSPECTION
# ===========================================================

def get_world_bounds(obj):
    """Return min and max corners in world coordinates."""
    min_c = Vector((1e10, 1e10, 1e10))
    max_c = Vector((-1e10, -1e10, -1e10))

    objects_to_check = [obj] + list(obj.children_recursive)

    for o in objects_to_check:
        if o.type == 'MESH':
            for v in o.bound_box:
                wv = o.matrix_world @ Vector(v)
                min_c = Vector((min(min_c[i], wv[i]) for i in range(3)))
                max_c = Vector((max(max_c[i], wv[i]) for i in range(3)))

    return min_c, max_c


def compute_center_of_mass(obj):
    """Return center of mass of all mesh vertices."""
    verts = []
    for o in [obj] + list(obj.children_recursive):
        if o.type == "MESH":
            mesh = o.to_mesh()
            for v in mesh.vertices:
                verts.append(o.matrix_world @ v.co)
            o.to_mesh_clear()

    com = sum(verts, Vector()) / len(verts)
    return com


def print_object_info(obj):
    """Verbose printing of all properties."""
    min_c, max_c = get_world_bounds(obj)
    dims = max_c - min_c
    com = compute_center_of_mass(obj)

    vprint("\n=== OBJECT INFO ===")
    vprint("Name:", obj.name)
    vprint("Position:", obj.location)
    vprint("Rotation (Euler):", obj.rotation_euler)
    vprint("Rotation (Quat):", obj.rotation_quaternion)
    vprint("Min corner:", min_c)
    vprint("Max corner:", max_c)
    vprint("Dimensions:", dims)
    vprint("Center of mass:", com)
    vprint("====================\n")


# ===========================================================
# OBJECT TRANSFORMS
# ===========================================================

def set_position(obj, xyz):
    obj.location = Vector(xyz)

def set_rotation_euler(obj, xyz_deg):
    obj.rotation_euler = Euler([math.radians(v) for v in xyz_deg], 'XYZ')

def set_rotation_quaternion(obj, quaternion):
    obj.rotation_quaternion = quaternion

def set_rotation_matrix(obj, R):
    obj.matrix_world = Matrix.Translation(obj.location) @ R.to_4x4()

def apply_target_orientation(target, q_IG, debug_frame=None):
    """
    Apply target orientation from ground truth quaternion.

    q_IG : array-like [x, y, z, w]
        Quaternion from Inertial (I) frame to Body/G frame.
        This represents the target's attitude in inertial space.

    Note on frame convention:
        - q_IG represents R_I_to_G (active rotation from I to G)
        - In Blender, rotation_quaternion rotates the object from its
          local frame to world frame.
        - Since Blender's world frame corresponds to inertial frame (I),
          and the target's local frame is body frame (G), we need q_GI.
        - q_GI = conjugate(q_IG)
    """
    target.rotation_mode = 'QUATERNION'

    # q_IG as [x, y, z, w] from trajectory generator
    qx, qy, qz, qw = q_IG

    # q_IG represents I->G rotation
    # For Blender, we need G->I (local->world), which is the conjugate
    # Conjugate: (w, -x, -y, -z)
    q_blender = Quaternion((qw, -qx, -qy, -qz))

    target.rotation_quaternion = q_blender

    if debug_frame is not None and debug_frame < 5:
        print(f"\n=== Frame {debug_frame} target orientation ===")
        print(f"  q_IG raw (x,y,z,w):  ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")
        print(f"  q_blender (w,x,y,z): {q_blender}")

    bpy.context.view_layer.update()


# ===========================================================
# CAMERA CONTROL
# ===========================================================

def look_at(obj, target_point):
    direction = (target_point - obj.location).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = quat.to_euler()

def print_camera_info(cam):
    vprint("\n=== CAMERA INFO ===")
    vprint("Location:", cam.location)
    vprint("Rotation:", cam.rotation_euler)
    vprint("Lens:", cam.data.lens)
    vprint("Sensor Width:", cam.data.sensor_width)
    vprint("Clip Start/End:", cam.data.clip_start, cam.data.clip_end)
    vprint("Resolution:", bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
    vprint("====================\n")


def set_camera_properties(cam, focal_length=50, clip_start=0.1, clip_end=10000):
    cam.data.lens = focal_length
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end

def rotate_target_about_z(target, angle_deg_per_step):
    """
    Applies a GRADUAL rotation to the target around its LOCAL Z-axis.
    Called once per simulation frame.

    angle_deg_per_step: float (degrees per step)
    """

    # Convert degrees to radians
    angle_rad = math.radians(angle_deg_per_step)

    # Rotation quaternion around LOCAL Z-axis (0,0,1)
    q_rot = Quaternion((0, 0, 1), angle_rad)

    # Apply in QUATERNION mode to avoid gimbal issues
    target.rotation_mode = 'QUATERNION'

    # New rotation = q_rot * old_rot (LOCAL rotation)
    target.rotation_quaternion = q_rot @ target.rotation_quaternion

    bpy.context.view_layer.update()
# ===========================================================
# LIGHT CONTROL
# ===========================================================

def set_sun_light(strength=5.0, direction=(0,0,-1)):
    """
    Create/Modify a sun lamp.
    """
    sun = bpy.data.objects.get("Sun")
    if sun is None:
        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", type="SUN"))
        bpy.context.collection.objects.link(sun)

    # set intensity
    sun.data.energy = strength

    # set direction
    direction = Vector(direction).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    sun.rotation_euler = quat.to_euler()


# ===========================================================
# RENDER SETTINGS
# ===========================================================

def set_render_settings(res=(800,800), bg_color=(0,0,0,1), motion_blur=0.0, noise_strength=0.0):
    scene = bpy.context.scene

    # Resolution
    scene.render.resolution_x, scene.render.resolution_y = res

    # Background color
    scene.world.color = bg_color[:3]

    # Motion blur
    scene.render.use_motion_blur = motion_blur > 0
    scene.render.motion_blur_shutter = motion_blur

    # Sensor noise (compositor)
    bpy.context.scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layer = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")

    if noise_strength > 0:
        noise_node = tree.nodes.new("CompositorNodeDenoise")
        noise_node.prefilter = 'FAST'
        # fake "noise" by disabling denoising (you can add other noise nodes)
        tree.links.new(render_layer.outputs["Image"], noise_node.inputs["Image"])
        tree.links.new(noise_node.outputs["Image"], composite.inputs["Image"])
    else:
        tree.links.new(render_layer.outputs["Image"], composite.inputs["Image"])

# ===========================================================
# SIMPLE DEMO RUN
# ===========================================================

def demo_orbit_render(target_obj, cam, output_dir="//renders/"):
    """
    Circular XY orbit around target.
    z = 0 plane
    10 images
    """

    com = compute_center_of_mass(target_obj)

    radius = 500
    z = 0
    num_frames = 10

    for i in range(num_frames):
        angle = 2*math.pi * i/num_frames
        cam.location = Vector((radius*math.cos(angle), radius*math.sin(angle), z))
        look_at(cam, com)

        render_image(f"{output_dir}orbit_{i:03d}.png")

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
def save_img(png_dir, arr, name, cmap='gray', vmin=None, vmax=None):
    path = os.path.join(png_dir, name)
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)
    print("Saved:", path)
def flow_to_rgb(flow):
    """HSV visualization for optical flow (u,v in pixels)."""
    u, v = flow[..., 0], flow[..., 1]
    ang = np.arctan2(-v, -u)
    mag = np.sqrt(u**2 + v**2)
    hsv = np.zeros((*u.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)          # hue
    hsv[..., 1] = np.clip(mag / np.percentile(mag, 99), 0, 1)  # sat
    hsv[..., 2] = 1.0
    return hsv_to_rgb(hsv)

def id_to_color(ids):
    ids = ids.astype(np.int32)
    r = ((ids * 37) % 255) / 255.0
    g = ((ids * 57) % 255) / 255.0
    b = ((ids * 97) % 255) / 255.0
    return np.stack([r, g, b], axis=-1)

def norm_to_rgb(normals):
    return np.clip(0.5 * (normals[..., :3] + 1.0), 0, 1)

# ===========================================================
# 3D SCENE VISUALIZATION
# ===========================================================

def plot_scene_frame(frame_idx, camera_loc, target_loc, sun_dir_I,
                     camera_trajectory, target_trajectory, output_dir,
                     show_trajectory_window=50, sun_az_deg=None, sun_el_deg=None,
                     sun_cam_angle_G=None):
    """
    TODO this seems like a very useful function
    Generate a 3D plot showing the Blender scene geometry for a single frame.

    Scene setup (INERTIAL FRAME):
    - Earth is at origin (0, 0, 0)
    - Sun direction from sun_az/sun_el (per-frame)
    - Target MOVES around Earth
    - Camera MOVES with target, looking at target

    Parameters:
    -----------
    frame_idx : int
        Current frame number
    camera_loc : array-like (3,)
        Camera position in inertial/world coordinates
    target_loc : array-like (3,)
        Target position in inertial/world coordinates
    sun_dir_I : array-like (3,)
        Sun direction unit vector (from sun_az/sun_el converted to Cartesian)
    camera_trajectory : array-like (N, 3)
        Full camera trajectory in inertial coordinates
    target_trajectory : array-like (N, 3)
        Full target trajectory in inertial coordinates
    output_dir : str
        Directory to save the plot
    show_trajectory_window : int
        Number of frames before/after to show in trajectory
    sun_az_deg : float, optional
        Sun azimuth in degrees (for display)
    sun_el_deg : float, optional
        Sun elevation in degrees (for display)
    sun_cam_angle_G : float, optional
        Pre-computed sun-camera angle in G frame (accurate lighting metric)
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Blender
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 5))

    # Convert to numpy arrays
    camera_loc = np.array(camera_loc)
    target_loc = np.array(target_loc)
    sun_dir = np.array(sun_dir_I)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-9)

    # Camera look-at direction (camera looks toward target)
    cam_to_target = target_loc - camera_loc
    cam_to_target_dist = np.linalg.norm(cam_to_target)
    look_at_dir = cam_to_target / (cam_to_target_dist + 1e-9)

    # Earth direction from target
    earth_dir_from_target = -target_loc / (np.linalg.norm(target_loc) + 1e-9)

    # Earth direction from target (for display)
    earth_lookat_angle = np.degrees(np.arccos(np.clip(np.dot(earth_dir_from_target, look_at_dir), -1, 1)))

    # Use pre-computed G-frame angle if available (accurate), otherwise fall back to inertial approx
    if sun_cam_angle_G is not None:
        sun_camera_angle = sun_cam_angle_G
    else:
        # Fallback: compute in inertial frame (less accurate)
        sun_camera_angle = np.degrees(np.arccos(np.clip(np.dot(sun_dir, -look_at_dir), -1, 1)))

    # Refined illumination classification with transition zone
    # - Full front-lit: sun_camera_angle < 80° (definitely lit)
    # - Transition zone: 80° <= angle < 100° (partial/uncertain lighting)
    # - Full back-lit: angle >= 100° (definitely in shadow)
    if sun_camera_angle < 80:
        target_color = 'gold'
        lit_status = "FRONT-LIT"
    elif sun_camera_angle < 100:
        target_color = 'orange'
        lit_status = f"TRANSITION ({sun_camera_angle:.1f}°)"
    else:
        target_color = 'gray'
        lit_status = "BACK-LIT"

    is_front_lit = sun_camera_angle < 90  # Keep for legacy compatibility

    # Trajectory window
    n_frames = len(camera_trajectory)
    start_idx = max(0, frame_idx - show_trajectory_window)
    end_idx = min(n_frames, frame_idx + show_trajectory_window)

    # ===== Plot 1: World frame view (Earth at origin) =====
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')

    # Earth at origin
    ax1.scatter([0], [0], [0], c='green', s=200, marker='o', label='Earth', zorder=5)

    # Target position
    ax1.scatter([target_loc[0]], [target_loc[1]], [target_loc[2]],
                c=target_color, s=100, marker='*', label='Target', zorder=5)

    # Camera position
    ax1.scatter([camera_loc[0]], [camera_loc[1]], [camera_loc[2]],
                c='blue', s=80, marker='^', label='Camera', zorder=5)

    # Camera look-at arrow (from camera toward target)
    arrow_scale = cam_to_target_dist * 0.8
    look_arrow = look_at_dir * arrow_scale
    ax1.quiver(camera_loc[0], camera_loc[1], camera_loc[2],
               look_arrow[0], look_arrow[1], look_arrow[2],
               color='blue', arrow_length_ratio=0.1, lw=2, label='Look-at')

    # Sun direction arrow (from origin, showing light direction)
    sun_arrow = sun_dir * arrow_scale
    ax1.quiver(0, 0, 0, sun_arrow[0], sun_arrow[1], sun_arrow[2],
               color='orange', arrow_length_ratio=0.1, lw=3, label='Sun dir')

    # Target trajectory
    if end_idx > start_idx:
        traj_t = target_trajectory[start_idx:end_idx]
        ax1.plot(traj_t[:, 0], traj_t[:, 1], traj_t[:, 2], 'r-', alpha=0.4, lw=1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    sun_info = f"az={sun_az_deg:.1f}° el={sun_el_deg:.1f}°" if sun_az_deg is not None else f"∠={sun_camera_angle:.1f}°"
    ax1.set_title(f'World Frame (Earth at origin)\n{lit_status} | Sun {sun_info}')
    ax1.legend(loc='upper left', fontsize=6)

    # ===== Plot 2: Target-centered view =====
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')

    # Positions relative to target
    cam_rel = camera_loc - target_loc
    earth_rel = -target_loc

    # Target at center
    ax2.scatter([0], [0], [0], c=target_color, s=150, marker='*', label='Target', zorder=5)

    # Camera position relative to target
    ax2.scatter([cam_rel[0]], [cam_rel[1]], [cam_rel[2]],
                c='blue', s=100, marker='^', label='Camera', zorder=5)

    # Camera look-at arrow (from camera toward target = toward origin)
    look_arrow_rel = -cam_rel / (np.linalg.norm(cam_rel) + 1e-9) * cam_to_target_dist * 0.5
    ax2.quiver(cam_rel[0], cam_rel[1], cam_rel[2],
               look_arrow_rel[0], look_arrow_rel[1], look_arrow_rel[2],
               color='blue', arrow_length_ratio=0.15, lw=2, label='Look-at')

    # Sun direction arrow
    arrow_scale2 = cam_to_target_dist * 0.6
    sun_arrow2 = sun_dir * arrow_scale2
    ax2.quiver(0, 0, 0, sun_arrow2[0], sun_arrow2[1], sun_arrow2[2],
               color='orange', arrow_length_ratio=0.15, lw=3, label='Sun dir')

    # Earth direction arrow (from target toward Earth)
    earth_arrow = earth_dir_from_target * arrow_scale2
    ax2.quiver(0, 0, 0, earth_arrow[0], earth_arrow[1], earth_arrow[2],
               color='green', arrow_length_ratio=0.15, lw=3, label='Earth dir')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Frame {frame_idx} | Range: {cam_to_target_dist:.1f}m')
    ax2.legend(loc='upper left', fontsize=6)

    max_range = cam_to_target_dist * 1.5
    ax2.set_xlim([-max_range, max_range])
    ax2.set_ylim([-max_range, max_range])
    ax2.set_zlim([-max_range, max_range])

    # ===== Plot 3: Top-down view (X-Y plane) =====
    ax3 = fig.add_subplot(1, 4, 3)

    # Draw lit hemisphere indicator
    r_indicator = max_range * 0.9
    sun_2d = sun_dir[:2]
    sun_2d_norm = np.linalg.norm(sun_2d)
    if sun_2d_norm > 0.1:
        sun_2d_unit = sun_2d / sun_2d_norm
        sun_angle = np.arctan2(sun_2d_unit[1], sun_2d_unit[0])
        theta_lit = np.linspace(sun_angle - np.pi/2, sun_angle + np.pi/2, 50)
        x_lit = r_indicator * np.cos(theta_lit)
        y_lit = r_indicator * np.sin(theta_lit)
        ax3.fill(np.append(x_lit, 0), np.append(y_lit, 0), color='yellow', alpha=0.15, label='Lit side')

    # Target at center
    ax3.scatter([0], [0], c=target_color, s=150, marker='*', label='Target', zorder=5)

    # Camera
    ax3.scatter([cam_rel[0]], [cam_rel[1]], c='blue', s=100, marker='^', label='Camera', zorder=5)

    # Camera look-at arrow
    ax3.arrow(cam_rel[0], cam_rel[1], look_arrow_rel[0], look_arrow_rel[1],
              head_width=2, head_length=1.5, fc='blue', ec='blue', lw=1.5)

    # Sun direction
    ax3.arrow(0, 0, sun_arrow2[0], sun_arrow2[1], head_width=3, head_length=2,
              fc='orange', ec='orange', lw=2, label='Sun')

    # Earth direction
    ax3.arrow(0, 0, earth_arrow[0], earth_arrow[1], head_width=3, head_length=2,
              fc='green', ec='green', lw=2, label='Earth')

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Top-Down (X-Y)\nSun-Cam angle: {sun_camera_angle:.1f}°')
    ax3.set_aspect('equal')
    ax3.legend(loc='upper left', fontsize=6)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-max_range, max_range])
    ax3.set_ylim([-max_range, max_range])

    # ===== Plot 4: Alignment diagram (conceptual) =====
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_aspect('equal')
    ax4.axis('off')

    # Draw alignment: Sun -> Camera -> Target -> Earth
    positions = {'Sun': -1.2, 'Camera': -0.4, 'Target': 0.4, 'Earth': 1.2}
    colors = {'Sun': 'orange', 'Camera': 'blue', 'Target': target_color, 'Earth': 'green'}
    markers = {'Sun': 'o', 'Camera': '^', 'Target': '*', 'Earth': 'o'}
    sizes = {'Sun': 200, 'Camera': 150, 'Target': 200, 'Earth': 200}

    for name, x in positions.items():
        ax4.scatter([x], [0], c=colors[name], s=sizes[name], marker=markers[name], zorder=5)
        ax4.text(x, -0.25, name, ha='center', fontsize=9)

    # Draw arrows showing ideal alignment
    ax4.annotate('', xy=(-0.5, 0), xytext=(-1.0, 0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax4.annotate('', xy=(0.3, 0), xytext=(-0.3, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.annotate('', xy=(1.0, 0), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))

    ax4.set_title(f'{lit_status}\nSun-Cam: {sun_camera_angle:.1f}° | Earth-LookAt: {earth_lookat_angle:.1f}°')

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scene_{frame_idx:04d}.png")
    plt.savefig(output_path, dpi=100)
    plt.close(fig)

    return output_path


def generate_scene_plots(camera_locs, target_locs, output_dir,
                         sun_az_arr, sun_el_arr, r_CG_arr, q_IG_arr,
                         every_n_frames=1, max_frames=None):
    """
    TODO prob want to keep this one as well
    Generate 3D scene plots for multiple frames in the INERTIAL FRAME.

    Scene setup:
    - Earth is at origin (0, 0, 0) in inertial frame
    - Sun is STATIC in inertial frame (set from frame 0 sun_az/sun_el)
    - Target MOVES around Earth AND TUMBLES (via q_IG)
    - Camera MOVES with target, looking at target
    - Lighting is computed by transforming sun direction into body frame

    Parameters:
    -----------
    camera_locs : list of array-like
        Camera positions in inertial coordinates for each frame
    target_locs : list of array-like
        Target positions in inertial coordinates for each frame
    output_dir : str
        Directory to save plots
    sun_az_arr : array-like (N,)
        Sun azimuth angles (degrees) per frame - frame 0 used for static sun
    sun_el_arr : array-like (N,)
        Sun elevation angles (degrees) per frame - frame 0 used for static sun
    r_CG_arr : array-like (N, 3)
        Camera position in G frame (body frame) per frame
    q_IG_arr : array-like (N, 4)
        Quaternion [x, y, z, w] from Inertial to Body (G) frame per frame
    every_n_frames : int
        Generate plot every N frames (1 = every frame)
    max_frames : int or None
        Maximum number of frames to process
    """
    from scipy.spatial.transform import Rotation as R

    camera_trajectory = np.array(camera_locs)
    target_trajectory = np.array(target_locs)
    r_CG_arr = np.array(r_CG_arr)
    q_IG_arr = np.array(q_IG_arr)
    n_frames = len(camera_locs)

    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    # Compute STATIC sun direction in inertial frame from frame 0
    az0_rad = np.radians(sun_az_arr[0])
    el0_rad = np.radians(sun_el_arr[0])
    sun_dir_I = np.array([
        np.cos(el0_rad) * np.cos(az0_rad),
        np.cos(el0_rad) * np.sin(az0_rad),
        np.sin(el0_rad)
    ])

    scene_dir = os.path.join(output_dir, "ScenePlots")
    os.makedirs(scene_dir, exist_ok=True)

    print(f"\n=== Generating scene plots to {scene_dir} ===")
    print(f"    Earth: at origin (0,0,0)")
    print(f"    Sun: STATIC in inertial frame (from frame 0: az={sun_az_arr[0]:.1f}° el={sun_el_arr[0]:.1f}°)")
    print(f"    Target: tumbling via q_IG")
    print(f"    Lighting: sun transformed to body frame, compared with camera direction")

    for i in range(0, n_frames, every_n_frames):
        # Get rotation from Inertial to Body frame at this timestep
        qx, qy, qz, qw = q_IG_arr[i]
        rot_IG = R.from_quat([qx, qy, qz, qw])

        # Transform static sun direction from Inertial to Body frame
        sun_dir_body = rot_IG.apply(sun_dir_I)

        # Camera direction toward target in body frame: -r_CG / |r_CG|
        r_CG_i = r_CG_arr[i]
        cam_dir_body = -r_CG_i / (np.linalg.norm(r_CG_i) + 1e-9)

        # Compute sun-camera angle in body frame (accurate lighting metric for tumbling target)
        sun_cam_angle_body = np.degrees(np.arccos(np.clip(np.dot(sun_dir_body, cam_dir_body), -1, 1)))

        plot_scene_frame(
            frame_idx=i,
            camera_loc=camera_locs[i],
            target_loc=target_locs[i],
            sun_dir_I=sun_dir_I,  # Static sun in inertial frame for visualization
            camera_trajectory=camera_trajectory,
            target_trajectory=target_trajectory,
            output_dir=scene_dir,
            sun_az_deg=sun_az_arr[0],  # Frame 0 values (static)
            sun_el_deg=sun_el_arr[0],
            sun_cam_angle_G=sun_cam_angle_body  # Accurate angle accounting for tumbling
        )

        if i % 50 == 0:
            print(f"  Generated scene plot for frame {i}/{n_frames}")

    print(f"  Scene plots saved to: {scene_dir}")
    return scene_dir


# ===========================================================
# MAIN
# ===========================================================

# Adjust the path as needed

# --------------------------------------------------------
# Helper: append all objects from a .blend file
# --------------------------------------------------------
def get_world_dimensions(obj):
    """Return world-space dimensions using the bounding box."""
    if obj.type != "MESH":
        return None

    # world-space bbox coordinates
    coords = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    min_c = Vector((min(v[i] for v in coords) for i in range(3)))
    max_c = Vector((max(v[i] for v in coords) for i in range(3)))

    dims = max_c - min_c     # XYZ dimensions
    return dims

def create_output_directories(traj_root, create_masked_images=False):
    """
    Create the output directory structure for a single trajectory.

    Parameters:
    -----------
    traj_root : str
        Root directory for the trajectory output
    create_masked_images : bool
        If True, raw images go to images_raw/, masked images go to images/
        If False, all images go to images/

    Returns a dict with all the directory paths.
    """
    # Handle existing directory
    if os.path.exists(traj_root):
        print(f"Removing existing directory: {traj_root}")
        shutil.rmtree(traj_root)

    os.makedirs(traj_root, exist_ok=True)

    # Create subdirectories
    # When masking: raw renders go to images_raw/, masked go to images/
    # When not masking: renders go directly to images/
    if create_masked_images:
        frames_dir = os.path.join(traj_root, "images_raw")  # Raw renders
        masked_dir = os.path.join(traj_root, "images")      # Masked images for SLAM
    else:
        frames_dir = os.path.join(traj_root, "images")      # Direct renders for SLAM
        masked_dir = None

    dirs = {
        'root': traj_root,
        'frames': frames_dir,
        'frames_masked': masked_dir,
        'cv': os.path.join(traj_root, "CV"),
        'scene_plots': os.path.join(traj_root, "ScenePlots"),
    }
    dirs['npz'] = os.path.join(dirs['cv'], "NPZ")
    dirs['cv_imgs'] = os.path.join(dirs['cv'], "imgs")

    # Create all directories
    for d in dirs.values():
        if d is not None:
            os.makedirs(d, exist_ok=True)

    # Create CV subfolders
    for sf in CV_SUBFOLDERS:
        sf_path = os.path.join(dirs['cv_imgs'], sf)
        os.makedirs(sf_path, exist_ok=True)
        dirs[f'cv_{sf.lower()}'] = sf_path

    print(f"\n=== OUTPUT DIRECTORY: {traj_root} ===")

    return dirs


def append_blend_objects(filepath):
    before = set(bpy.data.objects.keys())

    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects[:]  # append all objects

    new_objs = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            new_objs.append(obj)

    return new_objs
def format_obj_info(obj):
    loc = obj.location
    dim = obj.dimensions
    sc  = obj.scale
    rot = obj.rotation_euler  # radians
    
    return (
        f"Object: {obj.name}\n"
        f"  Location (m):      {loc.x:.6f}, {loc.y:.6f}, {loc.z:.6f}\n"
        f"  Rotation (deg):    {degrees(rot.x):.3f}, {degrees(rot.y):.3f}, {degrees(rot.z):.3f}\n"
        f"  Scale:              {sc.x:.6f}, {sc.y:.6f}, {sc.z:.6f}\n"
        f"  Dimensions (m):     {dim.x:.6f}, {dim.y:.6f}, {dim.z:.6f}\n"
        f"------------------------------------------------------------\n"
    )
def position_sun_from_az_el(sun_obj, az_deg, el_deg, distance, target):
    az = math.radians(az_deg)
    el = math.radians(el_deg)

    # Convert spherical -> Cartesian
    x = distance * math.cos(el) * math.cos(az)
    y = distance * math.cos(el) * math.sin(az)
    z = distance * math.sin(el)

    # Move Sun to new position
    sun_obj.location = Vector((x, y, z))

    # Point Sun at the target so -Z axis hits it
    direction = (target.location - sun_obj.location).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    sun_obj.rotation_euler = quat.to_euler()

    bpy.context.view_layer.update()

def scale_object_by_factor(obj, factor):
    """Multiply the current object scale by a factor."""
    old_scale = obj.scale.copy()
    obj.scale = Vector((old_scale.x * factor,
                        old_scale.y * factor,
                        old_scale.z * factor))
    
    print(f"Scaled {obj.name}: {old_scale} -> {obj.scale}")
def get_obj_dims(obj):
    min_corner = Vector((1e10, 1e10, 1e10))
    max_corner = Vector((-1e10, -1e10, -1e10))

    # Collect all objects under this root (including itself)
    to_process = [obj] + list(obj.children_recursive)

    for ob in to_process:
        if ob.type != 'MESH':
            continue
        
        mesh = ob.to_mesh()
        mesh.transform(ob.matrix_world)   # transform vertices to world space

        for v in mesh.vertices:
            min_corner.x = min(min_corner.x, v.co.x)
            min_corner.y = min(min_corner.y, v.co.y)
            min_corner.z = min(min_corner.z, v.co.z)

            max_corner.x = max(max_corner.x, v.co.x)
            max_corner.y = max(max_corner.y, v.co.y)
            max_corner.z = max(max_corner.z, v.co.z)

        ob.to_mesh_clear()

    total_dim = max_corner - min_corner
    return total_dim, min_corner, max_corner
def get_object_total_dimensions(obj):
    """
    Return the world-space bounding dimensions (x,y,z),
    min corner, and max corner of a single object, including
    all its child meshes recursively.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    min_corner = Vector((1e10, 1e10, 1e10))
    max_corner = Vector((-1e10, -1e10, -1e10))

    # Collect object + all children
    all_obs = [obj] + list(obj.children_recursive)

    for ob in all_obs:
        if ob.type != 'MESH':
            continue

        ob_eval = ob.evaluated_get(depsgraph)
        mesh = ob_eval.to_mesh()
        if mesh is None:
            continue

        mesh.transform(ob_eval.matrix_world)

        for v in mesh.vertices:
            min_corner.x = min(min_corner.x, v.co.x)
            min_corner.y = min(min_corner.y, v.co.y)
            min_corner.z = min(min_corner.z, v.co.z)

            max_corner.x = max(max_corner.x, v.co.x)
            max_corner.y = max(max_corner.y, v.co.y)
            max_corner.z = max(max_corner.z, v.co.z)

        ob_eval.to_mesh_clear()

    total_dim = max_corner - min_corner
    return total_dim, min_corner, max_corner
def force_uv_mapping(obj):
    """Ensure all image textures on this object use UV mapping."""
    if obj.type != 'MESH':
        return

    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find all image texture nodes
        img_nodes = [node for node in nodes if node.type == 'TEX_IMAGE']

        for img_node in img_nodes:
            # Create or reuse Texture Coordinate node
            texcoord = None
            for n in nodes:
                if n.type == 'TEX_COORD':
                    texcoord = n
                    break
            if not texcoord:
                texcoord = nodes.new("ShaderNodeTexCoord")
                texcoord.location = (img_node.location.x - 600, img_node.location.y)

            # Create or reuse Mapping node
            mapping = None
            for n in nodes:
                if n.type == 'MAPPING':
                    mapping = n
                    break
            if not mapping:
                mapping = nodes.new("ShaderNodeMapping")
                mapping.location = (img_node.location.x - 300, img_node.location.y)

            # Remove old links to the image texture's vector input
            for link in list(links):
                if link.to_node == img_node and link.to_socket == img_node.inputs["Vector"]:
                    links.remove(link)

            # Now connect: UV → Mapping → Image Texture
            links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
            links.new(mapping.outputs["Vector"], img_node.inputs["Vector"])

            print(f"Updated {mat.name}: now using UV mapping for {img_node.name}")


def prepare_slam_dataset(traj_root, traj_path, timestamps, num_frames):
    """
    Prepare the rendered output for SLAM pipeline consumption.

    This function performs the post-processing that was previously done by run_ue5.py:
    1. Creates imgList.txt with timestamp-image pairs
    2. Copies gtValues.txt and other source files to the render output

    Note: imgList.txt always references images/ folder, which contains:
    - Raw renders (when masking disabled)
    - Masked images with Earth/stars removed (when masking enabled)
    When masking is enabled, raw renders are in images_raw/.

    Parameters:
    -----------
    traj_root : str
        Root directory of the rendered output (e.g., renders/1203_Tumbling_mc0_cro_agent0_2024_...)
    traj_path : str
        Source trajectory folder path (contains gtValues.txt, Config.yaml, etc.)
    timestamps : array-like
        Timestamps for each frame
    num_frames : int
        Number of frames rendered
    """
    print("\n" + "="*60)
    print("PREPARING SLAM DATASET")
    print("="*60)

    images_dir = os.path.join(traj_root, "images")

    # Create imgList.txt
    imglist_path = os.path.join(traj_root, "imgList.txt")
    with open(imglist_path, "w") as f:
        for i in range(num_frames):
            ts = timestamps[i] if i < len(timestamps) else i * 1.0
            f.write(f"{ts:.6f} images/img_{i:05d}.png\n")
    print(f"  Created: {imglist_path}")

    # Copy source files from trajectory folder to render output
    files_to_copy = [
        "gtValues.txt",
        "Config.yaml",
        "sensormeasurements.txt",
        "camera_traj.txt"
    ]

    for filename in files_to_copy:
        src = os.path.join(traj_path, filename)
        dst = os.path.join(traj_root, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")
        else:
            print(f"  [WARN] Not found: {filename}")

    print("\n  SLAM dataset preparation complete!")
    print(f"  Output directory ready for SLAM: {traj_root}")
    print("="*60 + "\n")

    return images_dir, imglist_path


def discover_trajectory_folders(base_dir):
    """
    Scan base_dir for subfolders containing camera_traj.txt.
    Excludes 'json/' and 'trial_plots/' folders.
    Returns list of (folder_name, full_path) tuples.
    """
    excluded = {'json', 'trial_plots', 'renders'}
    trajectories = []

    if not os.path.isdir(base_dir):
        print(f"Warning: {base_dir} does not exist")
        return trajectories

    for name in sorted(os.listdir(base_dir)):
        if name in excluded:
            continue
        folder_path = os.path.join(base_dir, name)
        if not os.path.isdir(folder_path):
            continue
        traj_file = os.path.join(folder_path, "camera_traj.txt")
        if os.path.isfile(traj_file):
            trajectories.append((name, folder_path))

    return trajectories

def select_trajectories_interactive(trajectories):
    """
    TODO useful functionality that we may want
    Prompt user to select which trajectories to render.
    Returns list of selected (name, path) tuples.
    """
    if not trajectories:
        print("No trajectory folders found!")
        return []

    print("\n" + "="*60)
    print("TRAJECTORY FOLDER SELECTION")
    print("="*60)
    print(f"Found {len(trajectories)} trajectory folder(s):\n")

    selected = []
    for i, (name, path) in enumerate(trajectories):
        print(f"[{i+1}/{len(trajectories)}] {name}")
        choice = input("    Render this trajectory? (y/n/q to quit): ").strip().lower()

        if choice == 'q':
            print("\nSelection complete.")
            break
        elif choice in ('y', 'yes', ''):
            selected.append((name, path))
            print(f"    -> SELECTED")
        else:
            print(f"    -> skipped")

    print("\n" + "-"*60)
    print(f"Selected {len(selected)} trajectory folder(s) for rendering:")
    for name, _ in selected:
        print(f"  - {name}")
    print("-"*60 + "\n")

    if selected:
        confirm = input("Proceed with rendering? (y/n): ").strip().lower()
        if confirm not in ('y', 'yes', ''):
            print("Rendering cancelled.")
            return []

    return selected

def render_trajectory(traj_name, traj_path, output_base_dir):
    """
    Render a single trajectory folder.
    traj_name: folder name (e.g., "1203_Tumbling_mc0_cro_agent0")
    traj_path: full path to folder containing camera_traj.txt
    output_base_dir: base directory for renders
    """
    print("\n" + "="*60)
    print(f"RENDERING TRAJECTORY: {traj_name}")
    print("="*60)

    camera_traj_file = os.path.join(traj_path, "camera_traj.txt")
    if not os.path.isfile(camera_traj_file):
        print(f"ERROR: camera_traj.txt not found in {traj_path}")
        return False

    # Create output directory for this trajectory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    render_dir = os.path.join(output_base_dir, f"{traj_name}_{timestamp}")

    return camera_traj_file, render_dir

if __name__ == "__main__":

    # Configuration paths
    OUTPUTFILE_DIR = "/home/jdflo/satslam/SISIFOS/outputfile"
    earth_path = "./Earth.blend"
    hdri_path = "./HDRIs/8k_stars.jpg"
    SCENE_PATH = "/home/jdflo/satslam/SISIFOS/scene.blend"

    # Discover and select trajectories
    trajectories = discover_trajectory_folders(OUTPUTFILE_DIR)
    selected_trajectories = select_trajectories_interactive(trajectories)

    if not selected_trajectories:
        print("No trajectories selected. Exiting.")
        sys.exit(0)

    print(f"\nWill render {len(selected_trajectories)} trajectory folder(s)")

    # Prompt for render quality (Cycles samples)
    print("\n" + "="*60)
    print("RENDER QUALITY SELECTION")
    print("="*60)
    print("Cycles samples (higher = better quality, slower):")
    print("    4-8    = Very fast preview (noisy)")
    print("   16-32   = Fast draft")
    print("   64-128  = Good quality")
    print("  256-512  = High quality (slow)")
    print(" 1024+     = Production quality (very slow)")
    print("-"*60)
    samples_input = input("Enter number of samples [default=32]: ").strip()
    if samples_input == "":
        render_samples = 32
    else:
        try:
            render_samples = int(samples_input)
            if render_samples < 1:
                print("Invalid value, using default 32")
                render_samples = 32
        except ValueError:
            print("Invalid input, using default 32")
            render_samples = 32
    print(f"Using {render_samples} samples per pixel")

    # Prompt for post-processing masking (Earth/stars will be visible in renders for correct lighting)
    print("\n" + "="*60)
    print("POST-PROCESSING OPTIONS")
    print("="*60)
    print("Earth and stars are always rendered for physically accurate lighting (Earthshine).")
    print("You can create additional masked versions with Earth/stars removed using segmentation.")
    mask_input = input("Create masked images (Earth/stars removed)? (y/n) [default=n]: ").strip().lower()
    create_masked_images = mask_input in ('y', 'yes')
    print(f"Masked images: {'ENABLED' if create_masked_images else 'DISABLED'}")

    print("\n=== LOADING MAIN SCENE ===")
    bpy.ops.wm.open_mainfile(filepath=SCENE_PATH)
    print(f"Loaded scene file: {SCENE_PATH}")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # Enable GPU rendering (CUDA for GTX 1080 Ti)
    # TODO we may want to enable this
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = True
        print(f"  Enabled device: {device.name}")
    bpy.context.scene.cycles.device = 'GPU'

    bpy.context.scene.cycles.samples = render_samples
    # Always load Earth.blend (contains Sun light needed for illumination)
    new_objects = append_blend_objects(earth_path)
    RES_X = 480*2
    RES_Y = 480*2
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    print(f"Render resolution set to {RES_X}x{RES_Y}")

    print("=== ALL OBJECTS IN SCENE ===")
    for obj in bpy.data.objects:
        print(f" - {obj.name} ({obj.type})")
    print("=============================\n")
    addon_path = os.path.abspath("/home/jdflo/satslam/SISIFOS/scripts/addon_ground_truth_generation.py")

    spec = importlib.util.spec_from_file_location("vision_blender_addon", addon_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vision_blender_addon"] = mod
    spec.loader.exec_module(mod)
    mod.register()  # installs scene.vision_blender and render handlers

    
    # Earth.blend already loaded above (for Sun); print info about appended objects
    print("\n=== Objects from earth.blend ===")
    for obj in new_objects:
        print(f"\nObject: {obj.name}")
        print("  Scale:", tuple(round(s, 4) for s in obj.scale))
        dims = get_world_dimensions(obj)
        if dims is not None:
            dims_tuple = tuple(round(d, 4) for d in dims)
            print("  World Dimensions (XYZ):", dims_tuple)
        else:
            print("  World Dimensions: (not a mesh)")

    # Set pass_index for segmentation masking (needed for post-processing)
    # pass_index 0 = background, 1 = target (spacecraft), 2+ = Earth objects
    print("\n=== Setting up segmentation pass indices ===")
    target_obj = bpy.data.objects.get("Target")
    if target_obj:
        target_obj.pass_index = 1
        print(f"  Target pass_index = 1")
    for idx, obj_name in enumerate(["Earth", "Clouds", "Atmo"], start=2):
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.pass_index = idx
            print(f"  {obj_name} pass_index = {idx}")

    print("\n===============================================\n")
    # Ensure world uses nodes
    world = bpy.context.scene.world
    world.use_nodes = True

    # Clear existing nodes
    nodes = world.node_tree.nodes
    nodes.clear()

    # Create background node setup
    background = nodes.new(type="ShaderNodeBackground")
    output = nodes.new(type="ShaderNodeOutputWorld")

    # Always load HDRI for stars (needed for physically accurate lighting)
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(hdri_path)
    background.inputs[1].default_value = 1.0  # Strength

    # Link nodes: HDRI -> Background -> Output
    links = world.node_tree.links
    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    print("Loaded HDRI (stars):", hdri_path)
    # 2) Turn on ground-truth toggles (no panel needed)
    vb = scene.vision_blender
    vb.bool_save_gt_data = True
    vb.bool_save_depth = True
    vb.bool_save_normals = True
    vb.bool_save_cam_param = True
    vb.bool_save_opt_flow = True               # needs Cycles' Vector pass
    vb.bool_save_segmentation_masks = True     # needs object pass_index > 0
    vb.bool_save_obj_poses = True

    # Load a spacecraft model
    # -------------------------------------------------------
    # load_model("C:/path/to/target.glb")
    target = bpy.data.objects["Target"]  # assume already in scene
    def fix_texture_coords(obj):
        if obj.type != 'MESH':
            return
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            for img in [n for n in nodes if n.type == 'TEX_IMAGE']:

                # Create Texture Coordinate
                texcoord = nodes.new("ShaderNodeTexCoord")
                texcoord.location = (img.location.x - 600, img.location.y)

                # Create Mapping
                mapping = nodes.new("ShaderNodeMapping")
                mapping.location = (img.location.x - 300, img.location.y)

                # Remove existing vector links
                for l in list(links):
                    if l.to_node == img and l.to_socket == img.inputs["Vector"]:
                        links.remove(l)

                # UV → Mapping → Image
                links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
                links.new(mapping.outputs["Vector"], img.inputs["Vector"])

    # Usage
    fix_texture_coords(target)
    # Inspect
    print_object_info(target)

    # -------------------------------------------------------
    # Create/Set camera
    # -------------------------------------------------------
    cam = bpy.data.objects.get("Camera")
    cam.rotation_mode = 'QUATERNION'
    set_camera_properties(cam, focal_length=200, clip_end=5000000, clip_start=0.00001)
    print_camera_info(cam)

    # -------------------------------------------------------
    # Set lighting
    # -------------------------------------------------------
    #set_sun_light(strength=10.0, direction=(-1,-1,-1))

    # Store original scales for objects that get modified (to reset between trajectories)
    original_scales = {}
    for obj_name in ["Target", "Earth", "Clouds", "Atmo"]:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            original_scales[obj_name] = obj.scale.copy()

    # -------------------------------------------------------
    # MAIN LOOP: Render each selected trajectory
    # -------------------------------------------------------
    for traj_idx, (traj_name, traj_path) in enumerate(selected_trajectories):
        print("\n" + "="*70)
        print(f"RENDERING TRAJECTORY {traj_idx+1}/{len(selected_trajectories)}: {traj_name}")
        print("="*70)

        # Reset object scales to original values (prevents cumulative scaling)
        print("\n=== Resetting object scales ===")
        for obj_name, orig_scale in original_scales.items():
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.scale = orig_scale.copy()
                print(f"  {obj_name}: scale reset to {tuple(orig_scale)}")

        # Load trajectory data
        camera_traj_file = os.path.join(traj_path, "camera_traj.txt")
        if not os.path.isfile(camera_traj_file):
            print(f"ERROR: camera_traj.txt not found in {traj_path}, skipping...")
            continue

        data = read_gt_values(camera_traj_file)
        print(f"Loaded trajectory: {camera_traj_file}")
        print(f"  Number of frames: {data['N']}")

        # Create output directory for this trajectory
        traj_timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        traj_root = bpy.path.abspath(f"//renders/{traj_name}_{traj_timestamp}/")

        # Create all output directories using the helper function
        dirs = create_output_directories(traj_root, create_masked_images)
        traj_frames_dir = dirs['frames']           # Raw renders (images/ or images_raw/)
        traj_frames_masked_dir = dirs['frames_masked']  # Masked images (images/ or None)
        traj_npz_dir = dirs['npz']
        traj_cv_imgs_dir = dirs['cv_imgs']
        traj_cv_imgs_dir_depth = dirs['cv_depth']
        traj_cv_imgs_dir_norm = dirs['cv_normal']
        traj_cv_imgs_dir_flow = dirs['cv_flow']
        traj_cv_imgs_dir_seg = dirs['cv_seg']

        # Debug: print some trajectory info
        print("Norms of r_CG:")
        for i in range(0, min(500, data["N"]), 50):
            v = data["r_CG"][i]
            print(f"{i}: {np.linalg.norm(v)}")

        # Print first 10 norms of r_OG_G
        print("\nNorms of r_OG:")
        for i in range(0, min(500, data["N"]), 50):
            v = data["r_OG"][i]
            print(f"{i}: {np.linalg.norm(v)}")

        print("\n=== OBJECTS IN SCENE ===")
        for obj in bpy.data.objects:
            print(f" - {obj.name} ({obj.type})")
        print("============================================\n")

        # Get Earth objects (always loaded, but may be hidden)
        earth  = bpy.data.objects.get("Earth")
        clouds = bpy.data.objects.get("Clouds")
        atmo   = bpy.data.objects.get("Atmo")
        sun    = bpy.data.objects.get("Sun")

        target = bpy.data.objects["Target"]
        bpy.context.view_layer.update()

        # Configure Sun (always present from Earth.blend)
        if sun:
            sun.data.energy = 10.0
            print(f"Sun energy set to 10.0")

        # Configure Earth shadow casting
        if earth:
            EARTH_CASTS_SHADOW = True
            earth.visible_shadow = EARTH_CASTS_SHADOW
            clouds.visible_shadow = EARTH_CASTS_SHADOW
            atmo.visible_shadow = EARTH_CASTS_SHADOW
            print(f"Earth shadow casting: {'ENABLED' if EARTH_CASTS_SHADOW else 'DISABLED'}")

        # Set target to use quaternion rotation mode for proper tumbling
        target.rotation_mode = 'QUATERNION'
        #normalize_bounding_box(target, 1)
        dim, _, _ = get_object_total_dimensions(target)
        max_dim = max(dim.x, dim.y, dim.z)
        print("Target's dimensions: ", dim)
        print("Max dimension of target is: ", max_dim)
        if   max_dim > 1000:  factor = 1000
        elif max_dim >   50:  factor = 100
        else:                 factor = 1

        scale_object_by_factor(target, 1/100)
        #scale_object_by_factor(target, 1/10)
        dim, _, _ = get_object_total_dimensions(target)
        max_dim = max(dim.x, dim.y, dim.z)
        print("Target's dimensions: ", dim)
        print("Max dimension of target is: ", max_dim)

        # Always scale Earth objects (needed for correct Earthshine even when invisible)
        if earth:
            scale_object_by_factor(earth,  10)
            scale_object_by_factor(clouds, 10)
            scale_object_by_factor(atmo,   10)
            objects = [target, earth, clouds, atmo]
        else:
            objects = [target]

        for obj in objects:
            dim, minc, maxc = get_object_total_dimensions(obj)
            dim, minc, maxc = get_obj_dims(obj)
            print(f"\nObject: {obj.name}")
            print(f"  Min corner: {minc}")
            print(f"  Max corner: {maxc}")
            print(f"  Dimensions: {dim}")

        vs = bpy.context.scene.view_settings
        for attr in dir(vs):
            if not attr.startswith("_"):
                try:
                    value = getattr(vs, attr)
                    print(f"{attr}: {value}")
                except:
                    print(f"{attr}: <unreadable>")

        # Compute poses for all frames
        num_frames = min(500, data["N"])
        target_locs = []
        target_orientations = []
        camera_locs = []
        camera_orientations = []

        for i in range(num_frames):
            v = Vector(data["r_CG"][i])
            e = Vector(data["r_OG"][i])
            # Desired length
            target_norm = np.linalg.norm(e)/1000

            # Compute current length
            current_norm = e.length

            # Avoid divide-by-zero
            if current_norm < 1e-9:
                print("Warning: r_CG[i] has zero length. Using default direction.")
                t_target_to_ECI = Vector((target_norm, 0, 0))
            else:
                scale = target_norm / current_norm
                t_target_to_ECI = e * scale
            t = Vector(t_target_to_ECI)

            # Compute its norm
            norm_t = t.length

            # Desired new norm
            new_norm = np.linalg.norm(v) + norm_t

            # Avoid divide-by-zero (very rare but safe)
            if norm_t < 1e-9:
                print("Warning: t_target_to_ECI has near-zero length. Using fallback.")
                direction = Vector((1, 0, 0))
            else:
                direction = t.normalized()

            # Compute camera location and target location
            r_CG_i = data["r_CG"][i] + t_target_to_ECI
            q_GC_i = data["q_GC"][i]
            target.location = t_target_to_ECI
            camera_loc = r_CG_i

            # Use q_GC directly from trajectory data (already has stable orientation)
            quat = q_GC_i

            # Get target orientation from q_IG
            q_IG_i = data["q_IG"][i]

            # Store poses
            camera_locs.append(camera_loc)
            camera_orientations.append(quat)
            target_locs.append(t_target_to_ECI)
            target_orientations.append(q_IG_i)
            apply_camera_pose(cam, target, camera_loc, quat, debug_frame=i)

            if i % 50 == 0:  # Reduce console spam
                print(f"Frame {i}: camera={cam.location}, target={target.location}")

        output_path = os.path.join(traj_root, "object_info.txt")
        with open(output_path, "w") as f:
            f.write("=== TARGET LOCATIONS ===\n")
            for v in target_locs:
                f.write(f"{v},\t")
                f.write(f"{np.linalg.norm(v)}\n")
            f.write("\n=== TARGET ORIENTATIONS (quaternions) ===\n")
            for q in target_orientations:
                f.write(f"{q}\n")

            f.write("\n=== CAMERA LOCATIONS ===\n")
            for v in camera_locs:
                f.write(f"{v},\t")
                f.write(f"{np.linalg.norm(v)}\n")


            f.write("\n=== CAMERA ORIENTATIONS (quaternions) ===\n")
            for q in camera_orientations:
                f.write(f"{q}\n")

            # Write object info for target and Earth objects
            obj_list = [target]
            if earth:
                obj_list.extend([earth, clouds, atmo])
            for obj in obj_list:
                if obj:
                    f.write(format_obj_info(obj))
                else:
                    f.write("Object missing!\n")

        print("Done writing object info!")

        # Generate 3D scene plots accounting for target tumbling
        # Uses q_IG to transform static sun direction into body frame for accurate lighting
        generate_scene_plots(
            camera_locs=camera_locs,
            target_locs=target_locs,
            output_dir=traj_root,
            sun_az_arr=data["sun_az"],
            sun_el_arr=data["sun_el"],
            r_CG_arr=data["r_CG"],
            q_IG_arr=data["q_IG"],
            every_n_frames=1,  # Generate every nth frame
            max_frames=num_frames
        )

        # Compute sun direction for Blender from frame 0 sun_az/sun_el
        # Note: Blender sun is STATIC, set once at frame 0
        sun_dir_blender = sun_direction_from_az_el(data["sun_az"][0], data["sun_el"][0])
        apply_sun_direction(sun, sun_dir_blender)
        print(f"Sun direction (from frame 0 az={data['sun_az'][0]:.1f}° el={data['sun_el'][0]:.1f}°): {sun_dir_blender}")

        for i in range(num_frames):
            target.location = target_locs[i]
            apply_target_orientation(target, target_orientations[i], debug_frame=i)
            apply_camera_pose(cam, target, camera_locs[i], camera_orientations[i])

            if i % 50 == 0:  # Reduce console spam
                print(f"Rendering frame {i}: camera={cam.location}, target={target.location}")
            scene.render.filepath = f"{traj_frames_dir}/img_{i:05d}"  # SLAM expects img_XXXXX.png

            bpy.context.scene.frame_set(i)
            bpy.ops.render.render(write_still=True)

        # Process NPZ files for CV outputs
        for npz_path in sorted(glob.glob(os.path.join(traj_frames_dir, "*.npz"))):

            # Extract base name (e.g. "0000")
            fname = os.path.basename(npz_path)
            base  = os.path.splitext(fname)[0]

            # Move NPZ to CV/NPZ/
            dst_npz_path = os.path.join(traj_npz_dir, fname)
            os.replace(npz_path, dst_npz_path)
            print("Moved NPZ:", fname)

            # Load the data
            npz_data = np.load(dst_npz_path, allow_pickle=True)
            print("Processing:", base)

            # --------- DEPTH VISUALIZATION ---------
            if "depth_map" in npz_data:
                d = npz_data["depth_map"]
                valid = np.isfinite(d) & (d > 0)
                if np.any(valid):
                    p1, p99 = np.percentile(d[valid], [1, 99])
                    depth_vis = np.clip((d - p1) / (p99 - p1 + 1e-6), 0, 1)
                    depth_vis = 1 - depth_vis
                    save_img(traj_cv_imgs_dir_depth, depth_vis, f"{base}_Depth.png")

            # --------- NORMALS VISUALIZATION ---------
            if "normal_map" in npz_data:
                save_img(traj_cv_imgs_dir_norm, norm_to_rgb(npz_data["normal_map"]), f"{base}_Normal.png")

            # --------- SEGMENTATION VISUALIZATION ---------
            if "segmentation_masks" in npz_data:
                seg = npz_data["segmentation_masks"]
                save_img(traj_cv_imgs_dir_seg, id_to_color(seg), f"{base}_Seg.png")

            # --------- MASKED IMAGE (Earth/stars removed) ---------
            # Use depth map to mask - finite depth = spacecraft, infinite = background/Earth/stars
            # Masked images go to images/ folder (for SLAM), raw images are in images_raw/
            if create_masked_images and "depth_map" in npz_data:
                depth = npz_data["depth_map"]
                # Find corresponding rendered image (in images_raw/ folder)
                # NPZ base is "0000", rendered image is "img_00000.png"
                img_idx = int(base)
                rendered_img_path = os.path.join(traj_frames_dir, f"img_{img_idx:05d}.png")
                if os.path.exists(rendered_img_path):
                    # Load rendered image
                    rendered_img = plt.imread(rendered_img_path)
                    # Create mask: True where depth is finite and positive (spacecraft pixels)
                    # Background, Earth, and stars have infinite depth
                    spacecraft_mask = np.isfinite(depth) & (depth > 0)
                    # Create masked image (black where not spacecraft)
                    masked_img = np.zeros_like(rendered_img)
                    masked_img[spacecraft_mask] = rendered_img[spacecraft_mask]
                    # Save masked image to images/ folder (same naming as raw for SLAM compatibility)
                    masked_img_path = os.path.join(traj_frames_masked_dir, f"img_{img_idx:05d}.png")
                    plt.imsave(masked_img_path, masked_img)
                    print(f"  Saved masked image: img_{img_idx:05d}.png")
                else:
                    print(f"  Warning: Could not find rendered image {rendered_img_path}")

            # --------- OPTICAL FLOW VISUALIZATION ---------
            if "optical_flow" in npz_data:
                flow_rgb = flow_to_rgb(npz_data["optical_flow"])
                save_img(traj_cv_imgs_dir_flow, flow_rgb, f"{base}_Flow.png")

        # Generate video from rendered frames
        images_to_video_ffmpeg(
            input_pattern = os.path.join(traj_frames_dir, "img_%05d.png"),
            output_path   = os.path.join(traj_frames_dir, "output_video.mp4"),
            fps = 5,
            crf = 20
        )

        # Generate videos for each CV output type
        for sf in CV_SUBFOLDERS:
            folder = os.path.join(traj_cv_imgs_dir, sf)

            # Get all PNGs
            png_list = sorted(glob.glob(os.path.join(folder, "*.png")))
            if len(png_list) == 0:
                print(f"Skipping {sf}: folder is empty or missing.")
                continue

            # Detect correct suffix from the first file
            # Example: "0003_Depth.png" -> "Depth"
            first_file = os.path.basename(png_list[0])
            try:
                suffix = first_file.split("_", 1)[1].rsplit(".", 1)[0]  # "Depth"
            except:
                print(f"Skipping {sf}: could not determine suffix for {first_file}")
                continue

            # FFmpeg expects "%04d_Suffix.png"
            input_pattern = os.path.join(folder, f"%04d_{suffix}.png")
            output_path   = os.path.join(folder, f"{suffix}_video.mp4")

            print(f"Generating video for: {sf} ({suffix})")
            print("Pattern:", input_pattern)

            images_to_video_ffmpeg(
                input_pattern=input_pattern,
                output_path=output_path,
                fps=5,
                crf=20
            )

        # Generate video for scene plots
        scene_plots_dir = os.path.join(traj_root, "ScenePlots")
        if os.path.isdir(scene_plots_dir):
            scene_pngs = sorted(glob.glob(os.path.join(scene_plots_dir, "scene_*.png")))
            if len(scene_pngs) > 0:
                print(f"\nGenerating scene plots video ({len(scene_pngs)} frames)...")
                images_to_video_ffmpeg(
                    input_pattern=os.path.join(scene_plots_dir, "scene_%04d.png"),
                    output_path=os.path.join(scene_plots_dir, "scene_video.mp4"),
                    fps=10,
                    crf=20
                )

        # Prepare SLAM dataset (creates imgList.txt, copies gtValues.txt, etc.)
        prepare_slam_dataset(
            traj_root=traj_root,
            traj_path=traj_path,
            timestamps=data["t"],
            num_frames=num_frames
        )

        print(f"\nCompleted trajectory: {traj_name}")
        print(f"Output directory: {traj_root}")

    print("\n" + "="*70)
    print("ALL TRAJECTORIES COMPLETED")
    print("="*70)