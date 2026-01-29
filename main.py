"""
© Dynamics and Control Systems Laboratory, Georgia Institute of Technology
SISIFOS: Specialized Illumination SImulator For Orbiting Spacecraft

This is the main file, setting up the simulation, loading the configuration
and actually rendering the data (images and annotations).

Iason Georgios Velentzas (ivelentzas3@gatech.edu)

"""



import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import bpy
import math
from mathutils import Vector, Matrix, Euler, Quaternion
from typing import Tuple
import sys
import importlib.util
from datetime import datetime
import os
import glob
from tqdm import tqdm
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb
SUN_AZ_MAN = [0]
SUN_EL_MAN = [-90, -45, 0, 45, 90]


@dataclass
class ObjectConfig:
    name: str
    blend_path: Optional[str] = None      
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    hide_render: bool = False
    extra_scale: float = 1.0              # Potential dditional scaling after loading

@dataclass
class CameraConfig:
    """Camera properties"""
    focal_length: float = 400.0
    clip_start: float = 0.00001
    clip_end: float = 5000000.0
    resolution: tuple = (480, 480)

@dataclass
class RenderConfig:
    """Render settings"""
    engine: str = "CYCLES"
    samples: int = 32
    bg_color: tuple = (0.0, 0.0, 0.0, 1.0)
    motion_blur: float = 0.0
    noise_strength: float = 0.0

@dataclass
class SetupConfig:
    """Trajectory and Environment Setup"""
    num_frames: int = 200
    R_RPO: float = 70.0
    R_LEO: float = 10000.0
    # Sweep definitions (optional)
    sweep_exposure: Optional[Dict[str, float]] = None 
    sweep_sun_az_el: Optional[Dict[str, Dict[str, float]]] = None
     # Reference settings
    t_ref_s: float = 0.01666667
    sun_az_ref: float = 0.0
    sun_el_ref: float = 0.0
    earth_mode: str = "on"
    stars_mode: str = "on"
    enable_blur: str = "off"
    blur_shutter_factor: float =  0.8
    blur_motion_factor: float =  0.8
    enable_glare: str = "OFF"
    glare_threshold: float = 0.95
    glare_size: int = 6

@dataclass
class SceneConfig:
    """Total Configuration, model and output"""
    scene_blend_path: str
    hdri_path: str
    objects: Dict[str, ObjectConfig] = field(default_factory=dict)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
    exposure_times_s: List[float] = field(default_factory=lambda: [1/60])
    # Vision Blender addon settings
    save_depth: bool = True
    save_normals: bool = True
    save_optical_flow: bool = True
    save_segmentation: bool = True
    save_obj_poses: bool = True
    
    # Rendering control
    frame_ids: Optional[List[int]] = None  # If None, use all frames
    selected_models: List[str] = field(default_factory=list)  # Empty = render all RF_* models
    model_rotation_z_deg: float = 45.0  # Apply initial Z rotation, will be extended to X,Y
    
    @classmethod
    def from_json(cls, json_path: str) -> "SceneConfig":
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'camera' in data:
            data['camera'] = CameraConfig(**data['camera'])
        if 'render' in data:
            data['render'] = RenderConfig(**data['render'])
        if 'setup' in data:
            data['setup'] = SetupConfig(**data['setup'])
        if 'objects' in data:
            data['objects'] = {k: ObjectConfig(**v) for k, v in data['objects'].items()}
        
        return cls(**data)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file"""
        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            return str(obj)
        
        with open(json_path, 'w') as f:
            json.dump(asdict(self, dict_factory=lambda x: {k: serialize(v) for k, v in x}), 
                     f, indent=2)


def fibonacci_sphere(n: int, radius: float = 1.0) -> List[Vector]:
    """Fibonacci sampled points along a sphere of specific radius"""
    if n <= 0:
        return []
    pts = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i) / (n - 1) if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts.append(Vector((x, y, z)) * radius)
    return pts

def _rand_quat_uniform(rng: random.Random) -> Quaternion:
    """Generate uniformly random rotation in quaternion."""
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    
    q = Quaternion((
        math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
        math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
        math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
        math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
    ))
    # Convert from (x,y,z,w) to (w,x,y,z)
    q = Quaternion((q[3], q[0], q[1], q[2])).normalized()
    return q

"""
Generation and loading of ground truth file. This will be replaced with various options,
namely, the internal trajectory generation module, as well as options for easily
configurable random pose set.
"""
def write_camera_approach(
    out_path: str,
    N: int,
    R_LEO: float,
    R_RPO_start: float = 36.0,
    R_RPO_end: float = 5.0,
    n_revs: float = 1.0,              # number of orbital revolutions over N frames
    orbit_plane: str = "xy",          # "xy", "xz", or "yz"
    shuffle_points: bool = False,     
    seed: int = 0,
    sun_az: float = 0.0,
    sun_el: float = 0.0,
    verbose: bool = True,
    include_rpo_column: bool = True
) -> str:
    """
    This is a demo approach phase. Not actually implementing the trajectory module
    """

    out_path = str(Path(out_path))

    
    if N < 2:
        rpo_list = [float(R_RPO_start)]
    else:
        rpo_list = [
            float(R_RPO_start + (R_RPO_end - R_RPO_start) * (i / (N - 1)))
            for i in range(N)
        ]

    if verbose:
        print(f"R_RPO evolution: start={rpo_list[0]:.3f}, end={rpo_list[-1]:.3f}")
        if N >= 6:
            print("  first 3:", [round(x, 3) for x in rpo_list[:3]])
            print("  last  3:", [round(x, 3) for x in rpo_list[-3:]])

    lines = [
        "# camera_traj_orbit_approach.txt (inertial frame reference)",
        "# Target: circular orbit around origin at radius R_LEO",
        "# Camera: co-orbits with target and approaches with R_RPO(i) decreasing",
        "# Camera is center-pointing at target (-Z axis looks at target)",
        f"# N={N}, R_LEO={R_LEO:.6f}, R_RPO_start={R_RPO_start:.6f}, R_RPO_end={R_RPO_end:.6f}, n_revs={n_revs}",
        "# Columns:",
        "# p_G_I(xyz)  q_I_G(wxyz)  p_C_I(xyz)  q_I_C(wxyz)  sun_az  sun_el" + ("  R_RPO" if include_rpo_column else ""),
        "#",
    ]

    # Orbit angle over time
    # theta spans n_revs revolutions
    lag_ratio=-0.5
    for i in range(N):
        t = 0.0 if N < 2 else (i / (N - 1))
        theta = 2.0 * math.pi * n_revs * t
        if i <N//2:
            lag_ratio += 0.75/N
        else:
            lag_ratio -= 0.5/N
        

        # Unit radial direction (depends on chosen plane)
        if orbit_plane == "xy":
            u = Vector((math.cos(theta), math.sin(theta), 0.0))
        elif orbit_plane == "xz":
            u = Vector((math.cos(theta), 0.0, math.sin(theta)))
        elif orbit_plane == "yz":
            u = Vector((0.0, math.cos(theta), math.sin(theta)))
        else:
            raise ValueError("orbit_plane must be one of: 'xy', 'xz', 'yz'")
        if orbit_plane == "xy":
            t_hat = Vector((-math.sin(theta),  math.cos(theta), 0.0))
        elif orbit_plane == "xz":
            t_hat = Vector((-math.sin(theta), 0.0,  math.cos(theta)))
        elif orbit_plane == "yz":
            t_hat = Vector((0.0, -math.sin(theta),  math.cos(theta)))
        else:
            raise ValueError("orbit_plane must be one of: 'xy', 'xz', 'yz'")

        t_hat.normalize()
        u.normalize()

        # Target position on orbit
        p_G_I = u * R_LEO

        # Camera approaches target along the same radial line, staying OUTSIDE (Earth behind target)
        R_RPO_i = rpo_list[i]
        #p_C_I = u * (R_LEO + R_RPO_i)
        r_behind = lag_ratio *  rpo_list[0]
        r_rad = math.sqrt(max(0.0, R_RPO_i * R_RPO_i - r_behind * r_behind))
        p_C_I = p_G_I + (u * r_rad) + (t_hat * r_behind)

        # Random target orientation per frame (same as your v2 idea)
        #rng = random.Random(seed + i)
        q_IG = Quaternion((1.0, 0.0, 0.0, 0.0)).normalized()          # should return a mathutils.Quaternion (wxyz or xyzw depending on your helper)
        q_IG_wxyz = quat_to_wxyz(q_IG)

        # Camera orientation: center-point at target
        look_dir = (p_G_I - p_C_I)
        if look_dir.length < 1e-12:
            # Degenerate (shouldn't happen unless R_RPO_i == 0)
            look_dir = -u
        else:
            look_dir.normalize()

        q_IC = look_dir.to_track_quat('-Z', 'Y').normalized()
        q_IC_wxyz = quat_to_wxyz(q_IC)

        # Write row
        row = (
            f"{p_G_I.x:12.6f} {p_G_I.y:12.6f} {p_G_I.z:12.6f}  "
            f"{q_IG_wxyz[0]:.9f} {q_IG_wxyz[1]:.9f} {q_IG_wxyz[2]:.9f} {q_IG_wxyz[3]:.9f}  "
            f"{p_C_I.x:12.6f} {p_C_I.y:12.6f} {p_C_I.z:12.6f}  "
            f"{q_IC_wxyz[0]:.9f} {q_IC_wxyz[1]:.9f} {q_IC_wxyz[2]:.9f} {q_IC_wxyz[3]:.9f}  "
            f"{sun_az:.6f} {sun_el:.6f}"
        )
        if include_rpo_column:
            row += f"  {R_RPO_i:.6f}"

        lines.append(row)
    ensure_dir(Path(out_path).parent)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    if verbose:
        print(f"Trajectory (orbit + approach) written to: {out_path}")
    return out_path

def write_camera_trajectory_v2(
    out_path: str,
    N: int,
    R_LEO: float,
    R_RPO: float,
    shuffle_points: bool = False,
    seed: int = 0,
    verbose: bool = True,
) -> str:
    """
    Generate camera Fibonacci-style file with INERTIAL FRAME reference.
    
    Target (G) orbits on sphere at radius R_LEO.
    Camera (C) is positioned radially outward from target at distance R_RPO,
    looking back toward origin.
    Both use same Fibonacci sphere angles.
    
    File format (one row per frame):
      p_G_I(xyz)  q_I_G(wxyz)  p_C_I(xyz)  q_I_C(wxyz)  sun_az  sun_el
    
    Where:
      p_G_I = position of target (G) in inertial frame at radius R_LEO
      q_I_G = orientation of target frame relative to inertial
      p_C_I = position of camera (C) in inertial frame at radius (R_LEO + R_RPO)
      q_I_C = orientation of camera frame relative to inertial
      
    Earth/clouds/atmosphere stay at inertial origin (0,0,0) with fixed orientation.
    """
    out_path = str(Path(out_path))
    
    sphere_pts = fibonacci_sphere(N, radius=1.0)
    
    if shuffle_points:
        random.seed(seed)
        random.shuffle(sphere_pts)
    
    lines = [
        "# camera_traj_v2.txt (inertial frame reference - orbital configuration)",
        "# p_G_I(xyz)  q_I_G(wxyz)  p_C_I(xyz)  q_I_C(wxyz)  sun_az  sun_el",
        f"# Target orbits at radius R_LEO = {R_LEO:.2f} m",
        f"# Camera at radius R_LEO + R_RPO = {R_LEO + R_RPO:.2f} m, looking toward origin",
        "# Earth/atmosphere at origin with fixed orientation",
        "#"
    ]
    
    for i in range(N):
        # Fibonacci sphere point (unit direction)
        direction_radial = sphere_pts[i].normalized()
        
        # Target position at radius R_LEO
        p_G_I = direction_radial * R_LEO
        
        # Camera position at radius R_LEO + R_RPO (same direction)
        p_C_I = direction_radial * (R_LEO + R_RPO)
        
        # Random target orientation in inertial frame
        rng = random.Random(seed + i)
        q_IG = _rand_quat_uniform(rng)
        
        # Camera orientation: looking toward origin (nadir/Earth-pointed)
        # Direction from camera toward sc/origin (negative radial direction)
        look_direction = -direction_radial  # Points toward Earth at origin
        
        # Create quaternion that aligns -Z axis (camera forward) with look direction
        q_IC = look_direction.to_track_quat('-Z', 'Y').normalized()
        
        # Convert quaternions to wxyz format
        q_IG_wxyz = quat_to_wxyz(q_IG)
        q_IC_wxyz = quat_to_wxyz(q_IC)
        lines.append(
                    f"{p_G_I.x:12.6f} {p_G_I.y:12.6f} {p_G_I.z:12.6f}  "
                    f"{q_IG_wxyz[0]:.9f} {q_IG_wxyz[1]:.9f} {q_IG_wxyz[2]:.9f} {q_IG_wxyz[3]:.9f}  "
                    f"{p_C_I.x:12.6f} {p_C_I.y:12.6f} {p_C_I.z:12.6f}  "
                    f"{q_IC_wxyz[0]:.9f} {q_IC_wxyz[1]:.9f} {q_IC_wxyz[2]:.9f} {q_IC_wxyz[3]:.9f}  "
                )

        # Fibonacci sphere point (unit direction)
        direction_radial = -sphere_pts[i].normalized()
        
        # Target position at radius R_LEO
        p_G_I = direction_radial * R_LEO
        
        # Camera position at radius R_LEO - R_RPO (no Earth)
        p_C_I = direction_radial * (R_LEO - R_RPO)
        
        # Random target orientation in inertial frame
        rng = random.Random(seed + i)
        q_IG = _rand_quat_uniform(rng)
        
        # Camera orientation: looking toward OUT (-nadir/Earth-pointed from the other side)
        # Direction from camera toward sc (negative of negative already radial direction)
        look_direction = direction_radial  # Points toward Earth at origin
        
        # Create quaternion that aligns -Z axis (camera forward) with look direction
        q_IC = look_direction.to_track_quat('-Z', 'Y').normalized()
        
        # Convert quaternions to wxyz format
        q_IG_wxyz = quat_to_wxyz(q_IG)
        q_IC_wxyz = quat_to_wxyz(q_IC)
        
        lines.append(
            f"{p_G_I.x:12.6f} {p_G_I.y:12.6f} {p_G_I.z:12.6f}  "
            f"{q_IG_wxyz[0]:.9f} {q_IG_wxyz[1]:.9f} {q_IG_wxyz[2]:.9f} {q_IG_wxyz[3]:.9f}  "
            f"{p_C_I.x:12.6f} {p_C_I.y:12.6f} {p_C_I.z:12.6f}  "
            f"{q_IC_wxyz[0]:.9f} {q_IC_wxyz[1]:.9f} {q_IC_wxyz[2]:.9f} {q_IC_wxyz[3]:.9f}  "
        )
    ensure_dir(Path(out_path).parent)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    if verbose:
        print(f"Trajectory (v2 - inertial orbital) written to: {out_path}")
    
    return out_path

def load_camera_trajectory_v2(path: str) -> List[Dict]:
    """
    Load trajectory from v2 file (inertial frame reference).
    
    File format (per line):
      p_G_I(3)  q_I_G(4)  p_C_I(3)  q_I_C(4)  sun_az(1)  sun_el(1)
      = 16 floats total per line
    
    Returns list of dicts with keys:
      p_G_I: position of target in inertial frame
      q_I_G: orientation of target relative to inertial
      p_C_I: position of camera in inertial frame
      q_I_C: orientation of camera relative to inertial
      sun_az, sun_el: sun angles
    """
    traj = []
    with open(path, "r") as f:
        for line_num, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            
            parts = [float(x) for x in ln.split()]
            if len(parts) < 14:
                raise ValueError(f"Line {line_num}: Expected 14 floats, got {len(parts)}")
            
            # Parse: p_G_I(3) q_I_G(4) p_C_I(3) q_I_C(4) sun_az sun_el
            p_G_I = Vector((parts[0], parts[1], parts[2]))
            q_I_G = Quaternion((parts[3], parts[4], parts[5], parts[6])).normalized()
            p_C_I = Vector((parts[7], parts[8], parts[9]))
            q_I_C = Quaternion((parts[10], parts[11], parts[12], parts[13])).normalized()
            
            traj.append({
                "p_G_I": p_G_I,
                "q_I_G": q_I_G,
                "p_C_I": p_C_I,
                "q_I_C": q_I_C,
            })
    
    return traj

# ============================================================================
# Core utility functions of the simulator. To be transfered to utils.py
# ============================================================================

def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """HSV visualization for optical flow (u,v in pixels)."""
    u, v = flow[..., 0], flow[..., 1]
    ang = np.arctan2(-v, -u)
    mag = np.sqrt(u * u + v * v)
    hsv = np.zeros((*u.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2.0 * np.pi)  # hue
    p99 = np.percentile(mag, 99) if np.any(np.isfinite(mag)) else 1.0
    hsv[..., 1] = np.clip(mag / (p99 + 1e-9), 0, 1)  # sat
    hsv[..., 2] = 1.0  # val
    return hsv_to_rgb(hsv).astype(np.float32)

def _norm_to_rgb(normals: np.ndarray) -> np.ndarray:
    """Normals assumed in [-1,1]; map to [0,1]."""
    return np.clip(0.5 * (normals[..., :3] + 1.0), 0, 1).astype(np.float32)


def _id_to_color(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.int32)
    r = ((ids * 37) % 255) / 255.0
    g = ((ids * 57) % 255) / 255.0
    b = ((ids * 97) % 255) / 255.0
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _depth_vis_and_mask_from_rrpo(
    depth: np.ndarray,
    R_RPO: float,  
    cmap_name: str = "magma",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rgb: HxWx3 float32 in [0,1]
      mask: HxW bool (True = near object)
    """
    valid = np.isfinite(depth) & (depth > 0)
    mask = valid & (depth <  float(R_RPO)+5)

    dmin = 0.1
    dmax = float(R_RPO)+5
    denom = (dmax - dmin) if (dmax > dmin) else 1.0

    x = (depth - dmin) / denom
    x = np.clip(x, 0.0, 1.0)

    cmap = mpl.colormaps.get_cmap(cmap_name)   # or mpl.colormaps[cmap_name]
    rgb = cmap(x)[..., :3].astype(np.float32)    
    rgb[~mask] = 0.0  # background black
    return rgb, mask


def handle_gt_from_npz(
    npz_src: Path,
    gt_npz_dir: Path,
    gt_depth_dir: Path,
    gt_norm_dir: Path,
    gt_flow_dir: Path,
    gt_seg_dir: Path,
    R_RPO: float
):
    
    npz_src = Path(npz_src)
    gt_npz_dir = Path(gt_npz_dir)
    gt_depth_dir = Path(gt_depth_dir)
    gt_norm_dir = Path(gt_norm_dir)
    gt_flow_dir = Path(gt_flow_dir)
    gt_seg_dir = Path(gt_seg_dir)

    gt_npz_dir.mkdir(parents=True, exist_ok=True)
    gt_depth_dir.mkdir(parents=True, exist_ok=True)
    gt_norm_dir.mkdir(parents=True, exist_ok=True)
    gt_flow_dir.mkdir(parents=True, exist_ok=True)
    gt_seg_dir.mkdir(parents=True, exist_ok=True)

    # --- move npz into GT NPZ folder ---
    npz_dst = gt_npz_dir / npz_src.name
    if npz_dst.resolve() != npz_src.resolve():
        try:
            npz_src.replace(npz_dst)   # atomic move if possible
        except Exception:
            # fallback: copy then remove
            import shutil
            shutil.copy2(npz_src, npz_dst)
            npz_src.unlink(missing_ok=True)

    base = npz_dst.stem  # e.g. "frame_0001" or "frame_0001_sun_00"

    data = np.load(npz_dst, allow_pickle=True)

    
   
    # --------- DEPTH (masked + colormap) ---------
    if "depth_map" in data:
        d = data["depth_map"].astype(np.float32)
        depth_rgb, near_mask = _depth_vis_and_mask_from_rrpo(
            d, R_RPO,
            cmap_name="viridis",
        )
        plt.imsave(str(gt_depth_dir / f"{base}_Depth.png"), depth_rgb)

        # Save the near-mask too (handy for debugging / training)
        plt.imsave(str(gt_seg_dir / f"{base}_SegDepthGate.png"), near_mask.astype(np.float32), cmap="gray")

    # --------- NORMALS ---------
    if "normal_map" in data:
        n = data["normal_map"].astype(np.float32)
        plt.imsave(str(gt_norm_dir / f"{base}_Normal.png"), _norm_to_rgb(n))

    # --------- OPTICAL FLOW ---------
    if "optical_flow" in data:
        flow = data["optical_flow"].astype(np.float32)
        plt.imsave(str(gt_flow_dir / f"{base}_Flow.png"), _flow_to_rgb(flow))

    # --------- SEGMENTATION (addon-provided) ---------
    if "segmentation_masks" in data:
        seg = data["segmentation_masks"]
        plt.imsave(str(gt_seg_dir / f"{base}_Seg.png"), _id_to_color(seg))

def sun_sweep_90(az_step: int = 90,
    el_step: int = 90,
    az_range=(45, 360),     # [start, end) in degrees
    el_range=(-45, 90),    # inclusive
    include_end_az: bool = False,  # keep 360? usually False because 0 == 360
) -> dict:
    """ It does a sweep of sun directions every 90 degrees per angle.
        It will be implemented in a more generalized way.
    """
    az0, az1 = az_range
    el0, el1 = el_range

    az_vals = list(range(int(az0), int(az1), int(az_step)))
    if include_end_az and az1 not in az_vals:
        az_vals.append(int(az1))

    el_vals = list(range(int(el0), int(el1) + 1, int(el_step)))

    sweep = {}
    idx = 0
    for el in el_vals:
        for az in az_vals:
            sweep[f"{idx:02d}"] = {"sun_az_deg": float(az), "sun_el_deg": float(el)}
            idx += 1
    return sweep
def _rand_unit_vec(rng: random.Random) -> Vector:
    # Random point on sphere
    z = rng.uniform(-1.0, 1.0)
    t = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z*z))
    return Vector((r * math.cos(t), r * math.sin(t), z))

def _small_random_rotation(rng: random.Random, max_deg: float) -> Quaternion:
    """Return a small random axis-angle rotation quaternion."""
    if max_deg <= 0.0:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis = _rand_unit_vec(rng).normalized()
    ang = math.radians(rng.uniform(-max_deg, max_deg))
    return Quaternion(axis, ang).normalized()

def make_fake_frame_from_frame0(
    frame0: dict,
    seed: int,
    cam_dir_max_deg: float = 0.5,        # how much to "swing" camera direction around target
    cam_radius_scale_sigma: float = 0.0, # optional: random radial scaling (e.g. 0.01 = 1%)
    target_rot_max_deg: float = 0.0,     # optional: random target rotation
    force_camera_lookat: bool = True,    # recommended
) -> dict:
    """
    Create an artificial 'next frame' using only frame0, by perturbing camera pose
    (and optionally target orientation). Keeps target position fixed.

    frame0 keys: p_G_I, q_I_G, p_C_I, q_I_C
    returns dict with same keys.
    """
    rng = random.Random(seed)

    p_G_I0: Vector = frame0["p_G_I"]
    q_I_G0: Quaternion = frame0["q_I_G"]
    p_C_I0: Vector = frame0["p_C_I"]
    q_I_C0: Quaternion = frame0["q_I_C"]

    # --- target: keep position, optionally perturb orientation ---
    dq_t = _small_random_rotation(rng, target_rot_max_deg)
    q_I_G1 = (dq_t @ q_I_G0).normalized()
    p_G_I1 = p_G_I0.copy()

    # --- camera: perturb direction around target ---
    r = p_C_I0 - p_G_I0
    r_len = r.length
    if r_len < 1e-9:
        # degenerate: invent a radius
        r = Vector((0.0, 0.0, 1.0))
        r_len = 1.0
    r_hat = (r / r_len).normalized()

    # choose a random axis perpendicular to r_hat (tangent direction)
    a = _rand_unit_vec(rng)
    # remove component along r_hat
    a = (a - a.dot(r_hat) * r_hat)
    if a.length < 1e-9:
        # fallback: pick any perpendicular
        a = r_hat.orthogonal()
    a.normalize()

    # rotate r_hat by a small angle around axis a
    ang = math.radians(rng.uniform(-cam_dir_max_deg, cam_dir_max_deg))
    dq_cam_dir = Quaternion(a, ang).normalized()
    r_hat_1 = (dq_cam_dir @ r_hat).normalized()

    # optional radial scale
    if cam_radius_scale_sigma > 0.0:
        scale = max(1e-6, rng.gauss(1.0, cam_radius_scale_sigma))
    else:
        scale = 1.0

    r1 = r_hat_1 * (r_len * scale)
    p_C_I1 = p_G_I0 + r1

    # --- camera orientation ---
    if force_camera_lookat:
        look_dir = (p_G_I0 - p_C_I1)
        if look_dir.length < 1e-12:
            look_dir = Vector((0.0, 0.0, -1.0))
        else:
            look_dir.normalize()
        q_I_C1 = look_dir.to_track_quat('-Z', 'Y').normalized()
    else:
        # small random perturbation of existing q_I_C0
        dq_c = _small_random_rotation(rng, cam_dir_max_deg)
        q_I_C1 = (dq_c @ q_I_C0).normalized()

    return {
        "p_G_I": p_G_I1,
        "q_I_G": q_I_G1,
        "p_C_I": p_C_I1,
        "q_I_C": q_I_C1,
    }
def clear_anim(obj):
    if obj.animation_data:
        obj.animation_data_clear()

def keyframe_pose(obj, frame):
    obj.keyframe_insert(data_path="location", frame=frame)
    if obj.rotation_mode == 'QUATERNION':
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    else:
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

def format_R_RPO(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"R{int(round(value))}"
    # one decimal place, replace '.' with 'p'
    return f"R{str(round(value, 1)).replace('.', 'p')}"

def get_timestamp_folder():
    return datetime.now().strftime("%Y-%m-%d_%H%M")

def scale_object_by_factor(obj, factor):
    old_scale = obj.scale.copy()
    obj.scale = Vector((old_scale.x * factor,
                        old_scale.y * factor,
                        old_scale.z * factor))
    
    print(f"Scaled {obj.name}: {old_scale} -> {obj.scale}")
def vprint(msg: str, verbose: bool = True):
    if verbose:
        print(msg)

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
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
def get_world_bounds(obj) -> Tuple[Vector, Vector]:
    """Return min and max corners of object and children in world coordinates."""
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

def compute_center_of_mass(obj) -> Vector:
    """Return center of mass of all mesh vertices."""
    verts = []
    for o in [obj] + list(obj.children_recursive):
        if o.type == "MESH":
            mesh = o.to_mesh()
            for v in mesh.vertices:
                verts.append(o.matrix_world @ v.co)
            o.to_mesh_clear()
    
    if not verts:
        return Vector((0, 0, 0))
    com = sum(verts, Vector()) / len(verts)
    return com

def set_scale(obj, scale) -> None:
    """Set object scale."""
    if isinstance(scale, (float, int)):
        obj.scale = Vector((scale, scale, scale))
    else:
        obj.scale = Vector(scale)
    bpy.context.view_layer.update()

def set_position(obj, xyz: Vector) -> None:
    """Set object position."""
    obj.location = Vector(xyz)
    bpy.context.view_layer.update()

def set_rotation_euler_deg(obj, xyz_deg: List[float]) -> None:
    """Set object rotation in degrees (XYZ order)."""
    obj.rotation_euler = Euler([math.radians(v) for v in xyz_deg], 'XYZ')
    bpy.context.view_layer.update()

def look_at(obj, target_point: Vector) -> None:
    direction = (target_point - obj.location).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_quaternion = quat
    bpy.context.view_layer.update()

def quat_wxyz_to_quat(q_wxyz) -> Quaternion:
    """Convert (w,x,y,z) tuple to mathutils.Quaternion."""
    w, x, y, z = q_wxyz
    return Quaternion((w, x, y, z)).normalized()

def quat_to_wxyz(q: Quaternion) -> tuple:
    """Convert mathutils.Quaternion to (w,x,y,z) tuple."""
    q = q.normalized()
    return (q.w, q.x, q.y, q.z)

def make_T_from_q_t(q: Quaternion, t: Vector) -> Matrix:
    """Build 4x4 transform T from quaternion and translation."""
    R = q.to_matrix().to_4x4()
    T = R.copy()
    T.translation = t
    return T

def decompose_T(T: Matrix) -> Tuple[Vector, Quaternion]:
    """Extract translation and rotation from 4x4 matrix."""
    t = T.to_translation()
    R = T.to_3x3()
    q = R.to_quaternion()
    q.normalize()
    return t, q
def set_sun_direction(sun_obj, sun_az_deg: float, sun_el_deg: float):
        """Orient a Sun object to match given azimuth/elevation (no location change) for ray-casting day-night."""
        # Convert to radians
        az = math.radians(sun_az_deg)
        el = math.radians(sun_el_deg)

        # Direction vector in world coordinates
        d = Vector((
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ))
        d.normalize()

        # Point sun's -Z axis along direction d
        sun_obj.rotation_mode = 'QUATERNION'
        sun_obj.rotation_quaternion = d.to_track_quat('-Z', 'Y')

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
                        azel: Dict,
                        sun_key: str,
                        N_azel_keys:int,
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
        sun_az = azel["sun_az_deg"]
        sun_el = azel["sun_el_deg"]
        
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
        sun_tag = f"sun_{str(sun_key).zfill(N_azel_keys)}"
        #stem = f"{frame_id:04d}_{exp_tag}_{sun_tag}_{mode_suffix}"
        stem = f"{str(frame_id).zfill(N_digits)}_{sun_tag}"
        
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
                     azel: Dict,
                    sun_key: str,
                    N_azel_keys:int,
                    N_digits:int) -> None:
        clear_anim(cam)
        clear_anim(model)
        self.scene.frame_start = frame_id1
        self.scene.frame_end = frame_id1+1
        self.scene.frame_set(frame_id1)
        sun_az = azel["sun_az_deg"]
        sun_el = azel["sun_el_deg"]
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
        sun_tag = f"sun_{str(sun_key).zfill(N_azel_keys)}"
        #stem = f"{frame_id:04d}_{exp_tag}_{sun_tag}_{mode_suffix}"
        stem = f"{str(frame_id1).zfill(N_digits)}_{sun_tag}_blurred"
        self.scene.render.filepath = str(output_dir / f"frame_{stem}")
        bpy.ops.render.render(write_still=True)
        
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
    renders_base_dir = PROJECT_ROOT / "renders" / timestamp
    gt_path = renders_base_dir / "camera_traj.txt"
    ensure_dir(gt_path.parent)
    if not gt_path.exists():
        write_camera_trajectory_v2(
            str(gt_path),
            N=config.setup.num_frames,
            R_LEO=config.setup.R_LEO,  # Target orbital radius (LEO)
            R_RPO=config.setup.R_RPO,   # Radial distance from target to camera (RPO)
        )
        # write_camera_approach(
        #     str(gt_path),
        #     config.setup.num_frames,
        #     config.setup.R_LEO,
        #     config.setup.R_RPO,
        #     config.setup.R_RPO/3,
        #     n_revs= 0.2,              # number of orbital revolutions over N frames
        #     orbit_plane = "xy",          # "xy", "xz", or "yz"
        #     shuffle_points = False,     # kept for symmetry; not used for trajectories
        #     seed = 0,
        #     verbose = True,
        #     include_rpo_column = True
        # )
    # Load trajectory
    #frames = load_camera_trajectory(str(gt_path))
    frames = load_camera_trajectory_v2(str(gt_path))
    print(f"\n[Session] Timestamp: {timestamp}")
    print(f"[Session] Renders output: {renders_base_dir}/")
    # Get models to render
    models = renderer.select_models_to_render()
    vprint(f"Rendering {len(models)} models: {[m.name for m in models]}", True)
    
    # Determine frame IDs
    frame_ids = config.frame_ids if config.frame_ids else list(range(len(frames)))
    R_RPO_tag = format_R_RPO(config.setup.R_RPO)
    res_x, res_y = config.camera.resolution
    # Render
    traj_cfg = config.setup

    # Exposure sweeps
    if traj_cfg.sweep_exposure is not None:
        exp_sweep_map = traj_cfg.sweep_exposure  # keys: "00", "01", ...
    else:
        # default: single key "00" with base exposure
        exp_sweep_map = {"00": config.setup.t_ref_s}

    # Sun az/el sweeps
    if traj_cfg.sweep_sun_az_el is not None:
        sun_sweep_map = traj_cfg.sweep_sun_az_el  # keys: "00", "01", ... each with {sun_az_deg, sun_el_deg}
    else:
        # sun_sweep_map = {
        #     "00": {"sun_az_deg": config.setup.sun_az_ref,
        #         "sun_el_deg": config.setup.sun_el_ref}
        # }
        sun_sweep_map = sun_sweep_90()
    N_digits = int(math.log10(len(frames))) + 1

    N_azel_keys = int(math.log10(len(sun_sweep_map))) + 1
    N_exp_keys =  int(math.log10(len(exp_sweep_map))) + 1
    # exp_sweep_map = config.setup.sweep_exposure if config.setup.sweep_exposure else {}
    # sun_sweep_map = config.setup.sweep_sun_az_el if config.setup.sweep_sun_az_el else {}
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

        gt_npz_dir   = ensure_dir(gt_root / "NPZ")
        gt_depth_dir = ensure_dir(gt_root / "Depth")
        gt_norm_dir  = ensure_dir(gt_root / "Normal")
        gt_flow_dir  = ensure_dir(gt_root / "Flow")
        gt_seg_dir   = ensure_dir(gt_root / "Seg")
        if config.model_rotation_z_deg != 0:
            renderer.rotate_z(model, config.model_rotation_z_deg)

        total = len(frame_ids) * len(exp_sweep_map) * len(sun_sweep_map)/2
        print("Enabling blur is: ",  config.setup.enable_blur)
        
        with tqdm(total=total, desc=f"Rendering {model.name}") as pbar:
            for i in frame_ids:
                fr = frames[i]
                base_seed = 12345 + i
                if str(config.setup.enable_blur).casefold()=="on":
                    fake_fr2 = make_fake_frame_from_frame0(
                        fr,
                        seed=base_seed ,
                        cam_dir_max_deg=0.6*config.setup.blur_motion_factor,          # tune (try 0.5–3.0 if blur is subtle)
                        cam_radius_scale_sigma=0.01*config.setup.blur_motion_factor,  # try 0.01 for slight radial change
                        target_rot_max_deg=1.0*config.setup.blur_motion_factor,       # tune (0–5 deg)
                        force_camera_lookat=True,
                    )
                else:
                    fake_fr2 = None
                for sweep_key_exp, exp_value in exp_sweep_map.items():
                    output_dir = model_out_dir / f"exp_{str(sweep_key_exp).zfill(N_exp_keys)}"
                    for sweep_key_azel, azel in sun_sweep_map.items():
                        if int(sweep_key_azel)<4:
                            fps = renderer.scene.render.fps / renderer.scene.render.fps_base
                            shutter_frames = exp_value * fps * config.setup.blur_shutter_factor # physical-ish shutter in frames
                            if str(config.setup.enable_blur).casefold()=="on" and fake_fr2 is not None:
                                renderer.render_frame_motion_blur_traj(
                                    cam, model, sun,
                                    fr, fake_fr2,i,
                                    shutter_frames,
                                    output_dir,
                                    exp_value,
                                    azel,
                                    sweep_key_azel,
                                    N_azel_keys,
                                    N_digits
                                )
                            else:
                                renderer.render_frame_v2(cam,model,sun, fr,i, output_dir,exp_value,
                                    azel, sweep_key_azel,N_azel_keys, N_digits)
                            
                            pbar.update(1)
                # VisionBlender writes npz alongside the rendered frame:
                npz_src = Path(os.path.join( output_dir , f'{i:04d}.npz'))

                handle_gt_from_npz(
                    npz_src,
                    gt_npz_dir,
                    gt_depth_dir,
                    gt_norm_dir,
                    gt_flow_dir,
                    gt_seg_dir,
                    config.setup.R_RPO
                )
    pbar.close()

                   
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

