import math
import random
from pathlib import Path
from typing import List, Dict
from mathutils import Vector, Quaternion
from geometry import fibonacci_sphere, _rand_quat_uniform, _small_random_rotation, _rand_unit_vec, quat_to_wxyz
from ..io_utils import ensure_dir

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