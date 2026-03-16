import math
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from mathutils import Vector, Quaternion
from modules.trajectory.trajectory_math import fibonacci_sphere, _rand_quat_uniform, quat_to_wxyz
from modules.io_utils import ensure_dir
from modules.trajectory.trajectory_io import write_camera_trajectory


def write_camera_trajectory_fib(
    out_dir: str,
    N: int,
    R_LEO: float,
    R_RPO: float,
    sun_az: float = 0.0,
    sun_el: float = 0.0,
    shuffle_points: bool = False,
    seed: int = 0,
    verbose: bool = True,
) -> list[str]:
    """
    Generate camera Fibonacci-style trajectory in ECI arrays and write using
    the shared trajectory writer.

    Target (G) orbits on sphere at radius R_LEO.
    Camera (C) is positioned radially outward from target at distance R_RPO,
    looking back toward origin.
    Both use same Fibonacci sphere angles.

    Earth/clouds/atmosphere stay at inertial origin (0,0,0) with fixed orientation.
    """
    sphere_pts = fibonacci_sphere(N, radius=1.0)

    if shuffle_points:
        random.seed(seed)
        random.shuffle(sphere_pts)

    num_steps = 2 * N
    p_G_I = np.zeros((num_steps, 3), dtype=float)
    q_IG_wxyz = np.zeros((num_steps, 4), dtype=float)
    p_C_I = np.zeros((num_steps, 3), dtype=float)
    q_IC_wxyz = np.zeros((num_steps, 4), dtype=float)
    sun_az_arr = np.full((num_steps,), float(sun_az), dtype=float)
    sun_el_arr = np.full((num_steps,), float(sun_el), dtype=float)

    for i in range(N):
        idx_front = 2 * i
        idx_back = idx_front + 1

        # Fibonacci sphere point (unit direction)
        direction_radial = sphere_pts[i].normalized()

        # Target position at radius R_LEO
        p_G_I_front = direction_radial * R_LEO

        # Camera position at radius R_LEO + R_RPO (same direction)
        p_C_I_front = direction_radial * (R_LEO + R_RPO)

        # Random target orientation in inertial frame
        rng = random.Random(seed + i)
        q_IG = _rand_quat_uniform(rng)

        # Camera orientation: looking toward origin (nadir/Earth-pointed)
        # Direction from camera toward sc/origin (negative radial direction)
        look_direction = -direction_radial  # Points toward Earth at origin

        # Create quaternion that aligns -Z axis (camera forward) with look direction
        q_IC = look_direction.to_track_quat("-Z", "Y").normalized()

        # Convert quaternions to wxyz format
        q_IG_wxyz_front = quat_to_wxyz(q_IG)
        q_IC_wxyz_front = quat_to_wxyz(q_IC)

        p_G_I[idx_front] = (p_G_I_front.x, p_G_I_front.y, p_G_I_front.z)
        q_IG_wxyz[idx_front] = q_IG_wxyz_front
        p_C_I[idx_front] = (p_C_I_front.x, p_C_I_front.y, p_C_I_front.z)
        q_IC_wxyz[idx_front] = q_IC_wxyz_front

        # Fibonacci sphere point (unit direction)
        direction_radial = -sphere_pts[i].normalized()

        # Target position at radius R_LEO
        p_G_I_back = direction_radial * R_LEO

        # Camera position at radius R_LEO - R_RPO (no Earth)
        p_C_I_back = direction_radial * (R_LEO - R_RPO)

        # Random target orientation in inertial frame
        rng = random.Random(seed + i)
        q_IG = _rand_quat_uniform(rng)

        # Camera orientation: looking toward OUT (-nadir/Earth-pointed from the other side)
        # Direction from camera toward sc (negative of negative already radial direction)
        look_direction = direction_radial  # Points toward Earth at origin

        # Create quaternion that aligns -Z axis (camera forward) with look direction
        q_IC = look_direction.to_track_quat("-Z", "Y").normalized()

        # Convert quaternions to wxyz format
        q_IG_wxyz_back = quat_to_wxyz(q_IG)
        q_IC_wxyz_back = quat_to_wxyz(q_IC)

        p_G_I[idx_back] = (p_G_I_back.x, p_G_I_back.y, p_G_I_back.z)
        q_IG_wxyz[idx_back] = q_IG_wxyz_back
        p_C_I[idx_back] = (p_C_I_back.x, p_C_I_back.y, p_C_I_back.z)
        q_IC_wxyz[idx_back] = q_IC_wxyz_back

    ensure_dir(Path(out_dir))
    out_path = write_camera_trajectory(
        output_dir=out_dir,
        nbSteps=num_steps,
        timestamps=np.linspace(0.0, (num_steps - 1) * 0.5, num=num_steps),  # e.g. 0.5s step
        r_GO_I=-p_G_I,
        q_IG=q_IG_wxyz,
        r_CO_I=-p_C_I,
        q_IC=q_IC_wxyz,
        sun_az_I=sun_az_arr,
        sun_el_I=sun_el_arr,
    )

    if verbose:
        print(f"Trajectory (v2 - inertial orbital) written to: {out_path}")

    return [out_dir]


def write_camera_approach(
    out_path: str,
    N: int,
    R_LEO: float,
    R_RPO_start: float = 36.0,
    R_RPO_end: float = 5.0,
    n_revs: float = 1.0,  # number of orbital revolutions over N frames
    orbit_plane: str = "xy",  # "xy", "xz", or "yz"
    shuffle_points: bool = False,
    seed: int = 0,
    sun_az: float = 0.0,
    sun_el: float = 0.0,
    verbose: bool = True,
    include_rpo_column: bool = True,
) -> list[str]:
    """
    This is a demo approach phase. Not actually implementing the trajectory module
    """

    out_path = str(Path(out_path) / "camera_traj.txt")

    if N < 2:
        rpo_list = [float(R_RPO_start)]
    else:
        rpo_list = [float(R_RPO_start + (R_RPO_end - R_RPO_start) * (i / (N - 1))) for i in range(N)]

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
        "# p_G_I(xyz)  q_I_G(wxyz)  p_C_I(xyz)  q_I_C(wxyz)  sun_az  sun_el"
        + ("  R_RPO" if include_rpo_column else ""),
        "#",
    ]

    # Orbit angle over time
    # theta spans n_revs revolutions
    lag_ratio = -0.5
    for i in range(N):
        t = 0.0 if N < 2 else (i / (N - 1))
        theta = 2.0 * math.pi * n_revs * t
        if i < N // 2:
            lag_ratio += 0.75 / N
        else:
            lag_ratio -= 0.5 / N

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
            t_hat = Vector((-math.sin(theta), math.cos(theta), 0.0))
        elif orbit_plane == "xz":
            t_hat = Vector((-math.sin(theta), 0.0, math.cos(theta)))
        elif orbit_plane == "yz":
            t_hat = Vector((0.0, -math.sin(theta), math.cos(theta)))
        else:
            raise ValueError("orbit_plane must be one of: 'xy', 'xz', 'yz'")

        t_hat.normalize()
        u.normalize()

        # Target position on orbit
        p_G_I = u * R_LEO

        # Camera approaches target along the same radial line, staying OUTSIDE (Earth behind target)
        R_RPO_i = rpo_list[i]
        # p_C_I = u * (R_LEO + R_RPO_i)
        r_behind = lag_ratio * rpo_list[0]
        r_rad = math.sqrt(max(0.0, R_RPO_i * R_RPO_i - r_behind * r_behind))
        p_C_I = p_G_I + (u * r_rad) + (t_hat * r_behind)

        # Random target orientation per frame (same as your v2 idea)
        # rng = random.Random(seed + i)
        q_IG = Quaternion(
            (1.0, 0.0, 0.0, 0.0)
        ).normalized()  # should return a mathutils.Quaternion (wxyz or xyzw depending on your helper)
        q_IG_wxyz = quat_to_wxyz(q_IG)

        # Camera orientation: center-point at target
        look_dir = p_G_I - p_C_I
        if look_dir.length < 1e-12:
            # Degenerate (shouldn't happen unless R_RPO_i == 0)
            look_dir = -u
        else:
            look_dir.normalize()

        q_IC = look_dir.to_track_quat("-Z", "Y").normalized()
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
    return [out_path]
