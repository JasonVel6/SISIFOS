import math
import random
from pathlib import Path

import numpy as np
from mathutils import Quaternion, Vector

from modules.io_utils import ensure_dir
from modules.log_utils import get_logger
from modules.trajectory.plot_figure import generate_scene_plots
from modules.trajectory.trajectory_io import write_camera_trajectory
from modules.trajectory.trajectory_math import (
    _rand_quat_uniform,
    _rand_unit_vec,
    _small_random_rotation,
    fibonacci_sphere,
    quat_to_wxyz,
)

logger = get_logger()


def make_fake_frame_from_frame0(
    frame0: dict,
    seed: int,
    cam_dir_max_deg: float = 0.5,  # how much to "swing" camera direction around target
    cam_radius_scale_sigma: float = 0.0,  # optional: random radial scaling (e.g. 0.01 = 1%)
    target_rot_max_deg: float = 0.0,  # optional: random target rotation
    force_camera_lookat: bool = True,  # recommended
) -> dict:
    """
    Create an artificial 'next frame' using only frame0, by perturbing camera pose
    (and optionally target orientation). Keeps target position fixed.

    TODO we may want to find another place for this method

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
    a = a - a.dot(r_hat) * r_hat
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
        look_dir = p_G_I0 - p_C_I1
        if look_dir.length < 1e-12:
            look_dir = Vector((0.0, 0.0, -1.0))
        else:
            look_dir.normalize()
        q_I_C1 = look_dir.to_track_quat("-Z", "Y").normalized()
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
        logger.info("Trajectory (v2 - inertial orbital) written to: %s", out_path)

    return [out_dir]


def write_camera_trajectory_const_rotation(
    out_dir: str,
    R_LEO: float,
    R_RPO: float,
    tstep: float,
    tend: float,
    angular_velocity: tuple[float, float, float] | None = (0.0, 0.0, 1.0),
    sun_az: float = 0.0,
    sun_el: float = 0.0,
    verbose: bool = True,
) -> list[str]:
    """
    Generate a fixed camera/target geometry where the camera always looks at the
    target and the target rotates at constant angular velocity.

    The geometry mirrors the "back-side" Fibonacci samples: both target and
    camera lie on the same radial line away from Earth, with the target farther
    from the origin so the camera renders outward rather than toward Earth.
    """
    if tstep <= 0.0:
        raise ValueError(f"tstep must be positive, got {tstep}")
    if tend < 0.0:
        raise ValueError(f"tend must be non-negative, got {tend}")
    if R_RPO <= 0.0:
        raise ValueError(f"R_RPO must be positive, got {R_RPO}")
    if R_LEO <= R_RPO:
        raise ValueError(f"R_LEO must be greater than R_RPO, got R_LEO={R_LEO}, R_RPO={R_RPO}")

    timestamps = np.arange(0.0, tend + tstep, tstep, dtype=float)
    num_steps = len(timestamps)

    p_G_I = np.zeros((num_steps, 3), dtype=float)
    q_IG_wxyz = np.zeros((num_steps, 4), dtype=float)
    p_C_I = np.zeros((num_steps, 3), dtype=float)
    q_IC_wxyz = np.zeros((num_steps, 4), dtype=float)
    sun_az_arr = np.full((num_steps,), float(sun_az), dtype=float)
    sun_el_arr = np.full((num_steps,), float(sun_el), dtype=float)

    direction_radial = Vector((0.0, 0.0, -1.0))
    p_G_vec = direction_radial * R_LEO
    p_C_vec = direction_radial * (R_LEO - R_RPO)
    look_direction = (p_G_vec - p_C_vec).normalized()
    q_IC = look_direction.to_track_quat("-Z", "Y").normalized()
    q_IC_wxyz_single = quat_to_wxyz(q_IC)

    omega_vec = Vector(angular_velocity if angular_velocity is not None else (0.0, 0.0, 0.0))
    omega_mag = omega_vec.length
    if omega_mag > 0.0:
        omega_axis = omega_vec.normalized()
    else:
        omega_axis = None

    q_IG0 = Quaternion((1.0, 0.0, 0.0, 0.0))

    p_G_I[:] = (p_G_vec.x, p_G_vec.y, p_G_vec.z)
    p_C_I[:] = (p_C_vec.x, p_C_vec.y, p_C_vec.z)
    q_IC_wxyz[:] = q_IC_wxyz_single

    for i, timestamp in enumerate(timestamps):
        if omega_axis is None:
            q_IG = q_IG0
        else:
            q_delta = Quaternion(omega_axis, omega_mag * float(timestamp)).normalized()
            q_IG = (q_delta @ q_IG0).normalized()
        q_IG_wxyz[i] = quat_to_wxyz(q_IG)

    ensure_dir(Path(out_dir))
    out_path = write_camera_trajectory(
        output_dir=out_dir,
        nbSteps=num_steps,
        timestamps=timestamps,
        r_GO_I=-p_G_I,
        q_IG=q_IG_wxyz,
        r_CO_I=-p_C_I,
        q_IC=q_IC_wxyz,
        sun_az_I=sun_az_arr,
        sun_el_I=sun_el_arr,
    )

    plot_out_path = Path(out_dir) / "trajectory_plot.png"
    generate_scene_plots(
        output_dir=str(plot_out_path),
        p_G_I=p_G_I,
        p_C_I=p_C_I,
        r_CG_arr=p_G_I - p_C_I,
        q_IG_arr=q_IG_wxyz,
        q_IC_arr=q_IC_wxyz,
        sun_az_I=sun_az_arr,
        sun_el_I=sun_el_arr,
        timestamps=timestamps,
    )

    if verbose:
        logger.info("Constant-rotation trajectory written to: %s", out_path)

    return [out_dir]
