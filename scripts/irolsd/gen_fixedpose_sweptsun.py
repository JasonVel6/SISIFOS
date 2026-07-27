#!/usr/bin/env python3
"""Fixed-pose / swept-Sun camera_traj.csv generator for IRoLSD training data.

Emits a SISIFOS-compatible dataset folder whose ``camera_traj.csv`` holds the
CAMERA + TARGET POSE FIXED across a group of frames and sweeps only the SUN
azimuth/elevation within the group. Stock SISIFOS renders a *tumbling*
trajectory (pose changes every frame, Sun constant); IRoLSD training needs the
opposite (pose constant per group, Sun perturbed) so the label-gen can build an
illumination-robust line heatmap (paper Eq. 23) from same-pose variants.

We do NOT touch SISIFOS's dynamical trajectory generator. Instead we write the
CSV directly and point the render at it via ``trajectory_type: "filepath"``.

Frame layout (matches ``irolsd/label_gen`` DATA CONTRACT):

    frame index n = g * NVARIANTS + i
        g = pose group (0 .. NPOSES-1), one camera viewpoint each
        i = illumination variant (0 = canonical Sun, 1..NVARIANTS-1 = perturbed)

SISIFOS renders each row of the CSV to ``frame_{n:0Nd}.png`` and (with
save_depth) an NPZ carrying the raw float depth. The companion
``pack_contract.py`` remaps those into the flat ``img_{n}.png`` / ``depth_{g}.npy``
layout the label-gen consumes.

GEOMETRY (scale-safe): target G is placed at the inertial origin, camera C at
``R0 * view_dir`` looking back at the origin. SISIFOS's
``get_scaled_trajectory_in_ECI`` scales camera position by earth_dist_scale_factor
but re-adds the full C->G relative vector, so the RANGE and VIEW DIRECTION are
preserved exactly regardless of that factor.

Pose viewpoints: Fibonacci sphere (even aspect coverage) + small deterministic
jitter, so 1000 groups tile the viewing sphere without banding. The target's own
orientation q_I_G is a fixed identity (the *camera* moves around the target);
aspect diversity comes from the viewpoint, which is equivalent and keeps the Sun
azimuth/elevation interpretation stable across groups.

Run OUTSIDE Blender (plain Python + numpy):

    python3 scripts/irolsd/gen_fixedpose_sweptsun.py \
        --out renders/_phased/bepi_train_src/Config_1_RF_Bepi-mcs/Agent_0 \
        --nposes 1000 --nvariants 20 --range 40.0 --seed 12345
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np


# ---------------------------------------------------------------------------
# Minimal numpy quaternion helpers (wxyz), no bpy/mathutils dependency.
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def look_at_quat_wxyz(forward: np.ndarray, up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Quaternion (w,x,y,z) aligning camera -Z with ``forward`` and +Y with up.

    Mirrors mathutils ``to_track_quat('-Z', 'Y')``: camera looks along -Z, so the
    camera's forward (look) direction is ``forward`` and -Z maps onto it.
    Builds the rotation matrix [right, up, back] with back = -forward, then
    converts to a quaternion. Falls back to a different up hint at the poles.
    """
    f = _normalize(np.asarray(forward, dtype=float))
    up = np.asarray(up_hint, dtype=float)
    if abs(float(np.dot(f, _normalize(up)))) > 0.999:
        # forward nearly parallel to up hint -> pick an orthogonal hint
        up = np.array([0.0, 1.0, 0.0])
    back = -f  # camera +Z points backward (opposite the look direction)
    right = _normalize(np.cross(up, back))
    true_up = np.cross(back, right)
    # Columns are the camera axes expressed in world coords.
    R = np.column_stack((right, true_up, back))
    return _mat_to_quat_wxyz(R)


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w,x,y,z). Shepperd's method."""
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` roughly-even unit directions on the sphere (N x 3)."""
    if n <= 0:
        return np.zeros((0, 3))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    pts = np.zeros((n, 3), dtype=float)
    for i in range(n):
        y = 1.0 - (2.0 * i) / (n - 1) if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        pts[i] = (math.cos(theta) * r, y, math.sin(theta) * r)
    return pts


def sun_azel_from_dir(u: np.ndarray) -> tuple[float, float]:
    """World unit vector -> (az_deg, el_deg) matching set_sun_direction()."""
    u = _normalize(u)
    el = math.degrees(math.asin(max(-1.0, min(1.0, u[2]))))
    az = math.degrees(math.atan2(u[1], u[0])) % 360.0
    return az, el


CSV_HEADER = [
    "timestamp",
    "p_G_I_x", "p_G_I_y", "p_G_I_z",
    "q_I_G_w", "q_I_G_x", "q_I_G_y", "q_I_G_z",
    "p_C_I_x", "p_C_I_y", "p_C_I_z",
    "q_I_C_w", "q_I_C_x", "q_I_C_y", "q_I_C_z",
    "sun_az", "sun_el",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Agent output dir; camera_traj.csv is written here.")
    ap.add_argument("--nposes", type=int, default=1000, help="Number of fixed camera-pose groups (default 1000).")
    # --- sharding: render a contiguous slice of a larger global pose sphere, so
    #     N shards can run as N concurrent embers jobs and pack into one dataset.
    ap.add_argument("--total-poses", type=int, default=None,
                    help="If set, the GLOBAL Fibonacci sphere size; this shard renders --pose-count poses "
                         "starting at --pose-start of that global sphere. Frame indices are GLOBAL "
                         "(g_global*nvariants + i) so shards pack together without collision. "
                         "When unset, behaves as a standalone run of --nposes.")
    ap.add_argument("--pose-start", type=int, default=0, help="First global pose index for this shard (needs --total-poses).")
    ap.add_argument("--pose-count", type=int, default=None, help="Number of poses in this shard (needs --total-poses).")
    ap.add_argument("--nvariants", type=int, default=20,
                    help="Illumination variants per pose incl. canonical i=0 (default 20 = paper's number). "
                         "MUST equal the label-gen's NVARIANTS grouping.")
    ap.add_argument("--range", type=float, default=40.0, dest="range_m",
                    help="Fixed camera range to the target in metres (default 40 m ~ Bepi half-frame).")
    ap.add_argument("--sun-delta-deg", type=float, default=1.0,
                    help="Half-width of the per-variant Sun az/el perturbation, deg (paper delta in [-1,1]).")
    ap.add_argument("--sun-base-az", type=float, default=45.0, help="Canonical Sun azimuth per group, deg (base).")
    ap.add_argument("--sun-base-el", type=float, default=25.0, help="Canonical Sun elevation per group, deg (base).")
    ap.add_argument("--sun-group-jitter-deg", type=float, default=40.0,
                    help="Random per-GROUP shift of the canonical Sun az/el so groups see varied lighting (deg). "
                         "Illumination variants stay within +/- sun-delta-deg of that group base.")
    ap.add_argument("--pose-jitter-deg", type=float, default=3.0,
                    help="Small random angular jitter added to each Fibonacci viewpoint (deg) to break banding.")
    ap.add_argument("--seed", type=int, default=12345, help="Master RNG seed (reproducible).")
    ap.add_argument("--tstep", type=float, default=0.1, help="Nominal timestamp increment written to the CSV.")
    args = ap.parse_args()

    if args.nvariants < 1:
        ap.error("--nvariants must be >= 1")
    if args.nposes < 1:
        ap.error("--nposes must be >= 1")

    os.makedirs(args.out, exist_ok=True)

    # Resolve shard slice. Sharding renders a contiguous slice [pose_start,
    # pose_start+pose_count) of a GLOBAL sphere of `sphere_n` poses. Each pose's
    # RNG is seeded by (seed, g_global) so the pose is bit-identical regardless
    # of which shard renders it -> shards pack together as one dataset.
    if args.total_poses is not None:
        sphere_n = args.total_poses
        pose_start = args.pose_start
        pose_count = args.pose_count if args.pose_count is not None else (sphere_n - pose_start)
        if pose_start < 0 or pose_start + pose_count > sphere_n:
            ap.error(f"shard slice [{pose_start},{pose_start+pose_count}) out of range for total-poses={sphere_n}")
    else:
        sphere_n = args.nposes
        pose_start, pose_count = 0, args.nposes

    viewpoints = fibonacci_sphere(sphere_n)  # (sphere_n, 3) unit dirs, GLOBAL sphere

    rows = []
    n_total = pose_count * args.nvariants
    for g in range(pose_start, pose_start + pose_count):
        # per-pose RNG keyed on GLOBAL pose index -> shard-invariant draws.
        rng = np.random.default_rng([args.seed, g])

        # --- fixed camera pose for this group -------------------------------
        view_dir = viewpoints[g].copy()
        # small angular jitter: rotate view_dir by a random small rotation
        if args.pose_jitter_deg > 0.0:
            axis = _normalize(rng.standard_normal(3))
            ang = math.radians(rng.uniform(-args.pose_jitter_deg, args.pose_jitter_deg))
            view_dir = _rotate_vec(view_dir, axis, ang)
        view_dir = _normalize(view_dir)

        # target at origin; camera at range along view_dir, looking back at origin.
        p_G_I = np.zeros(3)
        q_I_G = np.array([1.0, 0.0, 0.0, 0.0])  # target identity orientation
        p_C_I = view_dir * args.range_m
        forward = -view_dir  # camera looks from C toward G (origin)
        q_I_C = look_at_quat_wxyz(forward)

        # --- per-group canonical Sun base (jittered so lighting varies) ------
        base_az = args.sun_base_az + rng.uniform(-args.sun_group_jitter_deg, args.sun_group_jitter_deg)
        base_el = args.sun_base_el + rng.uniform(-args.sun_group_jitter_deg, args.sun_group_jitter_deg)
        base_el = max(-89.0, min(89.0, base_el))

        for i in range(args.nvariants):
            n = g * args.nvariants + i  # GLOBAL frame index
            if i == 0:
                daz, del_ = 0.0, 0.0  # canonical variant is the unperturbed base
            else:
                daz = rng.uniform(-args.sun_delta_deg, args.sun_delta_deg)
                del_ = rng.uniform(-args.sun_delta_deg, args.sun_delta_deg)
            sun_az = (base_az + daz) % 360.0
            sun_el = max(-89.0, min(89.0, base_el + del_))

            rows.append([
                n * args.tstep,
                p_G_I[0], p_G_I[1], p_G_I[2],
                q_I_G[0], q_I_G[1], q_I_G[2], q_I_G[3],
                p_C_I[0], p_C_I[1], p_C_I[2],
                q_I_C[0], q_I_C[1], q_I_C[2], q_I_C[3],
                round(sun_az, 3), round(sun_el, 3),
            ])

    csv_path = os.path.join(args.out, "camera_traj.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

    # provenance sidecar so the pack step + label-gen can read NVARIANTS/NPOSES
    meta_path = os.path.join(args.out, "sweep_meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump({
            "kind": "fixedpose_sweptsun",
            "nposes": pose_count,
            "nvariants": args.nvariants,
            "n_total_frames": n_total,
            # sharding provenance: SISIFOS names frames by LOCAL CSV row (0..n_total-1),
            # so the packer maps local frame -> global via pose_start:
            #   g_global = pose_start + (local_frame // nvariants)
            "sharded": args.total_poses is not None,
            "total_poses": sphere_n,
            "pose_start": pose_start,
            "range_m": args.range_m,
            "sun_delta_deg": args.sun_delta_deg,
            "sun_base_az": args.sun_base_az,
            "sun_base_el": args.sun_base_el,
            "sun_group_jitter_deg": args.sun_group_jitter_deg,
            "pose_jitter_deg": args.pose_jitter_deg,
            "seed": args.seed,
            "naming_rule": "LOCAL frame n = local_g*nvariants + i; g_global = pose_start + local_g; canonical i=0",
        }, f, indent=2)

    print(f"[gen] wrote {n_total} frames ({pose_count} poses [{pose_start}..{pose_start+pose_count}) "
          f"of {sphere_n} x {args.nvariants} variants)")
    print(f"[gen]   camera_traj.csv -> {csv_path}")
    print(f"[gen]   sweep_meta.json -> {meta_path}")
    print(f"[gen]   range={args.range_m} m  sun_delta=+/-{args.sun_delta_deg} deg  seed={args.seed}")


def _rotate_vec(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation of vector v about unit axis by angle ang (rad)."""
    axis = _normalize(axis)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


if __name__ == "__main__":
    main()
