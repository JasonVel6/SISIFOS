#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generateTrajectoriesUnified.py — Unified trajectory generation and ground truth export

This script combines trajectory IC generation (formerly generateTrajectories.py) with
orbit propagation and ground truth file creation (formerly create_gtvalues.py) into
a single unified pipeline. No intermediate YAML files are generated.

FRAME CONVENTIONS:
- A = COM (pose origin used by UE5). We export r_AO_I and q_IG.
- G = SLAM body frame (same axes as A, different origin).
- Naming convention: r_YX = Y - X (vector from X to Y).
- I = inertial frame (ECI-like).
- C and S represent the same frame (camera/sensor frame).
- prefix state vectors with state_ to denote full 6-DOF state in inertial frame.
- prefix quaternions with q_ to denote quaternions.
- R_YX denotes rotation matrix from frame X to frame Y (i.e., R_YX * v_X = v_Y).
- prefix forces with f_
- r_AG_G = A - G (position of COM relative to SLAM frame origin, in G coords).
- Internal variable stores G - A for convenience; negated on output.
- Axes are body-fixed (G).
- 

OUTPUT FILES (per MC trial, per agent):
- gtValues.txt: Ground truth values for SLAM
- camera_traj.txt: SISFOS-compatible camera trajectory
- sensormeasurements.txt: Noisy sensor measurements
- Config.yaml: Configuration file

Usage:
    python generateTrajectoriesUnified.py [--seed SEED
"""

import os
import sys
import math
import json
import time
import shutil
import argparse
import numpy as np
from datetime import datetime
from scipy.linalg import expm

from motion_cases import (
    init_inertial, init_hill, init_tumbling,
    validate_omega_timeseries_excitation, sample_inertia_excited_omega_direction
)

# TODO Evaluate what we need in the math function
from math_function import (
    au2R, oe2cart, createHillFrame, propagate_orbit, parameterSetting,
    sk, R2q, q2R, solve_ne_equation, so3_log_vec, rodrigues, _vecI_to_azel,
    _seed_right, _lookat_continuous_RGS, _quat_hemi_continuous, enforce_quat_series_continuity
)
from ue5_function import calcInitCondChaser, read_gt_values, create_json
from plot_figure import plot_trial_trajectories

# ---------------- Paths ----------------
# TODO modify the paths to make more sense for SISFOS integration 
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SISIFOS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
OUTPUT_BASE = os.path.join(SISIFOS_DIR, "outputfile")

# ---------- Seed handling (reproducible MC) ----------
def resolve_master_seed() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    args, _ = parser.parse_known_args()
    if args.seed is not None:
        return int(args.seed)
    env_seed = os.getenv("SATSLAM_SEED")
    if env_seed:
        return int(env_seed)
    return int(time.time() * 1e6) & 0x7FFFFFFF


MASTER_SEED = resolve_master_seed()
ss_master = np.random.SeedSequence(MASTER_SEED)

# ----------- User-tunable defaults -----------
# TODO We should likely make a config file for these later
DEFAULT_r_AG_G = [0.0, 0.0, 0.0] # [0.1, 0.05, 0.15]

# Sensor noise (ASTRO APS3 star tracker and Astrix NS IMU at 10 Hz)
sigma_Rxy_aps3 = 0.8 * (np.pi / 180) / 3600  # rad (0.8 arcsec)
sigma_Rz_aps3 = 7.0 * (np.pi / 180) / 3600   # rad (7.0 arcsec)
sigma_omega = np.deg2rad(0.0025 / 60.0) * np.sqrt(10.0 / 2.0)
sigma_accel = 10.0 * 1e-6 * 9.80665 * np.sqrt(10.0 / 2.0)
COV_R_ASTRO_APS3 = [[sigma_Rxy_aps3**2, 0, 0], [0, sigma_Rxy_aps3**2, 0], [0, 0, sigma_Rz_aps3**2]]
COV_OMEGA_ASTRIX = [[sigma_omega**2, 0, 0], [0, sigma_omega**2, 0], [0, 0, sigma_omega**2]]
COV_ACCEL_ASTRIX = [[sigma_accel**2, 0, 0], [0, sigma_accel**2, 0], [0, 0, sigma_accel**2]]
MEAN_DEFAULT = [0.0, 0.0, 0.0]

# Bias models
GYRO_BIAS_SIGMA_DEGPHR = 0.005
GYRO_BIAS_TAU_S = 3600.0
ACCEL_BIAS_SIGMA_UG = 10.0
ACCEL_BIAS_TAU_S = 3600.0

# Camera
FOCAL_LENGTH_PX = 2500.0
CAMERA_RESOLUTION = 1024
LENS_FLARE = 0.01

# Distance to target
R0_const = 30.0

# Target model
SHAPE_MODEL_FILENAME = "../models/integral.obj"

# Sun alignment
SUN_ALIGN_ENABLE = True
SUN_ALIGN_CONE_DEG = 12.0
SUN_ALIGN_JITTER_D = 4.0

# Earth-in-background alignment: Sun -> Camera -> Target -> Earth
# This ensures Earth is visible behind the target in the initial frame
EARTH_BACKGROUND_ENABLE = True

# ---------- Constants / environment ----------
mu_ref = 3.986004418e14      # Earth mu (m^3/s^2)
h_orbit = 550e3              # circular altitude (m)
R_earth = 6371e3             # Earth radius (m)
a_ref = R_earth + h_orbit    # semi-major axis (m)
n_scalar = np.sqrt(mu_ref / a_ref**3)

# Time settings
IMAGE_MAX_DT_S = 1.0
tend = 500.0
tstep = 0.5
MIN_F2F_PX_MED = 3.0

# Inertia (ESA INTEGRAL satellite, box approximation)
# Dimensions: 2.8 x 3.2 x 5.0 m, mass ~4000 kg
# Triaxial inertia avoids axisymmetric observability degeneracy
m = 4000.0
l, w, h = 5.0, 3.2, 2.8  # x=longest axis (5m), y=3.2m, z=2.8m
J_xx = (1/12) * m * (w**2 + h**2)  # rotation about x (longest)
J_yy = (1/12) * m * (l**2 + h**2)  # rotation about y
J_zz = (1/12) * m * (l**2 + w**2)  # rotation about z
J = np.array([[J_xx, 0, 0], [0, J_yy, 0], [0, 0, J_zz]])

# ============================================================================
# MAIN Function
# ============================================================================
# TODO lets break this up into several methods for better modularity
def generate_trajectories(path_mode, rotMode_Gframe, num_agents, num_mc, rngs_mc):
    # ---------- Generate initial conditions ----------
    print("\n[STEP 1] Generating initial conditions...")

    if rotMode_Gframe == "1":
        print("  Inertial mode (CRO trajectory)")
        x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, _ = init_inertial(
            num_mc=num_mc, num_agents=num_agents, n_scalar=n_scalar,
            focal_length_px=FOCAL_LENGTH_PX, kf_dt=IMAGE_MAX_DT_S,
            px_min=MIN_F2F_PX_MED, rho_max=0.90, R0_const=R0_const,
            variant="cro", rngs_mc=rngs_mc
        )
    elif rotMode_Gframe == "2":
        print("  Hill mode")
        x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, _ = init_hill(
            num_mc=num_mc, num_agents=num_agents, n_scalar=n_scalar,
            rngs_mc=rngs_mc, focal_length_px=FOCAL_LENGTH_PX, kf_dt=IMAGE_MAX_DT_S,
            px_min=MIN_F2F_PX_MED, rho_max=0.90
        )
    elif rotMode_Gframe == "3":
        print("  Tumbling mode (CRO trajectory + target tumbling)")
        # Use faster tumbling (3-5 deg/s) for better inertia observability.
        # This is within the conservative design envelope (~5 deg/s upper bound).
        # Slower rates (0.5-2 deg/s default) have near-zero omega_dot, making
        # inertia poorly observable from Euler's equation I·ω̇ + ω×(I·ω) = 0.
        x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, _ = init_tumbling(
            num_mc=num_mc, num_agents=num_agents, n_scalar=n_scalar,
            rngs_mc=rngs_mc, focal_length_px=FOCAL_LENGTH_PX, kf_dt=IMAGE_MAX_DT_S,
            px_min=1.0, rho_max=0.95, R0_const=R0_const,
            omega_min_deg=3.0, omega_max_deg=5.0,
            J=J, min_asymmetry_component=0.4
        )

    # Ensure omega_GI_G_0 is a numpy array (tumbling returns (num_mc,3), others return [0,0,0])
    omega_GI_G_0 = np.atleast_2d(omega_GI_G_0)
    if omega_GI_G_0.shape[0] == 1 and num_mc > 1:
        # Broadcast single value to all MC trials
        omega_GI_G_0 = np.tile(omega_GI_G_0, (num_mc, 1))

    # Combine into r_0 array
    r_0 = np.zeros((num_mc, num_agents, 6))
    for i in range(num_mc):
        for a in range(num_agents):
            r_0[i, a] = [x_0[i, a], y_0[i, a], z_0[i, a],
                         xdot_0[i, a], ydot_0[i, a], zdot_0[i, a]]

    # ---------- Generate MC parameters ----------
    print("\n[STEP 2] Sampling orbital and attitude parameters...")

    inc = np.zeros(num_mc)
    ecc = np.zeros(num_mc)
    el_I = np.zeros(num_mc)
    az_I = np.zeros(num_mc)
    yaw = np.zeros(num_mc)
    pitch = np.zeros(num_mc)
    roll = np.zeros(num_mc)

    # TODO parameterize distributions
    for i in range(num_mc):
        rng = rngs_mc[i]
        inc[i] = float(rng.uniform(0.0, np.pi))
        ecc[i] = float(rng.uniform(0.005, 0.05))
        el_I[i] = float(rng.uniform(-np.pi / 2.0, np.pi / 2.0))
        az_I[i] = float(rng.uniform(0.0, 2.0 * np.pi))
        yaw[i] = float(rng.uniform(0.0, 2.0 * np.pi))
        pitch[i] = float(rng.uniform(0.0, np.pi))
        roll[i] = float(rng.uniform(0.0, 2.0 * np.pi))

    # ---------- Setup timestamps ----------
    timestamps = np.arange(0.0, tend, tstep, dtype=float)
    if tstep < IMAGE_MAX_DT_S:
        IMAGE_STRIDE = max(1, int(math.floor(IMAGE_MAX_DT_S / tstep)))
    else:
        IMAGE_STRIDE = 1
    timestamps = timestamps[::IMAGE_STRIDE]
    tstep_eff = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else tstep
    nbSteps = len(timestamps)

    print(f"  Time span: 0 to {tend}s, effective dt: {tstep_eff}s, samples: {nbSteps}")

    # ---------- Derived constants ----------
    a = parameterSetting(h_orbit)
    TU = np.sqrt(a**3 / mu_ref) * 2 * np.pi

    # ---------- Preallocations ----------
    # TODO going to want to change up naming conventions to be more readable
    r_AG_G_used = np.array(DEFAULT_r_AG_G, dtype=float)

    Rx = np.zeros((num_mc, 3, 3))
    Ry = np.zeros((num_mc, 3, 3))
    Rz_arr = np.zeros((num_mc, 3, 3))
    R_XG_0 = np.zeros((num_mc, 3, 3))
    oe_t = np.zeros((num_mc, 6))
    r_AO_I_0 = np.zeros((num_mc, 3))
    v_AO_I_0 = np.zeros((num_mc, 3))
    R_IH_0 = np.zeros((num_mc, 3, 3))
    n = np.zeros(num_mc)
    omega_HI_I_0 = np.zeros((num_mc, 3))

    # Target spacecraft states
    # G frame is the body fixed frame
    # A is the body COM frame (aligned with G axes, offset)
    state_A_I = np.zeros((num_mc, nbSteps, 6))
    q_IG = np.zeros((num_mc, nbSteps, 4))
    R_IG = np.zeros((num_mc, nbSteps, 3, 3))
    omega_GI_I = np.zeros((num_mc, nbSteps, 3))
    omega_GI_G = np.zeros((num_mc, nbSteps, 3))

    r_GO_I = np.zeros((num_mc, nbSteps, 3))
    v_GO_I = np.zeros((num_mc, nbSteps, 3))

    # Deffenitions for the Camera / Chaser (Sensor S) frame
    state_C_I = np.zeros((num_mc, num_agents, nbSteps, 6))
    state_C_I_0 = np.zeros((num_mc, num_agents, 6))

    # Combined IC and propagated states
    R_GC = np.zeros((num_mc, num_agents, nbSteps, 3, 3))
    q_GC = np.zeros((num_mc, num_agents, nbSteps, 4))
    R_IC = np.zeros((num_mc, num_agents, nbSteps, 3, 3))
    q_IC = np.zeros((num_mc, num_agents, nbSteps, 4))
    R_IC_m = np.zeros((num_mc, num_agents, nbSteps, 3, 3))
    q_IC_m = np.zeros((num_mc, num_agents, nbSteps, 4))
    omega_CI_C = np.zeros((num_mc, num_agents, nbSteps, 3))
    omega_CI_C_m = np.zeros((num_mc, num_agents, nbSteps, 3))

    r_CG_G = np.zeros((num_mc, num_agents, nbSteps, 3))
    v_CG_G = np.zeros((num_mc, num_agents, nbSteps, 3))
    dr_CG_G = np.zeros((num_mc, num_agents, nbSteps, 3))

    f_specific_S = np.zeros((num_mc, num_agents, nbSteps, 3))
    f_specific_S_m = np.zeros((num_mc, num_agents, nbSteps, 3))
    tau_specific_S = np.zeros((num_mc, num_agents, nbSteps, 3))

    r_OG_G = np.zeros((num_mc, nbSteps, 3))
    sun_dir_G = np.zeros((num_mc, nbSteps, 3))
    el_G = np.zeros((num_mc, nbSteps))
    az_G = np.zeros((num_mc, nbSteps))

    H_GI_G = np.zeros((num_mc, nbSteps, 3))
    H_GI_I = np.zeros((num_mc, nbSteps, 3))


    # TODO these could be considered constants if we want them to be defined somewhere else
    # Disturbances
    eta = np.random.multivariate_normal(MEAN_DEFAULT, COV_R_ASTRO_APS3, (num_agents, nbSteps))
    nu = np.random.multivariate_normal(MEAN_DEFAULT, COV_OMEGA_ASTRIX, (num_agents, nbSteps))

    # IMU bias
    g0 = 9.80665
    sigma_bg = np.deg2rad(GYRO_BIAS_SIGMA_DEGPHR) / 3600.0
    sigma_ba = ACCEL_BIAS_SIGMA_UG * 1e-6 * g0
    phi_g = np.exp(-tstep_eff / GYRO_BIAS_TAU_S) if GYRO_BIAS_TAU_S > 0 else 1.0
    phi_a = np.exp(-tstep_eff / ACCEL_BIAS_TAU_S) if ACCEL_BIAS_TAU_S > 0 else 1.0
    inc_std_g = sigma_bg * np.sqrt(max(0.0, 1.0 - phi_g**2))
    inc_std_a = sigma_ba * np.sqrt(max(0.0, 1.0 - phi_a**2))
    gyro_bias_state = np.zeros((num_mc, num_agents, 3))
    accel_bias_state = np.zeros((num_mc, num_agents, 3))

    # ---------- Propagate orbits and compute geometry ----------
    print("\n[STEP 3] Propagating orbits and computing geometry...")

    # Outer loop: MC trials
    for i in range(num_mc):
        print(f"  MC trial {i+1}/{num_mc}...", end=" ", flush=True)

        # Target initial attitude
        Rz_arr[i] = au2R(yaw[i], np.array([0, 0, 1]))
        Ry[i] = au2R(pitch[i], np.array([0, 1, 0]))
        Rx[i] = au2R(roll[i], np.array([1, 0, 0]))
        R_XG_0[i] = Rx[i] @ Ry[i] @ Rz_arr[i]

        # Target COM initial state
        oe_t[i] = np.array([a, ecc[i], inc[i], 0, 0, 0])
        r_AO_I_0[i], v_AO_I_0[i] = oe2cart(oe_t[i], mu_ref)
        R_IH_0[i] = createHillFrame(r_AO_I_0[i], v_AO_I_0[i])
        n[i] = np.sqrt(mu_ref / np.linalg.norm(r_AO_I_0[i])**3)
        omega_HI_I_0[i] = n[i] * R_IH_0[i, :, 2]
        state_A_I[i, 0] = np.hstack([r_AO_I_0[i], v_AO_I_0[i]])

        # Propagate target COM
        state_A_I[i] = propagate_orbit(mu_ref, state_A_I[i, 0], timestamps)

        # Per-agent chaser propagation
        for agent_idx in range(num_agents):
            r0_agent = r_0[i, agent_idx].copy()
            if rotMode_Gframe == "2":
                r0_agent[0] = -r0_agent[4] / (2 * n[i])

            r_CO_I_0, v_CO_I_0 = calcInitCondChaser(
                r0_agent[3], r0_agent[4], r0_agent[5],
                r0_agent[0], r0_agent[1], r0_agent[2],
                state_A_I[i, 0], R_IH_0[i], omega_HI_I_0[i],
            )
            state_C_I_0[i, agent_idx] = np.hstack([r_CO_I_0, v_CO_I_0])
            state_C_I[i, agent_idx] = propagate_orbit(mu_ref, state_C_I_0[i, agent_idx], timestamps)

        # Attitude law
        if rotMode_Gframe == "1":
            for j in range(nbSteps):
                R_IG[i, j] = R_XG_0[i]
                omega_GI_I[i, j] = np.array([0, 0, 0])
                omega_GI_G[i, j] = np.array([0, 0, 0])
        elif rotMode_Gframe == "2":
            for j in range(nbSteps):
                R_IH = createHillFrame(state_A_I[i, j, 0:3], state_A_I[i, j, 3:6])
                R_IG[i, j] = R_IH @ R_XG_0[i]
                omega_GI_I[i, j] = omega_HI_I_0[i]
                omega_GI_G[i, j] = R_IG[i, j].T @ omega_GI_I[i, j]
        elif rotMode_Gframe == "3":
            R0 = R_XG_0[i]
            # omega_GI_G_0 is (num_mc, 3) array - extract this trial's initial angular velocity
            # Use retry loop to ensure sufficient omega excitation for inertia observability
            MAX_OMEGA_RETRIES = 10
            omega_excitation_validated = False

            for omega_retry in range(MAX_OMEGA_RETRIES):
                omega_GI_G[i], R_IG[i] = solve_ne_equation(nbSteps, tstep_eff, omega_GI_G_0[i], R0, J)

                # Validate omega timeseries has sufficient excitation
                dt_array = np.full(nbSteps - 1, tstep_eff)
                is_valid, validation_stats = validate_omega_timeseries_excitation(
                    omega_GI_G[i], dt_array=dt_array,
                    omega_dot_min=1e-5, off_axis_min=0.3
                )

                if is_valid:
                    omega_excitation_validated = True
                    break
                else:
                    # Resample omega_0 with inertia-aware sampling
                    if omega_retry < MAX_OMEGA_RETRIES - 1:
                        omega_mag = np.linalg.norm(omega_GI_G_0[i])
                        d_new, _ = sample_inertia_excited_omega_direction(
                            rngs_mc[i], J,
                            min_asymmetry_component=0.4,
                            off_axis_min=0.3
                        )
                        omega_GI_G_0[i] = omega_mag * d_new

            if not omega_excitation_validated:
                print(f"\n  WARNING: MC trial {i} omega excitation validation failed after {MAX_OMEGA_RETRIES} retries")
                print(f"           max_omega_dot={validation_stats.get('max_omega_dot', 'N/A')}")

            for j in range(nbSteps):
                omega_GI_I[i, j] = R_IG[i, j] @ omega_GI_G[i, j]

        # Sun alignment
        if EARTH_BACKGROUND_ENABLE:
            # Earth-in-background alignment: Sun -> Camera -> Target -> Earth
            # Sun direction should be opposite to Earth direction from camera
            # i.e., sun is "behind" the camera, so camera looks toward target with Earth behind it
            j0 = 0
            agent0 = 0
            # Camera position in inertial frame (Earth at origin)
            r_cam_I = state_C_I[i, agent0, j0, 0:3]
            # Sun direction = away from Earth (same direction as camera position from Earth)
            if np.linalg.norm(r_cam_I) > 0:
                u_sun_I = r_cam_I / np.linalg.norm(r_cam_I)
                # Add small jitter if desired
                if SUN_ALIGN_JITTER_D > 0:
                    jitter_rad = np.deg2rad((np.random.rand() - 0.5) * 2.0 * SUN_ALIGN_JITTER_D)
                    up = np.array([0.0, 0.0, 1.0])
                    if abs(u_sun_I @ up) > 0.95:
                        up = np.array([0.0, 1.0, 0.0])
                    u_sun_I = rodrigues(u_sun_I, up, jitter_rad)
                az_I[i], el_I[i] = _vecI_to_azel(u_sun_I)
        elif SUN_ALIGN_ENABLE:
            # Original: align sun with camera line-of-sight (front-lit target)
            j0 = 0
            agent0 = 0
            # r_CG^I = r_CA^I - r_AG^I = r_CA^I - R_IG @ r_AG^G
            r_CG_I0 = (state_C_I[i, agent0, j0, 0:3] - state_A_I[i, j0, 0:3]) - R_IG[i, j0] @ r_AG_G_used
            r_CG_G0 = R_IG[i, j0].T @ r_CG_I0
            if np.linalg.norm(r_CG_G0) > 0:
                u_LOS_G = -r_CG_G0 / np.linalg.norm(r_CG_G0)
                cone_deg = SUN_ALIGN_CONE_DEG + (np.random.rand() - 0.5) * 2.0 * SUN_ALIGN_JITTER_D
                cone_rad = np.deg2rad(max(0.0, cone_deg))
                up = np.array([0.0, 0.0, 1.0])
                if abs(u_LOS_G @ up) > 0.95:
                    up = np.array([0.0, 1.0, 0.0])
                u_sun_G = rodrigues(u_LOS_G, up, cone_rad)
                u_sun_I = R_IG[i, j0] @ u_sun_G
                az_I[i], el_I[i] = _vecI_to_azel(u_sun_I)

        # Continuous look-at per agent
        x_right_prev = [None] * num_agents
        q_GC_prev = [None] * num_agents

        for j in range(nbSteps):
            q_IG[i, j] = R2q(R_IG[i, j])
            r_AO_I = state_A_I[i, j, 0:3]
            v_AO_I = state_A_I[i, j, 3:6]

            r_GO_I[i, j] = r_AO_I + R_IG[i, j] @ r_AG_G_used
            v_GO_I[i, j] = v_AO_I + R_IG[i, j] @ np.cross(omega_GI_G[i, j], r_AG_G_used)

            H_GI_G[i, j] = J @ omega_GI_G[i, j]
            H_GI_I[i, j] = R_IG[i, j] @ H_GI_G[i, j]

            r_OG_G[i, j] = -R_IG[i, j].T @ r_GO_I[i, j]
            sun_dir_G[i, j] = R_IG[i, j].T @ np.array([
                np.cos(el_I[i]) * np.cos(az_I[i]),
                np.cos(el_I[i]) * np.sin(az_I[i]),
                np.sin(el_I[i]),
            ])
            az_G[i, j] = math.degrees(math.atan2(sun_dir_G[i, j, 1], sun_dir_G[i, j, 0]))
            el_G[i, j] = math.degrees(math.atan2(
                sun_dir_G[i, j, 2],
                np.sqrt(sun_dir_G[i, j, 0]**2 + sun_dir_G[i, j, 1]**2)
            ))

            for agent_idx in range(num_agents):
                d_rI = state_C_I[i, agent_idx, j, 0:3] - r_AO_I
                d_vI = state_C_I[i, agent_idx, j, 3:6] - v_AO_I

                # r_CG = C - G = (C - A) - (G - A) = r_CA - r_AG
                # r_CG^G = R_IG.T @ r_CA^I - r_AG^G
                r_CG_G[i, agent_idx, j] = R_IG[i, j].T @ d_rI - r_AG_G_used
                v_CG_G[i, agent_idx, j] = R_IG[i, j].T @ d_vI - np.cross(omega_GI_G[i, j], r_AG_G_used)
                dr_CG_G[i, agent_idx, j] = v_CG_G[i, agent_idx, j] - np.cross(omega_GI_G[i, j], r_CG_G[i, agent_idx, j])

                fwd_G = -r_CG_G[i, agent_idx, j]
                Rgs, x_right_prev[agent_idx] = _lookat_continuous_RGS(
                    fwd_G=fwd_G,
                    world_up_G=np.array([0.0, 0.0, 1.0]),
                    x_prev=x_right_prev[agent_idx],
                    cos_thr=0.9995,
                    sin_thr=0.03
                )
                R_GC[i, agent_idx, j] = Rgs

                q_raw = R2q(R_GC[i, agent_idx, j])
                q_GC[i, agent_idx, j] = _quat_hemi_continuous(q_raw, q_GC_prev[agent_idx])
                q_GC_prev[agent_idx] = q_GC[i, agent_idx, j]

                R_IC[i, agent_idx, j] = R_IG[i, j] @ R_GC[i, agent_idx, j]
                q_IC[i, agent_idx, j] = R2q(R_IC[i, agent_idx, j])

                R_IC_m[i, agent_idx, j] = R_IC[i, agent_idx, j] @ expm(sk(eta[agent_idx, j]))
                q_IC_m[i, agent_idx, j] = R2q(R_IC_m[i, agent_idx, j])

        # Enforce quaternion continuity
        for agent_idx in range(num_agents):
            enforce_quat_series_continuity(q_GC[i, agent_idx, :])
            enforce_quat_series_continuity(q_IC[i, agent_idx, :])

        """
        Angular velocity computation - CONSISTENT with dynamics
        
        Previously: omega_CI_C was computed from finite differences of R_IC, but omega_GI_G
        came from the ODE. This caused ~0.00086 rad/s discretization error.
        
        Now: We compute omega_CI_C using the angular velocity addition formula:
          ω_IC^C = ω_IG^C + ω_GC^C
        where:
          ω_IG^C = R_CG @ ω_IG^G = -R_CG @ omega_GI_G  (from ODE, transformed to C frame)
          ω_GC^C = from finite differences of R_GC    (consistent with pose evolution)
        
        This ensures omega_SI_S is consistent with BOTH the dynamics (omega_GI_G) AND
        the pose evolution (finite differences of R_GS).
        
        NAMING NOTE: omega_CI_C variable stores ω_IC^C (I wrt C, in C frame).
        It is negated when outputting to gtValues.txt to get ω_SI^S = ω_CI^C = -ω_IC^C.
        """
        for agent_idx in range(num_agents):
            # First compute omega_GC_C from finite differences of R_GC
            # log(R_GC^T @ R_GC_next)/dt gives ω_GC^C (G wrt C, in C frame)
            omega_GC_C = np.zeros((nbSteps, 3))
            omega_GC_C[0] = so3_log_vec(R_GC[i, agent_idx, 0].T @ R_GC[i, agent_idx, 1]) / tstep_eff
            for k in range(1, nbSteps - 1):
                Rm = R_GC[i, agent_idx, k - 1].T @ R_GC[i, agent_idx, k + 1]
                omega_GC_C[k] = so3_log_vec(Rm) / (2.0 * tstep_eff)
            omega_GC_C[nbSteps - 1] = so3_log_vec(R_GC[i, agent_idx, nbSteps - 2].T @ R_GC[i, agent_idx, nbSteps - 1]) / tstep_eff

            # Now compute omega_IC^C using angular velocity addition:
            #   ω_IC = ω_IG + ω_GC  (I wrt C = I wrt G + G wrt C)
            # In C frame:
            #   ω_IC^C = ω_IG^C + ω_GC^C
            # where:
            #   ω_IG^C = -R_CG @ ω_GI^G (negate to convert G-wrt-I to I-wrt-G, then to C frame)
            #   ω_GC^C = -omega_GC_C (the variable stores ω_CG^C from finite differences)
            #
            # NOTE: Variable omega_CI_C stores ω_IC^C (I wrt C, in C frame), not ω_CI^C.
            # The finite difference log(R_GC^T @ R_GC') gives the body-frame angular velocity
            # for right-multiplication R_GC' = R_GC @ Exp(ω), which is ω_CG^C (C wrt G, in C).
            # The variable omega_GC_C is misnamed - it actually stores ω_CG^C = -ω_GC^C.
            # Angular velocity addition: ω_IC^C = ω_IG^C + ω_GC^C = ω_IG^C - ω_CG^C
            for k in range(nbSteps):
                R_CG = R_GC[i, agent_idx, k].T
                omega_IG_C = -R_CG @ omega_GI_G[i, k]  # ω_IG^C = -R_CG @ ω_GI^G
                omega_CI_C[i, agent_idx, k] = omega_IG_C - omega_GC_C[k]  # ω_IC^C = ω_IG^C - ω_CG^C (note: omega_GC_C stores ω_CG^C)

            for j in range(nbSteps):
                if j == 0:
                    gyro_bias_state[i, agent_idx] = np.random.normal(0.0, sigma_bg, size=3)
                else:
                    gyro_bias_state[i, agent_idx] = phi_g * gyro_bias_state[i, agent_idx] + inc_std_g * np.random.normal(0.0, 1.0, size=3)
                omega_CI_C_m[i, agent_idx, j] = omega_CI_C[i, agent_idx, j] + nu[agent_idx, j] + gyro_bias_state[i, agent_idx]

        # Specific force and torque
        for agent_idx in range(num_agents):
            for j in range(nbSteps):
                r_c_I = state_C_I[i, agent_idx, j, 0:3]
                r_magnitude = np.linalg.norm(r_c_I)
                gravity_I = -mu_ref * r_c_I / (r_magnitude**3)
                if j == 0:
                    non_grav_accel_I = (state_C_I[i, agent_idx, 1, 3:6] - state_C_I[i, agent_idx, 0, 3:6]) / tstep_eff - gravity_I
                elif j == nbSteps - 1:
                    non_grav_accel_I = (state_C_I[i, agent_idx, j, 3:6] - state_C_I[i, agent_idx, j - 1, 3:6]) / tstep_eff - gravity_I
                else:
                    non_grav_accel_I = (state_C_I[i, agent_idx, j + 1, 3:6] - state_C_I[i, agent_idx, j - 1, 3:6]) / (2 * tstep_eff) - gravity_I
                f_specific_S[i, agent_idx, j] = R_IC[i, agent_idx, j].T @ non_grav_accel_I

                if j == 0:
                    accel_bias_state[i, agent_idx] = np.random.normal(0.0, sigma_ba, size=3)
                else:
                    accel_bias_state[i, agent_idx] = phi_a * accel_bias_state[i, agent_idx] + inc_std_a * np.random.normal(0.0, 1.0, size=3)
                f_specific_S_m[i, agent_idx, j] = f_specific_S[i, agent_idx, j] + np.random.multivariate_normal(np.zeros(3), COV_ACCEL_ASTRIX) + accel_bias_state[i, agent_idx]

                if j == 0:
                    omega_dot = (omega_CI_C[i, agent_idx, 1] - omega_CI_C[i, agent_idx, 0]) / tstep_eff
                elif j == nbSteps - 1:
                    omega_dot = (omega_CI_C[i, agent_idx, j] - omega_CI_C[i, agent_idx, j - 1]) / tstep_eff
                else:
                    omega_dot = (omega_CI_C[i, agent_idx, j + 1] - omega_CI_C[i, agent_idx, j - 1]) / (2 * tstep_eff)
                J_omega = J @ omega_CI_C[i, agent_idx, j]
                tau_specific_S[i, agent_idx, j] = J @ omega_dot + np.cross(omega_CI_C[i, agent_idx, j], J_omega)

        print("done")

    # ---------- Write output files ----------

    def write_camera_trajectory():
        """
        Write camera_traj.txt for Blender import.
        TODO Lucas Will need to totally re format this
        TODO we may need to fix up the referencing frames here too
        a data structure to hold everything may be a good idea
        """
        blender_filepath = os.path.join(ue5_out, "camera_traj.txt")
        with open(blender_filepath, "w") as f:
            f.write("nbTruePts = \n")
            f.write(f"{nbSteps}\n")
            f.write("tspan = \n")
            np.savetxt(f, timestamps, fmt="%f")
            f.write("q_GC = \n")
            np.savetxt(f, q_GC_use, fmt="%f %f %f %f")
            f.write("r_CG = \n")
            np.savetxt(f, r_CG_G_use, fmt="%f %f %f")
            f.write("r_OG_G = \n")
            np.savetxt(f, r_OG_G[i], fmt="%f %f %f")
            f.write("sun_az = \n")
            np.savetxt(f, az_G[i], fmt="%f")
            f.write("sun_el = \n")
            np.savetxt(f, el_G[i], fmt="%f")
            f.write("q_IG = \n")
            np.savetxt(f, q_IG[i], fmt="%f %f %f %f")
        print(f"  [BLENDER] {blender_filepath}")

    print("\n[STEP 4] Writing output files...")

    now = datetime.today()
    date_prefix = now.strftime("%m%d")  # e.g., "1207"
    time_suffix = now.strftime("%Y_%m_%H%M")  # e.g., "2025_12_1430"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    camera_obj = {"focal_length": FOCAL_LENGTH_PX, "resolution": CAMERA_RESOLUTION, "lens_flare": LENS_FLARE}

    for i in range(num_mc):
        for agent_idx in range(num_agents):
            dir_name = f"{date_prefix}_{path_mode}_mc{i}_cro_agent{agent_idx}_{time_suffix}"
            ue5_out = os.path.join(OUTPUT_BASE, dir_name)
            os.makedirs(ue5_out, exist_ok=True)

            r_CG_G_use = r_CG_G[i, agent_idx]
            v_CG_G_use = v_CG_G[i, agent_idx]
            q_GC_use = q_GC[i, agent_idx]
            q_IC_use = q_IC[i, agent_idx]
            omega_CI_C_use = omega_CI_C[i, agent_idx]
            s_c_I_use = state_C_I[i, agent_idx]

            # Compute and print range statistics
            ranges = np.linalg.norm(r_CG_G_use, axis=1)
            r_min, r_max = float(np.min(ranges)), float(np.max(ranges))
            print(f"  [INFO] MC{i} Agent{agent_idx}: range=[{r_min:.2f}, {r_max:.2f}]m, focal_length={camera_obj['focal_length']}px")

            write_camera_trajectory()

            # --- gtValues.txt ---
            gtvalues_filepath = os.path.join(ue5_out, "gtValues.txt")
            with open(gtvalues_filepath, "w") as f:
                f.write("nSamples = \n")
                f.write(f"{nbSteps}\n")
                f.write("timestamps = \n")
                np.savetxt(f, timestamps, fmt="%f")

                # Inertia tensor at COM (frame A), trace-normalized, 5-parameter format
                # Format: Ixx Iyy Ixy Ixz Iyz (Izz = 1 - Ixx - Iyy from trace constraint)
                # This is the target's inertia tensor used in Euler dynamics propagation.
                # The inertia is expressed in body-fixed frame A (COM frame).
                trace_J = np.trace(J)
                J_norm = J / trace_J
                inertia_5 = np.array([J_norm[0, 0], J_norm[1, 1],
                                      J_norm[0, 1], J_norm[0, 2], J_norm[1, 2]])
                f.write("inertia_A = \n")
                f.write(f"{inertia_5[0]:.9f} {inertia_5[1]:.9f} {inertia_5[2]:.9f} {inertia_5[3]:.9f} {inertia_5[4]:.9f}\n")

                # r_AG_G: COM position relative to G-frame origin, in G coords
                # SIGN CONVENTION: Per LaTeX, r_YX = Y - X (vector from X to Y)
                # r_AG_G = A - G (position of A relative to G, vector from G to A)
                # Internal variable r_AG_G_used stores G - A, so negate for output
                f.write("r_AG_G = \n")
                r_AG_G_output = -r_AG_G_used  # Convert from internal G-A to convention A-G
                f.write(f"{r_AG_G_output[0]:.9f} {r_AG_G_output[1]:.9f} {r_AG_G_output[2]:.9f}\n")

                f.write("q_GS = \n")
                np.savetxt(f, q_GC_use, fmt="%f %f %f %f")
                f.write("q_IG = \n")
                np.savetxt(f, q_IG[i], fmt="%f %f %f %f")
                f.write("q_IS = \n")
                np.savetxt(f, q_IC_use, fmt="%f %f %f %f")

                # SIGN CONVENTION (omega_GI_G):
                # solve_ne_equation uses standard kinematics: R_{k+1} = R_k @ exp(sk(ω·dt))
                # where ω is the body angular velocity ω_GI^G from Newton-Euler equations.
                # This corresponds to dR_IG/dt = R_IG @ sk(ω_GI^G), which is the standard
                # textbook convention. No sign flip needed - output directly.
                f.write("omega_GI_G = \n")
                np.savetxt(f, omega_GI_G[i], fmt="%f %f %f")
                # SIGN CONVENTION (omega_SI_S):
                # Factor expects ω_SI^S = angular velocity of S wrt I, expressed in S.
                #
                # omega_CI_C is computed using the angular velocity addition formula:
                #   ω_IC^C = ω_IG^C + ω_GC^C
                # where:
                #   ω_IG^C = -R_CG @ ω_GI^G (sign flip: I-wrt-G = -(G-wrt-I))
                #   ω_GC^C = -omega_GC_C (variable stores ω_CG^C from finite differences)
                #
                # The variable omega_CI_C stores ω_IC^C (I wrt C, in C).
                # Output -omega_CI_C = ω_CI^C = ω_SI^S (since C = S).
                f.write("omega_SI_S = \n")
                np.savetxt(f, -omega_CI_C_use, fmt="%f %f %f")

                # SIGN CONVENTION: Per LaTeX, r_YX = Y - X (vector from X to Y)
                # r_SG = S - G, and since C = S (chaser = spacecraft): r_SG = r_CG = C - G
                # r_CG_G is computed correctly as C - G, so output directly (no negation)
                f.write("r_SG_G = \n")
                np.savetxt(f, r_CG_G_use, fmt="%f %f %f")
                f.write("v_SG_G = \n")
                np.savetxt(f, v_CG_G_use, fmt="%f %f %f")

                f.write("r_OG_G = \n")
                np.savetxt(f, r_OG_G[i], fmt="%f %f %f")
                f.write("sun_az_el = \n")
                for j in range(nbSteps):
                    f.write(f"{az_G[i, j]:.6f} {el_G[i, j]:.6f}\n")

                # SIGN CONVENTION: Per LaTeX, r_YX = Y - X (vector from X to Y)
                # r_SA = S - A (position of S relative to A, i.e., vector from A to S)
                r_SA_I_data = s_c_I_use[:, 0:3] - state_A_I[i, :, 0:3]  # S - A
                v_SA_I_data = s_c_I_use[:, 3:6] - state_A_I[i, :, 3:6]  # v_S - v_A
                f.write("r_SA_I = \n")
                np.savetxt(f, r_SA_I_data, fmt="%f %f %f")
                f.write("v_SA_I = \n")
                np.savetxt(f, v_SA_I_data, fmt="%f %f %f")

                f.write("r_AO_I = \n")
                np.savetxt(f, state_A_I[i, :, 0:3], fmt="%f %f %f")
                f.write("v_AO_I = \n")
                np.savetxt(f, state_A_I[i, :, 3:6], fmt="%f %f %f")

                f.write("r_GO_I = \n")
                np.savetxt(f, r_GO_I[i], fmt="%f %f %f")
                f.write("v_GO_I = \n")
                np.savetxt(f, v_GO_I[i], fmt="%f %f %f")
            print(f"  [GTVAL]   {gtvalues_filepath}")

            # --- JSON file for UE5 simulator (read from gtValues.txt) ---
            json_dir = os.path.join(OUTPUT_BASE, "json")
            os.makedirs(json_dir, exist_ok=True)
            json_filename = os.path.join(json_dir, f"{dir_name}.json")
            data_dict = read_gt_values(gtvalues_filepath)
            create_json(camera_obj, data_dict, tstep_eff, tend, json_filename, earth=False, stars=False)
            print(f"  [JSON]    {json_filename}")

            # --- sensormeasurements.txt ---
            sensor_filepath = os.path.join(ue5_out, "sensormeasurements.txt")
            with open(sensor_filepath, "w") as f:
                f.write("nSamples = \n")
                f.write(f"{nbSteps}\n")
                f.write("timestamps = \n")
                np.savetxt(f, timestamps, fmt="%f")
                f.write("q_IS_m = \n")
                np.savetxt(f, q_IC_m[i, agent_idx], fmt="%f %f %f %f")
                # Same sign convention as gtValues.txt: negate omega_CI_C to get ω_SI^S
                # (see detailed comment in gtValues.txt section above)
                f.write("omega_SI_S_m = \n")
                np.savetxt(f, -omega_CI_C_m[i, agent_idx], fmt="%f %f %f")
                f.write("r_AO_I = \n")
                np.savetxt(f, state_A_I[i, :, 0:3], fmt="%f %f %f")
                f.write("f_s_S_m = \n")
                np.savetxt(f, f_specific_S_m[i, agent_idx], fmt="%f %f %f")
                f.write("tau_s_S = \n")
                np.savetxt(f, tau_specific_S[i, agent_idx], fmt="%f %f %f")
            print(f"  [SENSOR]  {sensor_filepath}")

            # --- Config.yaml (OpenCV-style YAML for SatSLAM) ---
            config_filepath = os.path.join(ue5_out, "Config.yaml")
            with open(config_filepath, "w") as f:
                # Camera intrinsics from camera_obj
                focal_length = camera_obj.get("focal_length", FOCAL_LENGTH_PX)
                resolution = camera_obj.get("resolution", CAMERA_RESOLUTION)
                cx = resolution / 2.0
                cy = resolution / 2.0
                fps = 1.0 / tstep_eff

                f.write("%YAML:1.0\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Filenames to load\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("ImageList.Filename: imgList.txt\n")
                f.write("GroundTruth.Filename: gtValues.txt\n")
                f.write("SensorMeasurements.Filename: sensormeasurements.txt\n")
                f.write(f"ShapeModel.Filename: {SHAPE_MODEL_FILENAME}\n")
                f.write("ShapeModel.ScaleFactor: 1.0\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Experimentation Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("Settings.runExperiments: 0\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# General Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("Settings.IsAsync: 0\n")
                f.write("Settings.ActiveVisualizer: 0\n")
                f.write("Settings.ActiveRecord: 0\n")
                f.write("Settings.ActiveBackEnd: 1\n")
                f.write("Settings.ActiveLoopClosure: 1\n")
                f.write("Settings.ActiveMesher: 0\n")
                f.write("Settings.ActiveLines: 0\n")
                f.write("Settings.EmptyQueueBeforeTerminate: 0\n")
                f.write("Settings.HasGT: 1\n")
                f.write("Settings.UseGT: 0\n")
                f.write("Settings.StartIdx: 0\n")
                f.write(f"Settings.EndIdx: {nbSteps}\n")
                f.write("Settings.InitialOffset: 0\n")
                f.write("Settings.nFramesInit: 6\n")
                f.write("Settings.DownsampleFactor: 1\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Camera Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("Camera.R_CS_r11: 1.0\n")
                f.write("Camera.R_CS_r12: 0.0\n")
                f.write("Camera.R_CS_r13: 0.0\n")
                f.write("Camera.R_CS_r21: 0.0\n")
                f.write("Camera.R_CS_r22: 1.0\n")
                f.write("Camera.R_CS_r23: 0.0\n")
                f.write("Camera.R_CS_r31: 0.0\n")
                f.write("Camera.R_CS_r32: 0.0\n")
                f.write("Camera.R_CS_r33: 1.0\n")
                f.write(f"Camera.fx: {focal_length}\n")
                f.write(f"Camera.fy: {focal_length}\n")
                f.write(f"Camera.cx: {cx}\n")
                f.write(f"Camera.cy: {cy}\n")
                f.write("Camera.k1: 0.0\n")
                f.write("Camera.k2: 0.0\n")
                f.write("Camera.p1: 0.0\n")
                f.write("Camera.p2: 0.0\n")
                f.write("Camera.k3: 0.0\n")
                f.write(f"Camera.fps: {fps:.1f}\n")
                f.write("Camera.RGB: 0\n")
                f.write(f"Camera.resolution: [{resolution}, {resolution}]\n")
                f.write("Image.FITSValueScale: 22.849\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Front-End Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("FrontEnd.kfInterval: 7\n")
                f.write("FrontEnd.matcherWindowAndCandidates: [1, 1]\n")
                f.write("FrontEnd.minInlierPercentage: 0.75\n")
                f.write("FrontEnd.minInliersToLastKF: 30\n")
                f.write("FrontEnd.minInliers: 8\n")
                f.write("FrontEnd.knnK: 1\n")
                f.write("FrontEnd.loweRatio: 0.75\n")
                f.write("FrontEnd.ransacReprojThreshold: 6.0\n")
                f.write("FrontEnd.ransacConfidence: 0.999\n")
                f.write("FrontEnd.minBaselinePx: 1e6\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# ORB Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("ORBextractor.nFeatures: 1450\n")
                f.write("ORBextractor.scaleFactor: 1.25\n")
                f.write("ORBextractor.nLevels: 7\n")
                f.write("ORBextractor.edgeThreshold: 10\n")
                f.write("ORBextractor.patchSize: 31\n")
                f.write("ORBextractor.fastThreshold: 14\n")
                f.write("ORBextractor.minFastThreshold: 4\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Line Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("Line.minlength: 1.0\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Back-End Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("BackEnd.useRelDyn: 1\n")
                f.write("Backend.minTriangulationAngleDeg: 0.5\n")
                f.write("BackEnd.iSAMRelinearizationThresh: 0.1\n")
                f.write("BackEnd.iSAMRelinearizationSkip: 1\n")
                f.write("BackEnd.iSAMcacheLinearizedFactors: 1\n")
                f.write("BackEnd.iSAMfindUnusedFactorSlots: 1\n")
                f.write("BackEnd.iSAMfactorization: QR\n")
                f.write("BackEnd.wildfire_threshold: 0.001\n")
                f.write("BackEnd.numOptimize: 1\n")
                f.write("BackEnd.optimizationWindowSec: 60\n")
                f.write("BackEnd.iSAMevaluateNonlinearError: 0\n")
                f.write("BackEnd.iSAMenableDetailedResults: 0\n")
                f.write("BackEnd.SfSMalpha: 1.0\n")
                f.write("Prior.Q_sigmas: [0.01, 0.01, 0.01]\n")
                f.write("Prior.r_sigmas: [0.1, 0.1, 1.0]\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# RelDyn Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("RelDynFactor.nStages: 5\n")
                f.write("RelDynFactor.ninc: 16\n")
                f.write("RelDynFactor.LU: 1.0\n")
                f.write("RelDynFactor.TU: 1.0\n")
                f.write(f"RelDynFactor.muEarth: {mu_ref}\n")
                f.write("RelDynFactor.usePreviousMeasurement: 1\n")
                f.write("RelDynFactor.sigmasSpectralDensity: [ 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.3, 0.3, 0.3]\n")
                f.write("RelDynFactor.omega_GI_G_noise: [0.01, 0.01, 0.01]\n")
                f.write("RelDynFactor.I_A_noise: [1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1]\n")
                f.write("RelDynFactor.r_AG_G_noise: [1.0, 1.0, 0.01]\n")
                f.write("RelDynFactor.r_SA_I_noise: [1.0, 1.0, 1.0]\n")
                f.write("RelDynFactor.v_SA_I_noise: [100.0, 100.0, 100.0]\n")
                f.write("RelDynFactor.rTc_noise: [1e-06, 1e-06, 1e-06]\n")
                f.write("RelDynFactor.angularMomentumNoise: [0.0001, 0.0001, 0.0001]\n")
                f.write("RelDynFactor.sigmas_r_SG_G_meas: [0.01, 0.01, 0.01]\n")
                f.write("RelDynFactor.mu0_noise: 20\n")
                f.write("RelDynFactor.kinoRoto_noise: [0.002, 0.002, 0.002, 0.0005, 0.0005, 0.0005]\n")
                f.write("RelDynFactor.Q0_sigmas: [1e-05, 1e-05, 1e-05]\n")
                f.write("RelDynFactor.r0_sigmas: [5e-08, 5e-08, 5e-08]\n")
                f.write("RelDynFactor.R_meas_noise: [0.0002774, 0.0002774, 0.0002774]\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Loop-Closure Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("LoopClosure.ignoredFramesWindow: 10\n")
                f.write("LoopClosure.defaultMinScore: 0.1\n")
                f.write("LoopClosure.loweRatio: 0.75\n")
                f.write("LoopClosure.minInliers: 6\n")
                f.write("LoopClosure.minSharedWords: 3\n")
                f.write("LoopClosure.commonWordsRatio: 0.8\n")
                f.write("LoopClosure.ransacReprojThreshold: 1.0\n")
                f.write("LoopClosure.ransacConfidence: 0.99\n")
                f.write("LoopClosure.knnK: 1\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Viewer Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("Viewer.KeyFrameSize: 0.05\n")
                f.write("Viewer.KeyFrameLineWidth: 1\n")
                f.write("Viewer.GraphLineWidth: 0.9\n")
                f.write("Viewer.PointSize: 2\n")
                f.write("Viewer.CameraSize: 0.08\n")
                f.write("Viewer.CameraLineWidth: 3\n")
                f.write("Viewer.ViewpointX: -100.0\n")
                f.write("Viewer.ViewpointY: 100.0\n")
                f.write("Viewer.ViewpointZ: 100.0\n")
                f.write("Viewer.ViewpointF: 500\n")
                f.write("Viewer.TriadScale: 1\n")
                f.write("Viewer.Width: 1920\n")
                f.write("Viewer.Height: 1080\n")
                f.write("Viewer.clipNear: 0.1\n")
                f.write("Viewer.clipFar: 1000\n")
                f.write("Viewer.fov: 45\n")
                f.write("Viewer.DrawObject: 1\n")
                f.write("Viewer.LightPower: 20.0\n\n")

                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Depth Simulator Parameters\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("Simulator.clipNear: 0.1\n")
                f.write("Simulator.clipFar: 4000\n")
                f.write("Simulator.DrawObject: 1\n")
                f.write("Simulator.LightPower: 20.0\n\n")

                # Add trajectory generation metadata as comments
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write("# Trajectory Generation Metadata (for reference only)\n")
                f.write("#--------------------------------------------------------------------------------------------\n")
                f.write(f"# trial_seed: {int(child_ss[i].entropy)}\n")
                f.write(f"# path_mode: {path_mode}\n")
                f.write(f"# rotMode_Gframe: {rotMode_Gframe}\n")
                f.write(f"# agent_id: {agent_idx}\n")
                f.write(f"# mu_ref: {mu_ref}\n")
                f.write(f"# h_orbit: {h_orbit}\n")
                f.write(f"# tend: {tend}\n")
                f.write(f"# tstep: {tstep_eff}\n")
                f.write(f"# inc: {inc[i]}\n")
                f.write(f"# ecc: {ecc[i]}\n")

            print(f"  [CONFIG]  {config_filepath}")

    # ---------- Generate plots ----------
    print("\n[STEP 5] Generating trajectory plots...")

    plot_dir = os.path.join(OUTPUT_BASE, "trial_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(num_mc):
        try:
            plot_trial_trajectories(
                i=i,
                s_A_I=state_A_I,
                s_c_I=state_C_I,
                r_CG_G=r_CG_G,
                R_IG_all=R_IG,
                out_dir=plot_dir,
                rotMode_Gframe=rotMode_Gframe,
                show=False,
                save=True,
            )
        except Exception as e:
            print(f"  [WARN] Could not generate plot for MC {i}: {e}")

    print(f"\n[DONE] Output written to: {OUTPUT_BASE}")
    print(f"       Master seed: {MASTER_SEED}")
    print(f"       Mode: {path_mode}")
    print(f"       {num_mc} MC trials, {num_agents} agent(s) each")


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    print("=" * 60)
    print("UNIFIED TRAJECTORY GENERATOR")
    print("=" * 60)

    # ---------- User inputs ----------
    # TODO this will need to be changed to only run if in a standalone mode
    num_agents = int(input("Number of agents for inspection scenario: ").strip())
    rotMode_Gframe = input("Mode Setting (1: Inertial, 2: Hill, 3: Tumbling): ").strip()
    num_mc = int(input("Number of samples for Monte Carlo Simulation: ").strip())

    # Setup RNGs
    child_ss = ss_master.spawn(num_mc)
    rngs_mc = [np.random.default_rng(cs) for cs in child_ss]

    # Path mode
    # TODO it would be good to unify path mode and rotmode as they deffine the same thing we want a single source of truth and rotmode is more readable
    if rotMode_Gframe == "1":
        path_mode = "Inertial"
    elif rotMode_Gframe == "2":
        path_mode = "Hill"
    elif rotMode_Gframe == "3":
        path_mode = "Tumbling"
    else:
        raise ValueError("Mode must be 1, 2, or 3.")

    print(f"\n[INFO] Master seed: {MASTER_SEED}")
    print(f"[INFO] Mode: {path_mode}")
    print(f"[INFO] Agents: {num_agents}, MC trials: {num_mc}")

    generate_trajectories(path_mode, rotMode_Gframe, num_agents, num_mc, rngs_mc)

if __name__ == "__main__":
    main()
