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

# Add project root to path so imports work both when running directly and when imported
# TODO clean this up
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.config import (TrajectoryConfig)
from motion_cases import (
    init_inertial, init_hill, init_tumbling,
    validate_omega_timeseries_excitation, sample_inertia_excited_omega_direction
)

# TODO Evaluate what we need in the math function
from trajectory_math import (
    au2R, oe2cart, createHillFrame, propagate_orbit, parameterSetting,
    sk, R2q, q2R, solve_ne_equation, so3_log_vec, rodrigues, _vecI_to_azel,
    _seed_right, _lookat_continuous_RGS, _quat_hemi_continuous, enforce_quat_series_continuity
)
from trajectory_math import calcInitCondChaser
from plot_figure import plot_trial_trajectories
from trajectory_io import write_camera_trajectory, write_gtvalues, write_json, write_config, write_sensormeasurements

# ---------------- Paths ----------------
# TODO a lot of this should go in the main method as it should only be run if running from command line
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SISIFOS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
DEFAULT_OUTPUT_BASE = os.path.join(SISIFOS_ROOT, "output")


# ============================================================================
# MAIN Function
# ============================================================================
def generate_trajectories(config: TrajectoryConfig, base_output_file: str):
    # Initialize random seeds for reproducibility
    if config.seed is not None:
        master_seed = config.seed
        print(f"[INITIALIZATION]: detected pre defined seed: {master_seed}")
    else:        
        master_seed = int(time.time() * 1e6) & 0x7FFFFFFF
        config.seed = master_seed  # Store the resolved seed back in the config for output
        print(f"[INITIALIZATION]: no pre defined seed detected, generating new seed: {master_seed}")

    ss_master = np.random.SeedSequence(master_seed)
    child_ss = ss_master.spawn(config.num_mc)
    rngs_mc = [np.random.default_rng(cs) for cs in child_ss]

    # ---------- Generate initial conditions ----------
    print("\n[STEP 1] Generating initial conditions...")

    if config.rotMode_Gframe == "1":
        print("  Inertial mode (CRO trajectory)")
        x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, _ = init_inertial(
            num_mc=config.num_mc, num_agents=config.num_agents, n_scalar=config.n_scalar,
            focal_length_px=config.FOCAL_LENGTH_PX, kf_dt=config.IMAGE_MAX_DT_S,
            px_min=config.MIN_F2F_PX_MED, rho_max=0.90, R0_const=config.R0_const,
            variant="cro", rngs_mc=rngs_mc
        )
    elif config.rotMode_Gframe == "2":
        print("  Hill mode")
        x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, _ = init_hill(
            num_mc=config.num_mc, num_agents=config.num_agents, n_scalar=config.n_scalar,
            rngs_mc=rngs_mc, focal_length_px=config.FOCAL_LENGTH_PX, kf_dt=config.IMAGE_MAX_DT_S,
            px_min=config.MIN_F2F_PX_MED, rho_max=0.90
        )
    elif config.rotMode_Gframe == "3":
        print("  Tumbling mode (CRO trajectory + target tumbling)")
        # Use faster tumbling (3-5 deg/s) for better inertia observability.
        # This is within the conservative design envelope (~5 deg/s upper bound).
        # Slower rates (0.5-2 deg/s default) have near-zero omega_dot, making
        # inertia poorly observable from Euler's equation I·ω̇ + ω×(I·ω) = 0.
        x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, _ = init_tumbling(
            num_mc=config.num_mc, num_agents=config.num_agents, n_scalar=config.n_scalar,
            rngs_mc=rngs_mc, focal_length_px=config.FOCAL_LENGTH_PX, kf_dt=config.IMAGE_MAX_DT_S,
            px_min=1.0, rho_max=0.95, R0_const=config.R0_const,
            omega_min_deg=3.0, omega_max_deg=5.0,
            J=config.J, min_asymmetry_component=0.4
        )

    # Ensure omega_GI_G_0 is a numpy array (tumbling returns (num_mc,3), others return [0,0,0])
    omega_GI_G_0 = np.atleast_2d(omega_GI_G_0)
    if omega_GI_G_0.shape[0] == 1 and config.num_mc > 1:
        # Broadcast single value to all MC trials
        omega_GI_G_0 = np.tile(omega_GI_G_0, (config.num_mc, 1))

    # Combine into r_0 array
    r_0 = np.zeros((config.num_mc, config.num_agents, 6))
    for mc_trial in range(config.num_mc):
        for a in range(config.num_agents):
            r_0[mc_trial, a] = [x_0[mc_trial, a], y_0[mc_trial, a], z_0[mc_trial, a],
                         xdot_0[mc_trial, a], ydot_0[mc_trial, a], zdot_0[mc_trial, a]]

    # ---------- Generate MC parameters ----------
    print("\n[STEP 2] Sampling orbital and attitude parameters...")

    inc = np.zeros(config.num_mc)
    ecc = np.zeros(config.num_mc)
    el_I = np.zeros(config.num_mc)
    az_I = np.zeros(config.num_mc)
    yaw = np.zeros(config.num_mc)
    pitch = np.zeros(config.num_mc)
    roll = np.zeros(config.num_mc)

    # TODO parameterize distributions
    for mc_trial in range(config.num_mc):
        rng = rngs_mc[mc_trial]
        inc[mc_trial] = float(rng.uniform(0.0, np.pi))
        ecc[mc_trial] = float(rng.uniform(0.005, 0.05))
        el_I[mc_trial] = float(rng.uniform(-np.pi / 2.0, np.pi / 2.0))
        az_I[mc_trial] = float(rng.uniform(0.0, 2.0 * np.pi))
        yaw[mc_trial] = float(rng.uniform(0.0, 2.0 * np.pi))
        pitch[mc_trial] = float(rng.uniform(0.0, np.pi))
        roll[mc_trial] = float(rng.uniform(0.0, 2.0 * np.pi))

    # ---------- Setup timestamps ----------
    timestamps = np.arange(0.0, config.tend, config.tstep, dtype=float)
    if config.tstep < config.IMAGE_MAX_DT_S:
        IMAGE_STRIDE = max(1, int(math.floor(config.IMAGE_MAX_DT_S / config.tstep)))
    else:
        IMAGE_STRIDE = 1
    timestamps = timestamps[::IMAGE_STRIDE]
    tstep_eff = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else config.tstep
    nbSteps = len(timestamps)

    print(f"  Time span: 0 to {config.tend}s, effective dt: {tstep_eff}s, samples: {nbSteps}")

    # ---------- Derived constants ----------
    a = parameterSetting(config.h_orbit)
    TU = np.sqrt(a**3 / config.mu_ref) * 2 * np.pi

    # ---------- Preallocations ----------
    r_AG_G = np.array(config.r_AG_G, dtype=float)

    Rx = np.zeros((config.num_mc, 3, 3))
    Ry = np.zeros((config.num_mc, 3, 3))
    Rz_arr = np.zeros((config.num_mc, 3, 3))
    R_XG_0 = np.zeros((config.num_mc, 3, 3))
    oe_t = np.zeros((config.num_mc, 6))
    r_AO_I_0 = np.zeros((config.num_mc, 3))
    v_AO_I_0 = np.zeros((config.num_mc, 3))
    R_IH_0 = np.zeros((config.num_mc, 3, 3))
    n = np.zeros(config.num_mc)
    omega_HI_I_0 = np.zeros((config.num_mc, 3))
    # Target spacecraft states
    # G frame is the body fixed frame
    # A is the body COM frame (aligned with G axes, offset)
    state_A_I = np.zeros((config.num_mc, nbSteps, 6))
    q_IG = np.zeros((config.num_mc, nbSteps, 4))
    R_IG = np.zeros((config.num_mc, nbSteps, 3, 3))
    omega_GI_I = np.zeros((config.num_mc, nbSteps, 3))
    omega_GI_G = np.zeros((config.num_mc, nbSteps, 3))

    r_GO_I = np.zeros((config.num_mc, nbSteps, 3))
    v_GO_I = np.zeros((config.num_mc, nbSteps, 3))

    # Deffenitions for the Camera / Chaser (Sensor S) frame
    state_C_I = np.zeros((config.num_mc, config.num_agents, nbSteps, 6))
    state_C_I_0 = np.zeros((config.num_mc, config.num_agents, 6))

    # Combined IC and propagated states
    R_GC = np.zeros((config.num_mc, config.num_agents, nbSteps, 3, 3))
    q_GC = np.zeros((config.num_mc, config.num_agents, nbSteps, 4))
    R_IC = np.zeros((config.num_mc, config.num_agents, nbSteps, 3, 3))
    q_IC = np.zeros((config.num_mc, config.num_agents, nbSteps, 4))
    R_IC_m = np.zeros((config.num_mc, config.num_agents, nbSteps, 3, 3))
    q_IC_m = np.zeros((config.num_mc, config.num_agents, nbSteps, 4))
    omega_CI_C = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))
    omega_CI_C_m = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))

    r_CG_G = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))
    v_CG_G = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))
    dr_CG_G = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))
    f_specific_S = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))
    f_specific_S_m = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))
    tau_specific_S = np.zeros((config.num_mc, config.num_agents, nbSteps, 3))

    r_OG_G = np.zeros((config.num_mc, nbSteps, 3))
    sun_dir_G = np.zeros((config.num_mc, nbSteps, 3))
    el_G = np.zeros((config.num_mc, nbSteps))
    az_G = np.zeros((config.num_mc, nbSteps))

    H_GI_G = np.zeros((config.num_mc, nbSteps, 3))
    H_GI_I = np.zeros((config.num_mc, nbSteps, 3))


    # TODO these could be considered constants if we want them to be defined somewhere else
    # Disturbances
    eta = np.random.multivariate_normal(config.MEAN_DEFAULT, config.COV_R_ASTRO_APS3, (config.num_agents, nbSteps))
    nu = np.random.multivariate_normal(config.MEAN_DEFAULT, config.COV_OMEGA_ASTRIX, (config.num_agents, nbSteps))

    #TODO these as well
    # IMU bias
    g0 = 9.80665
    sigma_bg = np.deg2rad(config.GYRO_BIAS_SIGMA_DEGPHR) / 3600.0
    sigma_ba = config.ACCEL_BIAS_SIGMA_UG * 1e-6 * g0
    phi_g = np.exp(-tstep_eff / config.GYRO_BIAS_TAU_S) if config.GYRO_BIAS_TAU_S > 0 else 1.0
    phi_a = np.exp(-tstep_eff / config.ACCEL_BIAS_TAU_S) if config.ACCEL_BIAS_TAU_S > 0 else 1.0
    inc_std_g = sigma_bg * np.sqrt(max(0.0, 1.0 - phi_g**2))
    inc_std_a = sigma_ba * np.sqrt(max(0.0, 1.0 - phi_a**2))
    gyro_bias_state = np.zeros((config.num_mc, config.num_agents, 3))
    accel_bias_state = np.zeros((config.num_mc, config.num_agents, 3))

    # ---------- Propagate orbits and compute geometry ----------
    print("\n[STEP 3] Propagating orbits and computing geometry...")

    # Outer loop: MC trials
    for mc_trial in range(config.num_mc):
        print(f"  MC trial {mc_trial+1}/{config.num_mc}...", end=" ", flush=True)

        # Target initial attitude
        Rz_arr[mc_trial] = au2R(yaw[mc_trial], np.array([0, 0, 1]))
        Ry[mc_trial] = au2R(pitch[mc_trial], np.array([0, 1, 0]))
        Rx[mc_trial] = au2R(roll[mc_trial], np.array([1, 0, 0]))
        R_XG_0[mc_trial] = Rx[mc_trial] @ Ry[mc_trial] @ Rz_arr[mc_trial]

        # Target COM initial state
        oe_t[mc_trial] = np.array([a, ecc[mc_trial], inc[mc_trial], 0, 0, 0])
        r_AO_I_0[mc_trial], v_AO_I_0[mc_trial] = oe2cart(oe_t[mc_trial], config.mu_ref)
        R_IH_0[mc_trial] = createHillFrame(r_AO_I_0[mc_trial], v_AO_I_0[mc_trial])
        n[mc_trial] = np.sqrt(config.mu_ref / np.linalg.norm(r_AO_I_0[mc_trial])**3)
        omega_HI_I_0[mc_trial] = n[mc_trial] * R_IH_0[mc_trial, :, 2]
        state_A_I[mc_trial, 0] = np.hstack([r_AO_I_0[mc_trial], v_AO_I_0[mc_trial]])

        # Propagate target COM
        state_A_I[mc_trial] = propagate_orbit(config.mu_ref, state_A_I[mc_trial, 0], timestamps)

        # Per-agent chaser propagation
        for agent_idx in range(config.num_agents):
            r0_agent = r_0[mc_trial, agent_idx].copy()
            if config.rotMode_Gframe == "2":
                r0_agent[0] = -r0_agent[4] / (2 * n[mc_trial])

            r_CO_I_0, v_CO_I_0 = calcInitCondChaser(
                r0_agent[3], r0_agent[4], r0_agent[5],
                r0_agent[0], r0_agent[1], r0_agent[2],
                state_A_I[mc_trial, 0], R_IH_0[mc_trial], omega_HI_I_0[mc_trial],
            )
            state_C_I_0[mc_trial, agent_idx] = np.hstack([r_CO_I_0, v_CO_I_0])
            state_C_I[mc_trial, agent_idx] = propagate_orbit(config.mu_ref, state_C_I_0[mc_trial, agent_idx], timestamps)

        # Attitude law
        if config.rotMode_Gframe == "1":
            for j in range(nbSteps):
                R_IG[mc_trial, j] = R_XG_0[mc_trial]
                omega_GI_I[mc_trial, j] = np.array([0, 0, 0])
                omega_GI_G[mc_trial, j] = np.array([0, 0, 0])
        elif config.rotMode_Gframe == "2":
            for j in range(nbSteps):
                R_IH = createHillFrame(state_A_I[mc_trial, j, 0:3], state_A_I[mc_trial, j, 3:6])
                R_IG[mc_trial, j] = R_IH @ R_XG_0[mc_trial]
                omega_GI_I[mc_trial, j] = omega_HI_I_0[mc_trial]
                omega_GI_G[mc_trial, j] = R_IG[mc_trial, j].T @ omega_GI_I[mc_trial, j]
        elif config.rotMode_Gframe == "3":
            R0 = R_XG_0[mc_trial]
            # omega_GI_G_0 is (num_mc, 3) array - extract this trial's initial angular velocity
            # Use retry loop to ensure sufficient omega excitation for inertia observability
            MAX_OMEGA_RETRIES = 10
            omega_excitation_validated = False

            for omega_retry in range(MAX_OMEGA_RETRIES):
                omega_GI_G[mc_trial], R_IG[mc_trial] = solve_ne_equation(nbSteps, tstep_eff, omega_GI_G_0[mc_trial], R0, config.J)

                # Validate omega timeseries has sufficient excitation
                dt_array = np.full(nbSteps - 1, tstep_eff)
                is_valid, validation_stats = validate_omega_timeseries_excitation(
                    omega_GI_G[mc_trial], dt_array=dt_array,
                    omega_dot_min=1e-5, off_axis_min=0.3
                )

                if is_valid:
                    omega_excitation_validated = True
                    break
                else:
                    # Resample omega_0 with inertia-aware sampling
                    if omega_retry < MAX_OMEGA_RETRIES - 1:
                        omega_mag = np.linalg.norm(omega_GI_G_0[mc_trial])
                        d_new, _ = sample_inertia_excited_omega_direction(
                            rngs_mc[mc_trial], config.J,
                            min_asymmetry_component=0.4,
                            off_axis_min=0.3
                        )
                        omega_GI_G_0[mc_trial] = omega_mag * d_new

            if not omega_excitation_validated:
                print(f"\n  WARNING: MC trial {mc_trial} omega excitation validation failed after {MAX_OMEGA_RETRIES} retries")
                print(f"           max_omega_dot={validation_stats.get('max_omega_dot', 'N/A')}")

            for j in range(nbSteps):
                omega_GI_I[mc_trial, j] = R_IG[mc_trial, j] @ omega_GI_G[mc_trial, j]

        # Sun alignment
        if config.EARTH_BACKGROUND_ENABLE:
            # Earth-in-background alignment: Sun -> Camera -> Target -> Earth
            # Sun direction should be opposite to Earth direction from camera
            # i.e., sun is "behind" the camera, so camera looks toward target with Earth behind it
            j0 = 0
            agent0 = 0
            # Camera position in inertial frame (Earth at origin)
            r_cam_I = state_C_I[mc_trial, agent0, j0, 0:3]
            # Sun direction = away from Earth (same direction as camera position from Earth)
            if np.linalg.norm(r_cam_I) > 0:
                u_sun_I = r_cam_I / np.linalg.norm(r_cam_I)
                # Add small jitter if desired
                if config.SUN_ALIGN_JITTER_D > 0:
                    jitter_rad = np.deg2rad((np.random.rand() - 0.5) * 2.0 * config.SUN_ALIGN_JITTER_D)
                    up = np.array([0.0, 0.0, 1.0])
                    if abs(u_sun_I @ up) > 0.95:
                        up = np.array([0.0, 1.0, 0.0])
                    u_sun_I = rodrigues(u_sun_I, up, jitter_rad)
                az_I[mc_trial], el_I[mc_trial] = _vecI_to_azel(u_sun_I)
        elif config.SUN_ALIGN_ENABLE:
            # Original: align sun with camera line-of-sight (front-lit target)
            j0 = 0
            agent0 = 0
            # r_CG^I = r_CA^I - r_AG^I = r_CA^I - R_IG @ r_AG^G
            r_CG_I0 = (state_C_I[mc_trial, agent0, j0, 0:3] - state_A_I[mc_trial, j0, 0:3]) - R_IG[mc_trial, j0] @ r_AG_G
            r_CG_G0 = R_IG[mc_trial, j0].T @ r_CG_I0
            if np.linalg.norm(r_CG_G0) > 0:
                u_LOS_G = -r_CG_G0 / np.linalg.norm(r_CG_G0)
                cone_deg = config.SUN_ALIGN_CONE_DEG + (np.random.rand() - 0.5) * 2.0 * config.SUN_ALIGN_JITTER_D
                cone_rad = np.deg2rad(max(0.0, cone_deg))
                up = np.array([0.0, 0.0, 1.0])
                if abs(u_LOS_G @ up) > 0.95:
                    up = np.array([0.0, 1.0, 0.0])
                u_sun_G = rodrigues(u_LOS_G, up, cone_rad)
                u_sun_I = R_IG[mc_trial, j0] @ u_sun_G
                az_I[mc_trial], el_I[mc_trial] = _vecI_to_azel(u_sun_I)

        # Continuous look-at per agent
        x_right_prev = [None] * config.num_agents
        q_GC_prev = [None] * config.num_agents

        for j in range(nbSteps):
            q_IG[mc_trial, j] = R2q(R_IG[mc_trial, j])
            r_AO_I = state_A_I[mc_trial, j, 0:3]
            v_AO_I = state_A_I[mc_trial, j, 3:6]

            r_GO_I[mc_trial, j] = r_AO_I + R_IG[mc_trial, j] @ r_AG_G
            v_GO_I[mc_trial, j] = v_AO_I + R_IG[mc_trial, j] @ np.cross(omega_GI_G[mc_trial, j], r_AG_G)

            H_GI_G[mc_trial, j] = config.J @ omega_GI_G[mc_trial, j]
            H_GI_I[mc_trial, j] = R_IG[mc_trial, j] @ H_GI_G[mc_trial, j]

            r_OG_G[mc_trial, j] = -R_IG[mc_trial, j].T @ r_GO_I[mc_trial, j]
            sun_dir_G[mc_trial, j] = R_IG[mc_trial, j].T @ np.array([
                np.cos(el_I[mc_trial]) * np.cos(az_I[mc_trial]),
                np.cos(el_I[mc_trial]) * np.sin(az_I[mc_trial]),
                np.sin(el_I[mc_trial]),
            ])
            az_G[mc_trial, j] = math.degrees(math.atan2(sun_dir_G[mc_trial, j, 1], sun_dir_G[mc_trial, j, 0]))
            el_G[mc_trial, j] = math.degrees(math.atan2(
                sun_dir_G[mc_trial, j, 2],
                np.sqrt(sun_dir_G[mc_trial, j, 0]**2 + sun_dir_G[mc_trial, j, 1]**2)
            ))

            for agent_idx in range(config.num_agents):
                d_rI = state_C_I[mc_trial, agent_idx, j, 0:3] - r_AO_I
                d_vI = state_C_I[mc_trial, agent_idx, j, 3:6] - v_AO_I

                # r_CG = C - G = (C - A) - (G - A) = r_CA - r_AG
                # r_CG^G = R_IG.T @ r_CA^I - r_AG^G
                r_CG_G[mc_trial, agent_idx, j] = R_IG[mc_trial, j].T @ d_rI - r_AG_G
                v_CG_G[mc_trial, agent_idx, j] = R_IG[mc_trial, j].T @ d_vI - np.cross(omega_GI_G[mc_trial, j], r_AG_G)
                dr_CG_G[mc_trial, agent_idx, j] = v_CG_G[mc_trial, agent_idx, j] - np.cross(omega_GI_G[mc_trial, j], r_CG_G[mc_trial, agent_idx, j])

                fwd_G = -r_CG_G[mc_trial, agent_idx, j]
                Rgs, x_right_prev[agent_idx] = _lookat_continuous_RGS(
                    fwd_G=fwd_G,
                    world_up_G=np.array([0.0, 0.0, 1.0]),
                    x_prev=x_right_prev[agent_idx],
                    cos_thr=0.9995,
                    sin_thr=0.03
                )
                R_GC[mc_trial, agent_idx, j] = Rgs

                q_raw = R2q(R_GC[mc_trial, agent_idx, j])
                q_GC[mc_trial, agent_idx, j] = _quat_hemi_continuous(q_raw, q_GC_prev[agent_idx])
                q_GC_prev[agent_idx] = q_GC[mc_trial, agent_idx, j]

                R_IC[mc_trial, agent_idx, j] = R_IG[mc_trial, j] @ R_GC[mc_trial, agent_idx, j]
                q_IC[mc_trial, agent_idx, j] = R2q(R_IC[mc_trial, agent_idx, j])

                R_IC_m[mc_trial, agent_idx, j] = R_IC[mc_trial, agent_idx, j] @ expm(sk(eta[agent_idx, j]))
                q_IC_m[mc_trial, agent_idx, j] = R2q(R_IC_m[mc_trial, agent_idx, j])

        # Enforce quaternion continuity
        for agent_idx in range(config.num_agents):
            enforce_quat_series_continuity(q_GC[mc_trial, agent_idx, :])
            enforce_quat_series_continuity(q_IC[mc_trial, agent_idx, :])

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
        for agent_idx in range(config.num_agents):
            # First compute omega_GC_C from finite differences of R_GC
            # log(R_GC^T @ R_GC_next)/dt gives ω_GC^C (G wrt C, in C frame)
            omega_GC_C = np.zeros((nbSteps, 3))
            omega_GC_C[0] = so3_log_vec(R_GC[mc_trial, agent_idx, 0].T @ R_GC[mc_trial, agent_idx, 1]) / tstep_eff
            for k in range(1, nbSteps - 1):
                Rm = R_GC[mc_trial, agent_idx, k - 1].T @ R_GC[mc_trial, agent_idx, k + 1]
                omega_GC_C[k] = so3_log_vec(Rm) / (2.0 * tstep_eff)
            omega_GC_C[nbSteps - 1] = so3_log_vec(R_GC[mc_trial, agent_idx, nbSteps - 2].T @ R_GC[mc_trial, agent_idx, nbSteps - 1]) / tstep_eff

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
                R_CG = R_GC[mc_trial, agent_idx, k].T
                omega_IG_C = -R_CG @ omega_GI_G[mc_trial, k]  # ω_IG^C = -R_CG @ ω_GI^G
                omega_CI_C[mc_trial, agent_idx, k] = omega_IG_C - omega_GC_C[k]  # ω_IC^C = ω_IG^C - ω_CG^C (note: omega_GC_C stores ω_CG^C)

            for j in range(nbSteps):
                if j == 0:
                    gyro_bias_state[mc_trial, agent_idx] = np.random.normal(0.0, sigma_bg, size=3)
                else:
                    gyro_bias_state[mc_trial, agent_idx] = phi_g * gyro_bias_state[mc_trial, agent_idx] + inc_std_g * np.random.normal(0.0, 1.0, size=3)
                omega_CI_C_m[mc_trial, agent_idx, j] = omega_CI_C[mc_trial, agent_idx, j] + nu[agent_idx, j] + gyro_bias_state[mc_trial, agent_idx]

        # Specific force and torque
        for agent_idx in range(config.num_agents):
            for j in range(nbSteps):
                r_c_I = state_C_I[mc_trial, agent_idx, j, 0:3]
                r_magnitude = np.linalg.norm(r_c_I)
                gravity_I = -config.mu_ref * r_c_I / (r_magnitude**3)
                if j == 0:
                    non_grav_accel_I = (state_C_I[mc_trial, agent_idx, 1, 3:6] - state_C_I[mc_trial, agent_idx, 0, 3:6]) / tstep_eff - gravity_I
                elif j == nbSteps - 1:
                    non_grav_accel_I = (state_C_I[mc_trial, agent_idx, j, 3:6] - state_C_I[mc_trial, agent_idx, j - 1, 3:6]) / tstep_eff - gravity_I
                else:
                    non_grav_accel_I = (state_C_I[mc_trial, agent_idx, j + 1, 3:6] - state_C_I[mc_trial, agent_idx, j - 1, 3:6]) / (2 * tstep_eff) - gravity_I
                f_specific_S[mc_trial, agent_idx, j] = R_IC[mc_trial, agent_idx, j].T @ non_grav_accel_I

                if j == 0:
                    accel_bias_state[mc_trial, agent_idx] = np.random.normal(0.0, sigma_ba, size=3)
                else:
                    accel_bias_state[mc_trial, agent_idx] = phi_a * accel_bias_state[mc_trial, agent_idx] + inc_std_a * np.random.normal(0.0, 1.0, size=3)
                f_specific_S_m[mc_trial, agent_idx, j] = f_specific_S[mc_trial, agent_idx, j] + np.random.multivariate_normal(np.zeros(3), config.COV_ACCEL_ASTRIX) + accel_bias_state[mc_trial, agent_idx]

                if j == 0:
                    omega_dot = (omega_CI_C[mc_trial, agent_idx, 1] - omega_CI_C[mc_trial, agent_idx, 0]) / tstep_eff
                elif j == nbSteps - 1:
                    omega_dot = (omega_CI_C[mc_trial, agent_idx, j] - omega_CI_C[mc_trial, agent_idx, j - 1]) / tstep_eff
                else:
                    omega_dot = (omega_CI_C[mc_trial, agent_idx, j + 1] - omega_CI_C[mc_trial, agent_idx, j - 1]) / (2 * tstep_eff)
                J_omega = config.J @ omega_CI_C[mc_trial, agent_idx, j]
                tau_specific_S[mc_trial, agent_idx, j] = config.J @ omega_dot + np.cross(omega_CI_C[mc_trial, agent_idx, j], J_omega)

        print("done")

    # ---------- Write output files ----------
    print("\n[STEP 4] Writing output files...")
    # Ensure the file exists
    os.makedirs(base_output_file, exist_ok=True)
    # Write the trajectory config for this run
    trajectory_config_filepath = os.path.join(base_output_file, "trajectory_config.json")
    with open(trajectory_config_filepath, 'w') as f:
        payload = config.model_dump()
        json.dump(payload, f, indent=2)

    camera_obj = {"focal_length": config.FOCAL_LENGTH_PX, "resolution": config.CAMERA_RESOLUTION, "lens_flare": config.LENS_FLARE} # We prob dont need to do this and can just pass a config

    for mc_trial in range(config.num_mc):
        mc_folder = os.path.join(base_output_file, f"mc_trial_{mc_trial}")
        os.makedirs(mc_folder, exist_ok=True)

        # Select the monte carlo trial
        r_CG_G_mc = r_CG_G[mc_trial]
        v_CG_G_mc = v_CG_G[mc_trial]
        q_GC_mc = q_GC[mc_trial]
        q_IC_mc = q_IC[mc_trial]
        omega_CI_C_mc = omega_CI_C[mc_trial]
        state_C_I_mc = state_C_I[mc_trial]
        q_IC_m_mc = q_IC_m[mc_trial]
        omega_CI_C_m_mc = omega_CI_C_m[mc_trial]
        f_specific_S_m_mc = f_specific_S_m[mc_trial]
        tau_specific_S_mc = tau_specific_S[mc_trial]
        R_IG_mc = R_IG[mc_trial]

        state_A_I_mc = state_A_I[mc_trial]
        omega_GI_G_mc = omega_GI_G[mc_trial]
        r_OG_G_mc = r_OG_G[mc_trial]
        az_G_mc = az_G[mc_trial]
        el_G_mc = el_G[mc_trial]
        q_IG_mc = q_IG[mc_trial]
        r_GO_I_mc = r_GO_I[mc_trial]
        v_GO_I_mc = v_GO_I[mc_trial]
        az_I_mc = az_I[mc_trial]
        el_I_mc = el_I[mc_trial]

        # Generate plots
        # try:
        plot_trial_trajectories(
            state_A_I=state_A_I_mc,
            state_C_I=state_C_I_mc,
            r_CG_G=r_CG_G_mc,
            R_IG_all=R_IG_mc,
            out_dir=mc_folder,
            rotMode_Gframe=config.rotMode_Gframe,
            show=False,
            save=True,
            mc_idx=mc_trial
        )

        for agent_idx in range(config.num_agents):
            # Select the agent
            r_CG_G_mc_ag = r_CG_G_mc[agent_idx]
            v_CG_G_mc_ag = v_CG_G_mc[agent_idx]
            q_GC_mc_ag = q_GC_mc[agent_idx]
            q_IC_mc_ag = q_IC_mc[agent_idx]
            omega_CI_C_mc_ag = omega_CI_C_mc[agent_idx]
            state_C_I_mc_ag = state_C_I_mc[agent_idx]
            q_IC_m_mc_ag = q_IC_m_mc[agent_idx]
            omega_CI_C_m_mc_ag = omega_CI_C_m_mc[agent_idx]
            f_specific_S_m_mc_ag = f_specific_S_m_mc[agent_idx]
            tau_specific_S_mc_ag = tau_specific_S_mc[agent_idx]

            agent_folder = os.path.join(mc_folder, f"agent_{agent_idx}")
            os.makedirs(agent_folder, exist_ok=True)

            # Compute and print range statistics
            ranges = np.linalg.norm(r_CG_G_mc_ag, axis=1)
            r_min, r_max = float(np.min(ranges)), float(np.max(ranges))
            print(f"  [INFO] MC{mc_trial} Agent{agent_idx}: range=[{r_min:.2f}, {r_max:.2f}]m, focal_length={camera_obj['focal_length']}px")

            write_camera_trajectory(
                output_dir=agent_folder,
                nbSteps=nbSteps,
                r_GO_I=r_GO_I_mc,
                q_IG=q_IG_mc,
                r_CO_I=state_C_I_mc_ag[:, 0:3],
                q_IC=q_IC_mc_ag,
                sun_az=az_I_mc,
                sun_el=el_I_mc,
            )
            gtvalues_filepath = write_gtvalues(
                output_dir=agent_folder,
                nbSteps=nbSteps,
                timestamps=timestamps,
                J=config.J,
                r_AG_G=r_AG_G,
                q_GC=q_GC_mc_ag,
                q_IG=q_IG_mc,
                q_IC=q_IC_mc_ag,
                omega_GI_G=omega_GI_G_mc,
                omega_CI_C=omega_CI_C_mc_ag,
                r_CG_G=r_CG_G_mc_ag,
                v_CG_G=v_CG_G_mc_ag,
                r_OG_G=r_OG_G_mc,
                az_G=az_G_mc,
                el_G=el_G_mc,
                state_A_I=state_A_I_mc,
                r_GO_I=r_GO_I_mc,
                v_GO_I=v_GO_I_mc,
                state_C_I=state_C_I_mc_ag
            )
            write_json(output_dir=agent_folder, gtvalues_filepath=gtvalues_filepath, camera_obj=camera_obj, tstep_eff=tstep_eff, tend=config.tend)

            write_sensormeasurements(output_dir=agent_folder,
                                        nbSteps=nbSteps,
                                        timestamps=timestamps,
                                        q_IC_m=q_IC_m_mc_ag,
                                        omega_CI_C_m=omega_CI_C_m_mc_ag,
                                        state_A_I=state_A_I_mc,
                                        f_specific_S_m=f_specific_S_m_mc_ag,
                                        tau_specific_S=tau_specific_S_mc_ag
                                    )
            write_config(output_dir=agent_folder,
                            nbSteps=nbSteps,
                            camera_obj=camera_obj,
                            tstep_eff=tstep_eff,
                            child_ss=child_ss[mc_trial],
                            path_mode=config.path_mode,
                            rotMode_Gframe=config.rotMode_Gframe,
                            agent_idx=agent_idx,
                            mu_ref=config.mu_ref,
                            h_orbit=config.h_orbit,
                            tend=config.tend,
                            inc=inc[mc_trial],
                            ecc=ecc[mc_trial],)

    print(f"\n[DONE] Output written to: {base_output_file}")
    print(f"       Master seed: {config.seed}")
    print(f"       Mode: {config.path_mode}")
    print(f"       {config.num_mc} MC trials, {config.num_agents} agent(s) each")


# ============================================================================
# CLI Entry Point
# ============================================================================
# TODO fix this with the config refactor
def main():
    print("=" * 60)
    print("UNIFIED TRAJECTORY GENERATOR")
    print("=" * 60)

    # ---------- User inputs ----------
    num_agents = int(input("Number of agents for inspection scenario: ").strip())
    rotMode_Gframe = input("Mode Setting (1: Inertial, 2: Hill, 3: Tumbling): ").strip()
    num_mc = int(input("Number of samples for Monte Carlo Simulation: ").strip())

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    args, _ = parser.parse_known_args()

    seed = None
    if args.seed is not None:
        seed = int(args.seed)
    env_seed = os.getenv("SATSLAM_SEED")
    if env_seed:
        seed = int(env_seed)
    
    if rotMode_Gframe == "1":
        path_mode = "inertial"
    elif rotMode_Gframe == "2":
        path_mode = "hill"
    elif rotMode_Gframe == "3":
        path_mode = "tumbling"
    else:
        raise ValueError("Mode must be 1, 2, or 3.")
    
    config = TrajectoryConfig(
        path_mode=path_mode,
        num_agents=num_agents,
        num_mc=num_mc,
        seed=seed
    )

    print(f"\n[INFO] Master seed: {seed}")
    print(f"[INFO] Mode: {path_mode}")
    print(f"[INFO] Agents: {num_agents}, MC trials: {num_mc}")

    now = datetime.today()
    date_str = now.strftime("%m_%d_%y")  # e.g., "1207"
    time_str = now.strftime("%Y_%m_%H%M")  # e.g., "2025_12_1430"
    base_output_file = os.path.join(DEFAULT_OUTPUT_BASE, f"{date_str}_{time_str}_{config.path_mode}")
    os.makedirs(base_output_file, exist_ok=True)

    generate_trajectories(config=config, base_output_file=base_output_file)

if __name__ == "__main__":
    main()
