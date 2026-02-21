import numpy as np


def _ic_geom_from_state(r0, v0, f_px, dt):
    R = np.linalg.norm(r0)
    if R < 1e-12:
        return 0.0, 0.0, R, np.array([0.0, 0.0, 1.0]), np.eye(3)
    u = -r0 / R
    M = np.eye(3) - np.outer(u, u)
    v_perp = float(np.linalg.norm(M @ v0))
    px = float(f_px * (v_perp * dt) / R)
    return px, v_perp, R, u, M


def _required_vperp(px_min, R, f_px, dt):
    return float((px_min * R) / (f_px * dt))


def _cro_min_B_add(r0, v0, f_px, dt, n_scalar, px_min):
    # Current geometry
    px, v_perp, R, u, M = _ic_geom_from_state(r0, v0, f_px, dt)
    vperp_min = _required_vperp(px_min, R, f_px, dt)
    deficit = max(0.0, vperp_min - v_perp)
    if deficit <= 0.0:
        return 0.0, {"px": px, "v_perp": v_perp, "R": R}

    ez = np.array([0.0, 0.0, 1.0])
    kappa_z = float(np.linalg.norm(M @ ez))  # = sqrt(1 - (u·ez)^2)
    if kappa_z < 1e-9:
        # LOS ~ parallel to z: changing B alone won't help
        return np.inf, {"px": px, "v_perp": v_perp, "R": R, "kappa_z": kappa_z}
    B_add = deficit / (n_scalar * kappa_z)
    return float(B_add), {"px": px, "v_perp": v_perp, "R": R, "kappa_z": kappa_z, "deficit": deficit}


def _nmc_best_phase(r0, n_scalar, A):
    """
    Maximize ||(I - uu^T) v_inplane(phi)|| at t0 for classic bounded HCW:
        x =  A cos(nt + phi),  y = -2A sin(nt + phi)
        vx = -n A sin(nt + phi),  vy = -2n A cos(nt + phi)
    """
    R = np.linalg.norm(r0)
    u = -r0 / (R if R > 1e-12 else 1.0)
    M = np.eye(3) - np.outer(u, u)

    b_cos = np.array([0.0, -2.0*n_scalar*A, 0.0])  # vy term
    b_sin = np.array([-n_scalar*A, 0.0, 0.0])      # -vx term
    C = M @ b_cos
    S = -M @ b_sin
    C2, S2, CS = float(C @ C), float(S @ S), float(C @ S)
    phi_star = 0.5 * np.arctan2(2.0 * CS, (C2 - S2))
    return float(phi_star)


def detect_degeneracy(r0, v0, f_px, dt, px_min, rho_max):
    """
    Shared geometry test at t0 for all modes:
      - pixel-motion surrogate threshold (px >= px_min)
      - radial-dominance cap (rho <= rho_max)
    Returns: (is_bad, stats_dict)
    """
    px, v_perp, R, u, M = _ic_geom_from_state(r0, v0, f_px, dt)
    vmag = float(np.linalg.norm(v0)) + 1e-12
    rho  = float(abs(v0 @ u) / vmag)
    is_bad = (px < px_min) or (rho > rho_max)
    return is_bad, {"px": px, "rho": rho, "R": R, "v_perp": v_perp, "u": u, "M": M}


def repair_inertial_nmc(r0, n_scalar, A):
    """Re-phase in-plane (keep A) to maximize transverse motion; keep z terms as-is."""
    phi_star = _nmc_best_phase(r0, n_scalar, A)
    x  =  A * np.cos(phi_star)
    y  =  0.0
    vx = -n_scalar * A * np.sin(phi_star)
    vy = -2.0      * n_scalar * A * np.cos(phi_star)
    return x, y, vx, vy


def repair_inertial_cro(r0, v0, f_px, dt, n_scalar, px_min, vz0):
    """
    Add the minimal ΔB (via zdot) needed to meet px_min at t0, keep x,y unchanged.
    Returns new (z0, vz0_new). Enforces z-quadrature at t0.
    """
    B_add, _stats = _cro_min_B_add(r0, v0, f_px, dt, n_scalar, px_min)
    if not np.isfinite(B_add) or (B_add <= 0.0):
        return 0.0, vz0  # no change possible/needed
    B_prev   = abs(vz0) / max(n_scalar, 1e-12)
    B_new    = B_prev + B_add
    z0_new   = 0.0
    vz0_new  = n_scalar * B_new
    return z0_new, vz0_new


def repair_hill(r0, v0, n_scalar, f_px, dt, px_min, M_hint=None):
    """
    Minimal, in-plane fix for Hill ICs that preserves the CW constraint x0 = -ydot0/(2n).
    Strategy: scale |ydot0| just enough so v_perp meets px_min.
    """
    px, v_perp, R, u, M = _ic_geom_from_state(r0, v0, f_px, dt)
    vperp_req = _required_vperp(px_min, R, f_px, dt)
    if v_perp >= vperp_req:  # nothing to do
        return None
    # contribution of unit y-velocity to transverse speed
    k = float(np.linalg.norm(M @ np.array([0.0, 1.0, 0.0])))
    if k < 1e-9:
        # Fallback: tiny z-kick if LOS ~ y-direction
        vz_needed = vperp_req
        return {"x0": r0[0], "y0": r0[1], "z0": 0.0,
                "vx0": v0[0], "vy0": v0[1], "vz0": vz_needed,
                "force_z_quadrature": True}
    # scale vy to reach vperp_req; keep sign
    vy_sign   = 1.0 if v0[1] >= 0.0 else -1.0
    vy_new    = vy_sign * max(abs(v0[1]), vperp_req / k)
    x0_new    = -vy_new / (2.0 * n_scalar)
    return {"x0": x0_new, "y0": r0[1], "z0": r0[2],
            "vx0": v0[0], "vy0": vy_new, "vz0": v0[2]}


def repair_tumbling(r0, v0, f_px, dt, px_min, n_scalar=None):
    """
    Tumbling: target is rotating; relative translation may be too weak.
    Inject a minimal z-quadrature kick (like CRO) to meet px_min.
    """
    # Treat like CRO with n_scalar=1 for the algebra if n not provided
    n_eff = 1.0 if (n_scalar is None) else float(n_scalar)
    z0_new, vz0_new = repair_inertial_cro(r0, v0, f_px, dt, n_eff, px_min, vz0=v0[2])
    return {"x0": r0[0], "y0": r0[1], "z0": z0_new,
            "vx0": v0[0], "vy0": v0[1], "vz0": vz0_new}


# ------------------ Inertia Observability Helpers ----------------------------


def check_omega_excitation(omega_dir, off_axis_min=0.3):
    """
    Check if an angular velocity direction provides sufficient excitation
    for inertia estimation.

    For inertia to be observable from torque-free Euler dynamics, ω must NOT
    be aligned with a principal axis. When ω aligns with a principal axis of
    an axisymmetric (or near-axisymmetric) body, the precession/nutation
    vanishes, causing ω̇ ≈ 0 and making inertia unobservable.

    This function checks that at least 2 components of ω are significant,
    ensuring the angular velocity has multi-axis content.

    Parameters
    ----------
    omega_dir : array-like, shape (3,)
        Unit direction vector of angular velocity (normalized).
    off_axis_min : float
        Minimum ratio for the second-largest component.
        Default 0.3 ensures ω is at least ~17° off any principal axis.

    Returns
    -------
    is_excited : bool
        True if excitation is sufficient for inertia estimation.
    stats : dict
        Diagnostic statistics:
        - 'sorted_abs': sorted absolute values [smallest, middle, largest]
        - 'second_largest': the second-largest component magnitude
        - 'dominant_axis': index of the dominant axis (0=x, 1=y, 2=z)
        - 'off_axis_angle_deg': angle off the nearest principal axis
    """
    omega_dir = np.asarray(omega_dir, dtype=float)
    norm = np.linalg.norm(omega_dir)
    if norm < 1e-12:
        return False, {'sorted_abs': [0, 0, 0], 'second_largest': 0,
                       'dominant_axis': 0, 'off_axis_angle_deg': 0}

    # Normalize if not already
    d = omega_dir / norm
    abs_d = np.abs(d)

    # Sort to find [smallest, middle, largest]
    sorted_idx = np.argsort(abs_d)
    sorted_abs = abs_d[sorted_idx]

    # The second-largest component determines off-axis content
    second_largest = sorted_abs[1]
    dominant_axis = sorted_idx[2]

    # Compute angle off the nearest principal axis
    # cos(theta) = |d_max|, so theta = arccos(|d_max|)
    off_axis_angle_rad = np.arccos(np.clip(sorted_abs[2], 0, 1))
    off_axis_angle_deg = np.rad2deg(off_axis_angle_rad)

    is_excited = second_largest >= off_axis_min

    stats = {
        'sorted_abs': sorted_abs.tolist(),
        'second_largest': float(second_largest),
        'dominant_axis': int(dominant_axis),
        'off_axis_angle_deg': float(off_axis_angle_deg),
    }

    return is_excited, stats


def sample_excited_omega_direction(rng, off_axis_min=0.3, max_retries=100):
    """
    Sample a random angular velocity direction that provides sufficient
    excitation for inertia estimation.

    Uses rejection sampling to ensure the sampled direction is not aligned
    with any principal axis, which is required for inertia observability
    in torque-free rigid body dynamics.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.
    off_axis_min : float
        Minimum ratio for the second-largest component of the direction.
        Default 0.3 ensures ω is at least ~17° off any principal axis.
    max_retries : int
        Maximum rejection sampling attempts before raising an error.

    Returns
    -------
    d : ndarray, shape (3,)
        Unit direction vector with sufficient off-axis content.
    stats : dict
        Diagnostic statistics from check_omega_excitation.

    Raises
    ------
    RuntimeError
        If a valid direction cannot be found within max_retries attempts.
    """
    for _ in range(max_retries):
        # Sample uniformly on the sphere via normalized Gaussians
        d = np.array([rng.normal(), rng.normal(), rng.normal()])
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        d = d / norm

        is_excited, stats = check_omega_excitation(d, off_axis_min)
        if is_excited:
            return d, stats

    raise RuntimeError(
        f"sample_excited_omega_direction: could not find non-principal-axis "
        f"direction after {max_retries} attempts (off_axis_min={off_axis_min})"
    )


def sample_inertia_excited_omega_direction(rng, J, min_asymmetry_component=0.4,
                                            off_axis_min=0.3, max_retries=100):
    """
    Sample omega direction that excites the largest inertia asymmetry.

    For a triaxial body, inertia observability requires omega to have significant
    component along the axis with the most different principal moment. This function
    identifies the "outlier" principal axis and biases sampling toward it.

    For example, with J = diag(0.21, 0.38, 0.41):
    - Ixx = 0.21 is the outlier (most different from the others)
    - Iyy ≈ Izz ≈ 0.40 are similar
    - Omega should have significant x-component to excite the Ixx vs Iyy/Izz asymmetry

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.
    J : ndarray, shape (3, 3)
        Inertia tensor (must be diagonal in principal axes).
    min_asymmetry_component : float
        Minimum absolute value of omega component along the outlier axis.
        Default 0.4 ensures at least 40% of omega magnitude is along the
        axis that excites the largest inertia asymmetry.
    off_axis_min : float
        Minimum ratio for the second-largest component (same as before).
        Ensures omega has multi-axis content for general excitation.
    max_retries : int
        Maximum rejection sampling attempts.

    Returns
    -------
    d : ndarray, shape (3,)
        Unit direction vector with inertia-aware excitation.
    stats : dict
        Diagnostic statistics including asymmetry information.
    """
    # Extract principal moments (assuming J is diagonal)
    I_diag = np.diag(J)

    # Find the "outlier" axis - the one most different from the mean of the other two
    # This is the axis whose excitation provides most observability
    asymmetry_scores = np.zeros(3)
    for i in range(3):
        others = [I_diag[j] for j in range(3) if j != i]
        mean_others = np.mean(others)
        asymmetry_scores[i] = abs(I_diag[i] - mean_others)

    outlier_axis = int(np.argmax(asymmetry_scores))
    max_asymmetry = asymmetry_scores[outlier_axis]

    for attempt in range(max_retries):
        # Sample uniformly on the sphere
        d = np.array([rng.normal(), rng.normal(), rng.normal()])
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        d = d / norm

        # Check 1: General off-axis requirement
        is_excited, base_stats = check_omega_excitation(d, off_axis_min)
        if not is_excited:
            continue

        # Check 2: Sufficient component along the outlier axis
        outlier_component = abs(d[outlier_axis])
        if outlier_component < min_asymmetry_component:
            continue

        # Both checks passed
        stats = {
            **base_stats,
            'outlier_axis': outlier_axis,
            'outlier_axis_name': ['x', 'y', 'z'][outlier_axis],
            'outlier_component': float(outlier_component),
            'inertia_diagonal': I_diag.tolist(),
            'asymmetry_scores': asymmetry_scores.tolist(),
            'max_asymmetry': float(max_asymmetry),
        }
        return d, stats

    raise RuntimeError(
        f"sample_inertia_excited_omega_direction: could not find valid direction "
        f"after {max_retries} attempts. outlier_axis={outlier_axis}, "
        f"min_asymmetry_component={min_asymmetry_component}, off_axis_min={off_axis_min}"
    )


def validate_omega_timeseries_excitation(omega_timeseries, dt_array=None,
                                          omega_dot_min=1e-5,
                                          off_axis_min=0.3):
    """
    Validate that an omega time series has sufficient excitation for
    inertia estimation.

    This function checks TWO conditions:
    1. Initial ω direction is not aligned with a principal axis
    2. There is sufficient angular acceleration (ω̇) over the trajectory

    For inertia estimation via Euler's equation I·ω̇ + ω×(I·ω) = 0,
    we need |ω̇| to be non-negligible. When |ω̇| ≈ 0, the equation
    is satisfied for ANY inertia, making it unobservable.

    Parameters
    ----------
    omega_timeseries : ndarray, shape (N, 3)
        Time series of angular velocity vectors [rad/s].
    dt_array : ndarray, shape (N-1,), optional
        Time steps between samples. If None, assumes uniform dt=1.
    omega_dot_min : float
        Minimum acceptable mean |ω̇| [rad/s²]. Default 1e-5.
    off_axis_min : float
        Minimum off-axis content for initial ω direction.

    Returns
    -------
    is_valid : bool
        True if excitation is sufficient for inertia estimation.
    diagnostics : dict
        Detailed diagnostics including:
        - 'omega_dir_excited': bool, whether initial direction is good
        - 'mean_omega_dot': mean |ω̇| over trajectory
        - 'max_omega_dot': maximum |ω̇| over trajectory
        - 'omega_dot_sufficient': bool, whether ω̇ meets threshold
        - 'initial_omega_stats': stats from check_omega_excitation
    """
    omega_timeseries = np.asarray(omega_timeseries, dtype=float)
    N = omega_timeseries.shape[0]

    if N < 2:
        return False, {'error': 'Need at least 2 samples'}

    # Check 1: Initial omega direction
    omega_0 = omega_timeseries[0]
    omega_0_norm = np.linalg.norm(omega_0)
    if omega_0_norm < 1e-12:
        return False, {'error': 'Initial omega is zero'}

    omega_dir_excited, initial_omega_stats = check_omega_excitation(
        omega_0 / omega_0_norm, off_axis_min
    )

    # Check 2: Angular acceleration
    if dt_array is None:
        dt_array = np.ones(N - 1)
    else:
        dt_array = np.asarray(dt_array, dtype=float)

    # Compute finite-difference ω̇
    omega_dot_norms = []
    for i in range(N - 1):
        omega_dot = (omega_timeseries[i + 1] - omega_timeseries[i]) / dt_array[i]
        omega_dot_norms.append(np.linalg.norm(omega_dot))

    omega_dot_norms = np.array(omega_dot_norms)
    mean_omega_dot = float(np.mean(omega_dot_norms))
    max_omega_dot = float(np.max(omega_dot_norms))
    omega_dot_sufficient = mean_omega_dot >= omega_dot_min

    is_valid = omega_dir_excited and omega_dot_sufficient

    diagnostics = {
        'omega_dir_excited': omega_dir_excited,
        'mean_omega_dot': mean_omega_dot,
        'max_omega_dot': max_omega_dot,
        'omega_dot_sufficient': omega_dot_sufficient,
        'omega_dot_min_threshold': omega_dot_min,
        'initial_omega_stats': initial_omega_stats,
        'initial_omega_norm_rad_s': float(omega_0_norm),
        'initial_omega_norm_deg_s': float(np.rad2deg(omega_0_norm)),
    }

    return is_valid, diagnostics


# ------------------ Inertial Case ----------------------------------------


def inertial_nmc(num_mc, num_agents, n_scalar=None, focal_length_px=None,
                 kf_dt=None, px_min=6.0, rho_max=0.90, R0_const=100.0,
                 rngs_mc=None):
    """
    inertial_nmc
    ------------
    Natural-Motion Circumnavigation (NMC) about a circular chief in LEO.
    Bounded, coast-only HCW solution with ~constant relative range.
      x(t)=A cos(nt),  y(t)=-2A sin(nt),  z(t)=√3 A cos(nt)
    with ICs at t0: x0=A, y0=0, z0=√3 A, xdot0=0, ydot0=-2 n A, zdot0=0,
    where A = R0_const/2 and n = n_scalar.

    Returns Hill/LVLH initial state arrays (no drift), inertial path_mode tag,
    and ω_GI_G_0 = [0,0,0] (G-frame fixed in inertial mode).
    """
    x_0     = np.zeros((num_mc, num_agents))
    y_0     = np.zeros((num_mc, num_agents))
    z_0     = np.zeros((num_mc, num_agents))
    xdot_0  = np.zeros((num_mc, num_agents))
    ydot_0  = np.zeros((num_mc, num_agents))
    zdot_0  = np.zeros((num_mc, num_agents))

    A_nom = 0.5 * float(R0_const)

    for i in range(num_mc):
        
        # Derive a per-agent RNG in a reproducible way if none supplied
        base_rng = rngs_mc[i] if (rngs_mc is not None) else np.random.default_rng(12345 + i)

        for a in range(num_agents):
            
            # Small amplitude jitter (e.g., ±10%)
            eps = float(base_rng.uniform(-0.10, 0.10))
            A   = max(1e-6, A_nom * (1.0 + eps))

            # Random in-plane phase
            phi = float(base_rng.uniform(0.0, 2.0*np.pi))

            # Classic NMC ICs at t0 with phase phi
            x0, y0   =  A*np.cos(phi), -2.0*A*np.sin(phi)
            vx0, vy0 = -n_scalar*A*np.sin(phi), -2.0*n_scalar*A*np.cos(phi)

            # Keep existing z-model (can also jitter lightly if desired)
            B = np.sqrt(3.0) * A
            z0, vz0 = B, 0.0

            x_0[i,a], y_0[i,a], z_0[i,a]       = x0, y0, z0
            xdot_0[i,a], ydot_0[i,a], zdot_0[i,a] = vx0, vy0, vz0

            # ---- shared degeneracy detector + NMC repair ----
            if (focal_length_px is not None) and (kf_dt is not None):
                r0 = np.array([x0, y0, z0], float)
                v0 = np.array([vx0, vy0, vz0], float)
                bad, _ = detect_degeneracy(r0, v0, focal_length_px, float(kf_dt), px_min, rho_max)
                if bad:
                    xn, yn, vxn, vyn = repair_inertial_nmc(r0, n_scalar, A)
                    x_0[i,a], y_0[i,a], xdot_0[i,a], ydot_0[i,a] = xn, yn, vxn, vyn
                    # keep z terms as set
                    # Raise error if still degenerate
                    r1 = np.array([x_0[i,a], y_0[i,a], z_0[i,a]], float)
                    v1 = np.array([xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]], float)
                    bad2, stats2 = detect_degeneracy(r1, v1, focal_length_px, float(kf_dt), px_min, rho_max)
                    if bad2:
                        raise RuntimeError(f"NMC IC unrecoverable: px={stats2['px']:.2f}, rho={stats2['rho']:.3f}")
    omega_GI_G_0 = [0.0, 0.0, 0.0]
    path_mode    = "Inertial"
    return x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, path_mode


def inertial_cro(num_mc, num_agents, n_scalar=None, focal_length_px=None,
                 kf_dt=None, px_min=6.0, rho_max=0.90, R_nom=None,
                 span_frac=None, r_min=None, r_max=None, A_xy=None, B_z=None,
                 rngs_mc=None):
    """
    inertial_cro
    ------------
    Drift-free HCW bounded ellipse around the target:
      x =  A cos(n t + phi),   y = -2A sin(n t + phi),   z = B cos(n t + phi_z)

    ***Tied-to-NMC sizing (recommended):***
      Pass R_nom (equal to NMC's constant range) and span_frac.
      We hold A = 0.5*R_nom (== NMC A), so r_min = 2A = R_nom (same as NMC),
      and set r_max = (1+span_frac)*R_nom  ->  B = sqrt(r_max^2 - A^2).

    We initialize at *z-quadrature* (z(0)=0, zdot(0)=n*B) to inject immediate cross-track
    velocity. In-plane (x,y) start at a seeded random phase phi for variability.

    Returns Hill/LVLH IC arrays and path_mode 'Inertial'.
    """

    # ---------- decide base A,B from inputs ----------
    # Geometry: r_min = 2A, r_max = sqrt((2A)^2 + B^2)
    # Therefore: B = sqrt(r_max^2 - (2A)^2)
    if (R_nom is not None) and (span_frac is not None):
        # Tied-to-NMC (preferred)
        A_base   = 0.5 * float(R_nom)                     # => r_min = 2A = R_nom
        r_max_in = (1.0 + float(span_frac)) * float(R_nom)
        B_base   = float(np.sqrt(max(r_max_in**2 - (2.0 * A_base)**2, 0.0)))
    elif (r_min is not None) and (r_max is not None):
        A_base = 0.5 * float(r_min)  # so 2*A_base = r_min
        B_base = float(np.sqrt(max(float(r_max)**2 - (2.0 * A_base)**2, 0.0)))
    elif (A_xy is not None) and (B_z is not None):
        A_base = float(A_xy)
        B_base = float(B_z)
    else:
        # Sensible tied-to-NMC default ~100 m with +20% upper swing
        R_nom   = 100.0
        span_frac = 0.20
        A_base  = 0.5 * R_nom  # so 2*A_base = R_nom
        r_max_in = (1.0 + span_frac) * R_nom
        B_base  = float(np.sqrt(max(r_max_in**2 - (2.0 * A_base)**2, 0.0)))

    # ---------- allocate ----------
    x_0     = np.zeros((num_mc, num_agents), dtype=float)
    y_0     = np.zeros((num_mc, num_agents), dtype=float)
    z_0     = np.zeros((num_mc, num_agents), dtype=float)
    xdot_0  = np.zeros((num_mc, num_agents), dtype=float)
    ydot_0  = np.zeros((num_mc, num_agents), dtype=float)
    zdot_0  = np.zeros((num_mc, num_agents), dtype=float)

    # Small amplitude jitter (±10%) for variability; phase ∈ U[0, 2π)
    amp_jitter = 0.10

    for i in range(num_mc):
        rng = rngs_mc[i] if (rngs_mc is not None) else np.random.default_rng(7919 + i)

        for a in range(num_agents):
            eps_xy = float(rng.uniform(-amp_jitter, amp_jitter))
            eps_z  = float(rng.uniform(-amp_jitter, amp_jitter))
            A      = max(1e-9, A_base * (1.0 + eps_xy))
            B      = max(1e-9, B_base * (1.0 + eps_z))

            # In-plane random start phase for (x,y)
            phi = float(rng.uniform(0.0, 2.0*np.pi))

            # In-plane ICs at t0 with phase phi
            # x = A cos(phi), y = -2A sin(phi)
            # vx = -n A sin(phi), vy = -2 n A cos(phi)
            x0   =  A * np.cos(phi)
            y0   = -2.0 * A * np.sin(phi)
            vx0  = -n_scalar * A * np.sin(phi)
            vy0  = -2.0 * n_scalar * A * np.cos(phi)

            # z-quadrature at t0: z(0)=0,  zdot(0)=+n*B
            z0   = 0.0
            vz0  = n_scalar * B

            x_0[i, a], y_0[i, a], z_0[i, a]       = x0, y0, z0
            xdot_0[i, a], ydot_0[i, a], zdot_0[i, a] = vx0, vy0, vz0

            # ---------- shared detector + CRO repair ----------
            if (focal_length_px is not None) and (kf_dt is not None):
                r0 = np.array([x0, y0, z0], float)
                v0 = np.array([vx0, vy0, vz0], float)
                bad, _ = detect_degeneracy(r0, v0, focal_length_px, float(kf_dt), px_min, rho_max)
                if bad:
                    zN, vzN = repair_inertial_cro(r0, v0, focal_length_px, float(kf_dt), n_scalar, px_min, vz0=vz0)
                    z_0[i, a], zdot_0[i, a] = zN, vzN
                    # x,y unchanged — extra z-velocity injects usable v_perp
                    # Check if still degenerate
                    r1 = np.array([x_0[i,a], y_0[i,a], z_0[i,a]], float)
                    v1 = np.array([xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]], float)
                    bad2, stats2 = detect_degeneracy(r1, v1, focal_length_px, float(kf_dt), px_min, rho_max)

                    if bad2:
                        # z-repair failed (likely LOS ~ z-axis). Try phase shift to get
                        # better in-plane geometry. Shift phase by 90° to maximize
                        # transverse in-plane velocity component. (Same as tumbling case)
                        phi_new = phi + np.pi / 2.0
                        x0_new = A * np.cos(phi_new)
                        y0_new = -2.0 * A * np.sin(phi_new)
                        vx0_new = -n_scalar * A * np.sin(phi_new)
                        vy0_new = -2.0 * n_scalar * A * np.cos(phi_new)

                        x_0[i, a], y_0[i, a] = x0_new, y0_new
                        xdot_0[i, a], ydot_0[i, a] = vx0_new, vy0_new
                        # Reset z to quadrature with boosted B
                        z_0[i, a] = 0.0
                        zdot_0[i, a] = vzN  # keep the boosted vz

                        r2 = np.array([x_0[i,a], y_0[i,a], z_0[i,a]], float)
                        v2 = np.array([xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]], float)
                        bad3, stats3 = detect_degeneracy(r2, v2, focal_length_px, float(kf_dt), px_min, rho_max)
                        if bad3:
                            raise RuntimeError(f"CRO IC unrecoverable after phase shift: px={stats3['px']:.2f}, rho={stats3['rho']:.3f}")

    omega_GI_G_0 = [0.0, 0.0, 0.0]
    path_mode    = "Inertial"
    return x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, path_mode


def init_inertial(num_mc, num_agents, focal_length_px, kf_dt=None,
                  px_min=6.0, rho_max=0.90, n_scalar=None,
                  R0_const=100.0,     # NMC range (≈ constant)
                  variant="nmc",      # "nmc" or "cro"
                  rngs_mc=None
                  ):
    """Initialize Hill-form ICs for the inertial (G fixed to I) case.

    Parameters
    ----------
    variant : {"nmc","cro"}
        - "nmc": Natural-Motion Circumnavigation (constant-range, drift-free)
        - "cro": Bounded CRO ellipse with optional out-of-plane amplitude

    Returns: x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, path_mode
    """
    if variant.lower() == "nmc":
        return inertial_nmc(num_mc, num_agents, n_scalar=n_scalar, 
                            focal_length_px=focal_length_px, kf_dt=kf_dt,
                            px_min=px_min, rho_max=rho_max, R0_const=R0_const,
                            rngs_mc=rngs_mc)
    elif variant.lower() == "cro":
        # span_frac=2.0 provides ~81% actual range variation after repair
        # This gives tri_angle_min ~1.27° (above 1.0° threshold) and px_min ~3.2
        # Validated via sweep_inertial_trajectories.py parameter sweep
        return inertial_cro(num_mc, num_agents, n_scalar=n_scalar,
                            focal_length_px=focal_length_px, kf_dt=kf_dt,
                            px_min=px_min, rho_max=rho_max, R_nom=R0_const,
                            span_frac=2.0, rngs_mc=rngs_mc)
    else:
        raise ValueError("init_inertial: unknown variant (use 'nmc' or 'cro').")


# ------------------ Hill Case ----------------------------------------------


def init_hill(num_mc, num_agents, n_scalar, rngs_mc=None, focal_length_px=None, kf_dt=None,
              px_min=6.0, rho_max=0.90):
    """Initialize Hill/LVLH case
    Returns: x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, path_mode
    """
    x_0     = np.zeros((num_mc, num_agents))
    y_0     = np.zeros((num_mc, num_agents))
    z_0     = np.zeros((num_mc, num_agents))
    xdot_0  = np.zeros((num_mc, num_agents))
    ydot_0  = np.zeros((num_mc, num_agents))
    zdot_0  = np.zeros((num_mc, num_agents))

    path_mode = "Hill"
    omega_GI_G_0 = [0.0, 0.0, 0.0]  # not used for Hill

    # Along-track rate
    for i in range(num_mc):
        # ydot_0[i, :] = np.random.uniform(0.3, 0.8, num_agents)  # m/s
        rng = rngs_mc[i] if (rngs_mc is not None) else np.random.default_rng(14699 + i)
        ydot_0[i, :] = rng.uniform(0.1, 0.25, size=num_agents)  # m/s

        for a in range(num_agents):
            # CW in-plane consistency
            x_0[i, a] = -ydot_0[i, a] / (2.0 * n_scalar)

            
            # Build state for guard
            r0 = np.array([x_0[i,a], y_0[i,a], z_0[i,a]], float)
            v0 = np.array([xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]], float)
            if (focal_length_px is not None) and (kf_dt is not None):
                bad, _ = detect_degeneracy(r0, v0, focal_length_px, float(kf_dt), px_min, rho_max)
                if bad:
                    fix = repair_hill(r0, v0, n_scalar, focal_length_px, float(kf_dt), px_min)
                    if fix is not None:
                        x_0[i,a], y_0[i,a], z_0[i,a]           = fix["x0"], fix["y0"], fix["z0"]
                        xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]  = fix["vx0"], fix["vy0"], fix["vz0"]
                        r1 = np.array([x_0[i,a], y_0[i,a], z_0[i,a]], float)
                        v1 = np.array([xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]], float)
                        bad2, stats2 = detect_degeneracy(r1, v1, focal_length_px, float(kf_dt), px_min, rho_max)
                        if bad2:
                            raise RuntimeError(f"Hill IC unrecoverable: px={stats2['px']:.2f}, rho={stats2['rho']:.3f}")

    return x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, path_mode


# ------------------ Tumbling Case ----------------------------------------------


def init_tumbling(num_mc, num_agents, rngs_mc=None, focal_length_px=None, kf_dt=None,
                  px_min=6.0, rho_max=0.90, n_scalar=None, R0_const=100.0,
                  omega_min_deg=0.5, omega_max_deg=2.0,
                  off_axis_min=0.3, max_omega_retries=100,
                  J=None, min_asymmetry_component=0.4):
    """Initialize tumbling case with CRO trajectory.

    Uses CRO (Cross-Range Oscillation) bounded HCW trajectory for the chaser,
    combined with target tumbling. This provides:
      - Translational parallax from CRO motion
      - Apparent feature motion from target tumbling
      - Range variation for scale observability
      - Single-burn, fuel-optimal inspection trajectory

    Parameters
    ----------
    omega_min_deg, omega_max_deg : float
        Tumbling rate magnitude bounds in deg/s. Default 0.5-2.0 deg/s is
        SLAM-friendly while providing sufficient feature motion for observability.
        Note: This caps the *initial* rate; torque-free dynamics may cause
        |ω| to vary over time for asymmetric inertia tensors.
    off_axis_min : float
        Minimum ratio for the second-largest component of ω direction.
        Default 0.3 ensures ω is at least ~17° off any principal axis,
        which is required for inertia observability. When ω aligns with
        a principal axis of an axisymmetric body, precession vanishes
        and ω̇ ≈ 0, making inertia unobservable from Euler's equation.
    max_omega_retries : int
        Maximum rejection sampling attempts for finding a valid ω direction.
    J : ndarray, shape (3, 3), optional
        Inertia tensor for inertia-aware omega sampling. When provided,
        omega direction is biased toward the axis with most different
        principal moment to maximize inertia observability.
    min_asymmetry_component : float
        When J is provided, minimum component of omega along the "outlier"
        inertia axis. Default 0.4 ensures at least 40% of omega magnitude
        excites the largest inertia asymmetry.

    Returns: x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, path_mode
    """
    # Validate required parameters
    if n_scalar is None:
        raise ValueError("n_scalar must be provided (mean motion, rad/s).")
    if (rngs_mc is not None) and (len(rngs_mc) != num_mc):
        raise ValueError("rngs_mc must have length num_mc.")
    x_0     = np.zeros((num_mc, num_agents))
    y_0     = np.zeros((num_mc, num_agents))
    z_0     = np.zeros((num_mc, num_agents))
    xdot_0  = np.zeros((num_mc, num_agents))
    ydot_0  = np.zeros((num_mc, num_agents))
    zdot_0  = np.zeros((num_mc, num_agents))

    path_mode = "Tumbling"

    # Omega: sample magnitude and axis per MC trial for proper Monte Carlo coverage
    # When inertia J is provided, omega direction is biased toward the "outlier" principal
    # axis (the one most different from the others) to maximize inertia observability.
    # Without J, omega is sampled uniformly with off-axis constraints.
    omega_GI_G_0 = np.zeros((num_mc, 3), dtype=float)

    # CRO trajectory parameters (same as inertial_cro)
    # A sets in-plane amplitude, B sets out-of-plane amplitude
    # r_min = 2A = R0_const, r_max = sqrt((2A)^2 + B^2)
    # Therefore: B = sqrt(r_max^2 - (2A)^2) = sqrt(r_max^2 - R0_const^2)
    # Use span_frac=0.20 (20% range variation) as baseline - repair will boost B if needed
    span_frac = 0.20
    A_base = 0.5 * float(R0_const)  # so 2*A_base = R0_const
    r_max_in = (1.0 + span_frac) * float(R0_const)
    B_base = float(np.sqrt(max(r_max_in**2 - (2.0 * A_base)**2, 0.0)))
    amp_jitter = 0.10  # ±10% amplitude jitter

    for i in range(num_mc):
        rng = rngs_mc[i] if (rngs_mc is not None) else np.random.default_rng(32141 + i)

        # Sample tumbling rate: magnitude uniform in [omega_min, omega_max] deg/s
        # Direction sampled with REJECTION SAMPLING to ensure inertia observability.
        # When ω aligns with a principal axis, precession/nutation vanishes (ω̇ ≈ 0),
        # making inertia unobservable from Euler's equation I·ω̇ + ω×(I·ω) = 0.
        w_deg = float(rng.uniform(omega_min_deg, omega_max_deg))
        w = np.deg2rad(w_deg)

        # Use rejection sampling to ensure ω has multi-axis content
        # If inertia J is provided, use inertia-aware sampling to excite
        # the largest asymmetry (e.g., for J with Ixx << Iyy ≈ Izz,
        # ensure omega has significant x-component)
        try:
            if J is not None:
                d, _ = sample_inertia_excited_omega_direction(
                    rng, J,
                    min_asymmetry_component=min_asymmetry_component,
                    off_axis_min=off_axis_min,
                    max_retries=max_omega_retries
                )
            else:
                d, _ = sample_excited_omega_direction(
                    rng, off_axis_min=off_axis_min,
                    max_retries=max_omega_retries
                )
        except RuntimeError as e:
            raise RuntimeError(f"init_tumbling MC trial {i}: {e}") from e

        omega_GI_G_0[i, :] = w * d

        for a in range(num_agents):
            # Apply small amplitude jitter for MC variability
            eps_xy = float(rng.uniform(-amp_jitter, amp_jitter))
            eps_z  = float(rng.uniform(-amp_jitter, amp_jitter))
            A = max(1e-9, A_base * (1.0 + eps_xy))
            B = max(1e-9, B_base * (1.0 + eps_z))

            # Random in-plane phase for viewing geometry diversity
            phi = float(rng.uniform(0.0, 2.0 * np.pi))

            # CRO ICs: x = A cos(phi), y = -2A sin(phi), z = 0 (z-quadrature)
            # vx = -n A sin(phi), vy = -2 n A cos(phi), vz = n B
            x0   =  A * np.cos(phi)
            y0   = -2.0 * A * np.sin(phi)
            z0   =  0.0  # z-quadrature start
            vx0  = -n_scalar * A * np.sin(phi)
            vy0  = -2.0 * n_scalar * A * np.cos(phi)
            vz0  =  n_scalar * B  # immediate cross-track velocity

            x_0[i, a], y_0[i, a], z_0[i, a] = x0, y0, z0
            xdot_0[i, a], ydot_0[i, a], zdot_0[i, a] = vx0, vy0, vz0

            # Degeneracy check and repair
            if (focal_length_px is not None) and (kf_dt is not None):
                r0 = np.array([x0, y0, z0], float)
                v0 = np.array([vx0, vy0, vz0], float)
                bad, stats = detect_degeneracy(r0, v0, focal_length_px, float(kf_dt), px_min, rho_max)
                if bad:
                    # Try CRO repair first (boost B via zdot)
                    zN, vzN = repair_inertial_cro(r0, v0, focal_length_px, float(kf_dt),
                                                   n_scalar, px_min, vz0=vz0)
                    z_0[i, a], zdot_0[i, a] = zN, vzN
                    # Verify fix
                    r1 = np.array([x_0[i,a], y_0[i,a], z_0[i,a]], float)
                    v1 = np.array([xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]], float)
                    bad2, stats2 = detect_degeneracy(r1, v1, focal_length_px, float(kf_dt), px_min, rho_max)

                    if bad2:
                        # z-repair failed (likely LOS ~ z-axis). Try phase shift to get
                        # better in-plane geometry. Shift phase by 90° to maximize
                        # transverse in-plane velocity component.
                        phi_new = phi + np.pi / 2.0
                        x0_new = A * np.cos(phi_new)
                        y0_new = -2.0 * A * np.sin(phi_new)
                        vx0_new = -n_scalar * A * np.sin(phi_new)
                        vy0_new = -2.0 * n_scalar * A * np.cos(phi_new)

                        x_0[i, a], y_0[i, a] = x0_new, y0_new
                        xdot_0[i, a], ydot_0[i, a] = vx0_new, vy0_new
                        # Reset z to quadrature with boosted B
                        z_0[i, a] = 0.0
                        zdot_0[i, a] = vzN  # keep the boosted vz

                        r2 = np.array([x_0[i,a], y_0[i,a], z_0[i,a]], float)
                        v2 = np.array([xdot_0[i,a], ydot_0[i,a], zdot_0[i,a]], float)
                        bad3, stats3 = detect_degeneracy(r2, v2, focal_length_px, float(kf_dt), px_min, rho_max)
                        if bad3:
                            raise RuntimeError(f"Tumbling IC unrecoverable after phase shift: px={stats3['px']:.2f}, rho={stats3['rho']:.3f}")

    return x_0, y_0, z_0, xdot_0, ydot_0, zdot_0, omega_GI_G_0, path_mode
