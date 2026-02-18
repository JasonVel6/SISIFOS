"""
This is the file complete all of the basic math functions used in this project.
If you want to use these function, you can import this file as follows:

TODO this could prob use some unit testing to make sure that we dont accidentally break anything
"""

### Import necessary libraries
import numpy as np
from scipy.linalg import logm
from scipy.linalg import expm
import math
from mathutils import Vector, Matrix, Quaternion
import random
from typing import List, Tuple


def sk(u):
    """
    Compute the skew-symmetric matrix of a vector u.

    Parameters:
    u -- a 3-element vector

    Returns:
    U_x -- the 3x3 skew-symmetric matrix
    """
    U_x = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return U_x


def unsk(U_x):
    """
    Compute a vector u from a skew-symmetric matrix U_x.

    Parameters:
    U_x -- a 3x3 skew-symmetric matrix

    Returns:
    u -- the 3-element vector
    """

    u = np.array([U_x[2, 1], U_x[0, 2], U_x[1, 0]])
    return u


def au2R(a, u):
    """
    Compute the rotation matrix given angle a and unit vector u.
    """
    U_x = sk(u)
    R = np.eye(3) + np.sin(a) * U_x + (1 - np.cos(a)) * np.dot(U_x, U_x)
    return R


def Jr(v):
    """
    Compute the Jacobian matrix Jr for a given vector v.
    """
    a = np.linalg.norm(v)

    if a == 0:
        return np.eye(3)

    U_x = sk(v)

    Jr = a - (1 - np.cos(a)) / a**2 * U_x + (a - np.sin(a)) / a**3 * np.dot(U_x, U_x)

    return Jr


def q2R(q):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    q -- a quaternion represented as a 4-element array [s, x, y, z]

    Returns:
    R -- the corresponding 3x3 rotation matrix
    """
    s = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - s * z), 2 * (x * z + s * y)],
            [2 * (x * y + s * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - s * x)],
            [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x**2 + y**2)],
        ]
    )

    return R


def qc(q):
    """
    Compute the conjugate of a quaternion.

    Parameters:
    q -- a quaternion represented as a 4-element array [s, x, y, z]

    Returns:
    qc -- the conjugate of the quaternion, where the vector part is negated
    """
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return qc


def qProd(q1, q2):
    """
    Compute the quaternion product of q1 and q2.

    Parameters:
    q1 -- first quaternion (4-element array)
    q2 -- second quaternion (4-element array)

    Returns:
    q3 -- the product of the two quaternions (4-element array)
    """
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Ensure quaternions are column vectors
    if q1.shape == (4,):
        q1 = q1.reshape(4, 1)
    if q2.shape == (4,):
        q2 = q2.reshape(4, 1)

    # Quaternion product calculation
    q3 = np.dot(
        np.block([[q1[0], -q1[1:4].T], [q1[1:4], q1[0] * np.eye(3) - sk(q1[1:4])]]), q2
    )

    return q3.flatten()


def R2au(R):
    """
    Compute the axis-angle representation (a, u) from a rotation matrix R.

    Parameters:
    R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    a (float): The rotation angle.
    u (numpy.ndarray): The rotation axis (unit vector).
    """
    AU_x = logm(R)  # Matrix logarithm to get the skew-symmetric matrix
    au = np.array([AU_x[2, 1], AU_x[0, 2], AU_x[1, 0]])  # Extract the axis-angle vector
    a = np.linalg.norm(au)  # The rotation angle
    u = au / a  # The rotation axis (unit vector)
    return a, u


def rad2deg(rad):
    """
    Convert radians to degrees.
    """
    return rad * 180 / np.pi


def R2q(R):
    """
    Convert a rotation matrix to a quaternion using the specific cases provided.

    Parameters:
    R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    q (numpy.ndarray): A quaternion represented as a 4x1 numpy array.
    """
    case1 = (R[1, 1] > -R[2, 2]) and (R[0, 0] > -R[1, 1]) and (R[0, 0] > -R[2, 2])
    case2 = (R[1, 1] < -R[2, 2]) and (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2])
    case3 = (R[1, 1] > R[2, 2]) and (R[0, 0] < R[1, 1]) and (R[0, 0] < -R[2, 2])
    case4 = (R[1, 1] < R[2, 2]) and (R[0, 0] < -R[1, 1]) and (R[0, 0] < R[2, 2])

    whichcase = np.argmax([case1, case2, case3, case4]) + 1

    if whichcase == 0:
        a, u = R2au(R)
        q = np.array(
            [
                np.cos(a / 2),
                u[0] * np.sin(a / 2),
                u[1] * np.sin(a / 2),
                u[2] * np.sin(a / 2),
            ]
        )
    else:
        if whichcase == 1:
            den = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            q = 0.5 * np.array(
                [
                    den,
                    (R[2, 1] - R[1, 2]) / den,
                    (R[0, 2] - R[2, 0]) / den,
                    (R[1, 0] - R[0, 1]) / den,
                ]
            )
        elif whichcase == 2:
            den = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q = 0.5 * np.array(
                [
                    (R[2, 1] - R[1, 2]) / den,
                    den,
                    (R[1, 0] + R[0, 1]) / den,
                    (R[0, 2] + R[2, 0]) / den,
                ]
            )
        elif whichcase == 3:
            den = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            q = 0.5 * np.array(
                [
                    (R[0, 2] - R[2, 0]) / den,
                    (R[1, 0] + R[0, 1]) / den,
                    den,
                    (R[1, 2] + R[2, 1]) / den,
                ]
            )
        elif whichcase == 4:
            den = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            q = 0.5 * np.array(
                [
                    (R[1, 0] - R[0, 1]) / den,
                    (R[0, 2] + R[2, 0]) / den,
                    (R[2, 1] + R[1, 2]) / den,
                    den,
                ]
            )

    return q


def unit(v):
    """
    Compute the unit vector of v.

    Parameters:
    v -- a vector

    Returns:
    u -- the unit vector of v
    """
    v = np.asarray(v, dtype=float).reshape(
        3,
    )
    return v / np.linalg.norm(v)


def unitcross(v1, v2):
    """
    Compute the unit vector perpendicular to both v1 and v2 using their cross product.

    Parameters:
    v1 -- first vector
    v2 -- second vector

    Returns:
    u3 -- the unit vector perpendicular to both v1 and v2
    """
    v3 = np.cross(v1, v2)
    if np.linalg.norm(v3) > np.finfo(float).eps:
        u3 = v3 / np.linalg.norm(v3)
        return u3
    else:
        # If the vectors are collinear, the cross product will be zero
        # Return None or handle the case as needed
        return None


def v2q(v):
    """
    Convert a 3D vector v into a quaternion q.

    Parameters:
    v -- a 3D vector

    Returns:
    q -- the corresponding quaternion
    """
    a = np.linalg.norm(v)  # Calculate the norm of the vector
    u = v / a  # Normalize the vector

    q = np.array(
        [
            np.cos(a / 2),  # Quaternion scalar part
            u[0] * np.sin(a / 2),  # Quaternion vector part
            u[1] * np.sin(a / 2),
            u[2] * np.sin(a / 2),
        ]
    )

    return q


def vee(X):
    """
    Extract the vector from a skew-symmetric matrix X.

    Parameters:
    X -- a 3x3 skew-symmetric matrix

    Returns:
    x_vee -- the vector corresponding to the skew-symmetric matrix
    """
    x_vee = np.array([X[2, 1], X[0, 2], X[1, 0]])  # X(3,2)  # X(1,3)  # X(2,1)

    return x_vee

def so3_log_vec(R):
    R = np.asarray(R)
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    theta = math.acos(c)
    if theta < 1e-7:
        return 0.5 * vee(R - R.T)
    return (theta / (2.0 * math.sin(theta))) * vee(R - R.T)

def rodrigues(u, k, ang):
    k = k / (np.linalg.norm(k) + 1e-12)
    u_par = (u @ k) * k
    u_perp = u - u_par
    if np.linalg.norm(u_perp) < 1e-12:
        k2 = np.array([1.0, 0.0, 0.0])
        if abs(k @ k2) > 0.9:
            k2 = np.array([0.0, 1.0, 0.0])
        u_perp = u - (u @ k2) * k2
    u_perp = u_perp / (np.linalg.norm(u_perp) + 1e-12)
    axis = np.cross(u, k)
    if np.linalg.norm(axis) < 1e-12:
        axis = u_perp
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R @ u

def _seed_right(f):
    f = f / (np.linalg.norm(f) + 1e-12)
    a = np.array([1., 0., 0.]) if abs(f[0]) < 0.9 else np.array([0., 1., 0.])
    x = np.cross(a, f)
    return x / (np.linalg.norm(x) + 1e-12)


def _lookat_continuous_RGS(fwd_G, world_up_G, x_prev=None,
                           cos_thr=0.9995, sin_thr=0.03, eps=1e-8):
    f = fwd_G / (np.linalg.norm(fwd_G) + 1e-12)

    use_prev = (x_prev is not None)
    x_proj = None
    n_proj = 0.0

    if use_prev:
        x_proj = x_prev - f * (f @ x_prev)
        n_proj = np.linalg.norm(x_proj)

    x_up = np.cross(world_up_G, f)
    n_up = np.linalg.norm(x_up)
    near_sing = (abs(f @ world_up_G) > cos_thr) or (n_up < sin_thr)

    if use_prev and n_proj > eps:
        x = x_proj / n_proj
    elif n_up > eps:
        x = x_up / n_up
    else:
        x = _seed_right(f)

    y = np.cross(f, x)
    R_GS = np.column_stack((x, y, f))
    return R_GS, x


def _quat_hemi_continuous(q, q_prev):
    if q_prev is None:
        return q
    return q if float(np.dot(q, q_prev)) >= 0.0 else -q


def enforce_quat_series_continuity(q_series):
    if q_series.shape[0] == 0:
        return q_series
    q_prev = q_series[0]
    q_series[0] = q_prev / (np.linalg.norm(q_prev) + 1e-12)
    for k in range(1, q_series.shape[0]):
        qk = q_series[k]
        qk = qk / (np.linalg.norm(qk) + 1e-12)
        if float(np.dot(qk, q_prev)) < 0.0:
            qk = -qk
        q_series[k] = qk
        q_prev = qk
    return q_series

def calcCandS(z):
    """
    Calculate C and S based on the input z.

    Parameters:
    z -- a scalar input value

    Returns:
    C -- calculated value of C
    S -- calculated value of S
    """
    if abs(z) < 1e-3:
        C = 1 / 2 - z / 24 + z**2 / 720
        S = 1 / 6 - z / 120 + z**2 / 5040
    elif z < 0:
        g = np.sqrt(-z)
        C = (1 - np.cosh(g)) / z
        S = (np.sinh(g) - g) / g**3
    else:
        g = np.sqrt(z)
        C = (1 - np.cos(g)) / z
        S = (g - np.sin(g)) / g**3

    return C, S


def sph2cart(az, el, r):
    """
    Convert spherical coordinates (azimuth, elevation, radius) to Cartesian coordinates (x, y, z).

    Parameters:
    az (float): Azimuth angle in radians, the angle from the x-axis in the xy-plane.
    el (float): Elevation angle in radians, the angle from the xy-plane.
    r (float): Radial distance from the origin to the point.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    x = r * np.cos(az) * np.cos(el)
    y = r * np.sin(az) * np.cos(el)
    z = r * np.sin(el)
    return x, y, z


def solveUniVarKepler(s0, mu, tgoal):
    """
    Convert spherical coordinates (azimuth, elevation, radius) to Cartesian coordinates (x, y, z).

    Parameters:
    az (float): Azimuth angle in radians, the angle from the x-axis in the xy-plane.
    el (float): Elevation angle in radians, the angle from the xy-plane.
    r (float): Radial distance from the origin to the point.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    r0 = s0[0:3]
    v0 = s0[3:6]

    nv0 = np.linalg.norm(v0)
    nr0 = np.linalg.norm(r0)
    smu = np.sqrt(mu)
    r0_dot_v0 = np.dot(r0, v0)
    a = -mu / (nv0**2 - 2 * mu / nr0)

    # Initialize x variable
    t0 = 0
    x = smu * (tgoal - t0) / a

    while True:
        # Compute z variable
        z = x**2 / a

        C, S = calcCandS(z)

        tn = 1 / smu * (r0_dot_v0 / smu * x**2 * C + (1 - nr0 / a) * x**3 * S + nr0 * x)

        if abs(tgoal - tn) / abs(tgoal) < 1e-8:
            break

        dtdxn = (
            1 / smu * (x**2 * C + r0_dot_v0 / smu * x * (1 - z * S) + nr0 * (1 - z * C))
        )

        x = x + (tgoal - tn) / dtdxn

    z = x**2 / a

    C, S = calcCandS(z)

    f = 1 - x**2 / nr0 * C
    g = tn - x**3 / smu * S

    r = f * r0 + g * v0
    nr = np.linalg.norm(r)

    fdot = smu / (nr0 * nr) * x * (z * S - 1)
    gdot = 1 - x**2 / nr * C

    v = fdot * r0 + gdot * v0

    s = np.hstack([r, v])

    return s


def optMeasFunc(r):
    """
    Calculate the measurement function and Jacobian matrix for a given position vector r.
    Converts Cartesian coordinates to angular measurements and computes partial derivatives.

    Parameters:
    r (array): Position vector [x, y, z].

    Returns:
    tuple: Measurement vector [theta1, theta2] and the Jacobian matrix H.
    """
    x = r[0]
    y = r[1]
    z = r[2]
    nr = np.linalg.norm(r)

    theta1 = np.arctan2(y, x)
    theta2 = np.arcsin(z / nr)

    h = np.array([theta1, theta2])

    dt1_dx = -y / (x**2 + y**2)
    dt1_dy = x / (x**2 + y**2)

    dt2_dx = -x * z / (nr**2 * np.sqrt(x**2 + y**2))
    dt2_dy = -y * z / (nr**2 * np.sqrt(x**2 + y**2))
    dt2_dz = np.sqrt(x**2 + y**2) / nr**2

    H = np.array([[dt1_dx, dt1_dy, 0], [dt2_dx, dt2_dy, dt2_dz]])

    return h, H


def oe2cart(oe, muCB):
    """
    Convert orbital elements to Cartesian position and velocity vectors.

    Parameters:
    oe (array): Orbital elements [a, e, inc, RAAN, w, nu], where:
                a is the semi-major axis,
                e is the eccentricity,
                inc is the inclination,
                RAAN is the right ascension of ascending node,
                w is the argument of periapsis,
                nu is the true anomaly.
    muCB (float): Gravitational parameter of the central body.

    Returns:
    tuple: Cartesian position vector r_I and velocity vector dr_I in the inertial frame.
    """
    a = oe[0]
    e = oe[1]
    inc = oe[2]
    RAAN = oe[3]
    w = oe[4]
    nu = oe[5]

    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))

    r_P = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0])
    dr_P = np.sqrt(muCB / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    R_IP = au2R(RAAN, [0, 0, 1]) @ au2R(inc, [1, 0, 0]) @ au2R(w, [0, 0, 1])

    r_I = np.dot(R_IP, r_P)
    dr_I = R_IP @ dr_P

    return r_I, dr_I


def keplerian_dyn(t, s, mu):
    """
    Compute the time derivatives of position and velocity for a spacecraft in Keplerian orbit.

    Parameters:
    t (float): Time (not used in the calculation but required by ODE solvers).
    s (array): State vector [r, v], where r is the position and v is the velocity.
    mu (float): Gravitational parameter of the central body.

    Returns:
    array: Time derivatives of the state vector [rdot, vdot].
    """
    r = s[0:3]
    v = s[3:6]

    nr = np.linalg.norm(r)

    rdot = v
    vdot = -mu / nr**3 * r

    sdot = np.array[(rdot, vdot)]

    return sdot


def cart2sph(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, radius).

    Parameters:
    x (float): x-coordinate.
    y (float): y-coordinate.
    z (float): z-coordinate.

    Returns:
    tuple: Spherical coordinates (az, el, r), where:
           az is the azimuth angle in radians,
           el is the elevation angle in radians,
           r is the radial distance from the origin.
    """
    az = np.atan2(y, x)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return az, el, r

def _vecI_to_azel(v_I):
    """
    converts a vector in inertial frame to azimuth and elevation angles.
    
    :param v_I: Description
    """
    x, y, z = v_I
    r = np.linalg.norm(v_I) + 1e-12
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))
    az = np.arctan2(y, x) % (2.0 * np.pi)
    return az, el

def cart2oe(r_vec, v_vec, muCB):
    """
    Convert Cartesian coordinates to orbital elements.

    Parameters:
    r_vec -- position vector (numpy array of shape (3,))
    v_vec -- velocity vector (numpy array of shape (3,))
    muCB -- gravitational parameter (scalar)

    Returns:
    oe -- orbital elements [semi-major axis, eccentricity, inclination, RAAN, argument of periapsis, true anomaly]
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Normal vector to the orbital plane
    n_vec = np.array([-h_vec[1], h_vec[0], 0])
    n = np.linalg.norm(n_vec)

    # Semi-latus rectum
    p = h**2 / muCB

    # Eccentricity vector
    e_vec = (1 / muCB) * ((v**2 - muCB / r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    if e != 1:  # e not equal to 1
        a = p / (1 - e**2)
    else:
        a = np.inf

    # Inclination
    inc = np.arccos(h_vec[2] / h)
    raan = np.arccos(n_vec[0] / n)
    w = np.arccos(np.dot(n_vec, e_vec) / (n * e))
    nu0 = np.arccos(np.dot(e_vec, r_vec) / (e * r))

    if n_vec[1] < 0:
        raan = 2 * np.pi - raan

    if e_vec[2] < 0:
        w = 2 * np.pi - w

    if np.dot(r_vec, v_vec) < 0:
        nu0 = 2 * np.pi - nu0

    oe = np.array([a, e, inc, raan, w, nu0])

    return oe


def calcSTT(r0, v0, mu, t):
    """
    Calculate the nominal state and the state transition matrix (Phi and Phi2).

    Parameters:
    r0 -- initial position vector
    v0 -- initial velocity vector
    mu -- gravitational parameter
    t -- time

    Returns:
    r_nom -- nominal position vector
    v_nom -- nominal velocity vector
    Phi -- state transition matrix : Jacobian
    Phi2 -- second-order state transition matrix : Hessian
    """
    s0 = np.hstack([r0, v0])

    # Calculate nominal case
    s_nom = solveUniVarKepler(s0, mu, t)

    hr = 1e-5 * np.linalg.norm(r0)
    hv = 1e-5 * np.linalg.norm(v0)
    ei = np.diag([hr, hr, hr, hv, hv, hv])

    Phi = np.zeros((6, 6))
    for i in range(6):
        ds = ei[:, i]
        s_ds = solveUniVarKepler(s0 + ds, mu, t)
        Phi[:, i] = (s_ds - s_nom) / ei[i, i]

    Phi2 = np.zeros((6, 6, 6))
    for i in range(6):
        dsi = ei[:, i]
        s_dsi = solveUniVarKepler(s0 + dsi, mu, t)
        for j in range(6):
            dsj = ei[:, j]
            s_dsi_dsj = solveUniVarKepler(s0 + dsi + dsj, mu, t)
            s_dsj = solveUniVarKepler(s0 + dsj, mu, t)
            Phi2[:, i, j] = (s_dsi_dsj - s_dsi - s_dsj + s_nom) / (ei[i, i] * ei[j, j])

    r_nom = s_nom[0:3]
    v_nom = s_nom[3:6]

    return r_nom, v_nom, Phi, Phi2


def parameterSetting(d_ref):
    # define reference orbit
    R_E = 6378e3  # m
    a = R_E + d_ref
    return a


def centerPointingAttitude(r, v, up_fallback=np.array([0, 1, 0]), v_collinear_deg=15.0):
    """
    Stable 'look-at' camera pointing: +Z toward -r (target at the image center).
    Uses velocity projected on image plane to define 'up'; falls back to fixed up if needed.
    Returns R_GC: columns are camera x,y,z axes expressed in the G frame.
    """
    r_vec = unit(
        -np.asarray(r, dtype=float).reshape(
            3,
        )
    )  # camera +Z
    v_vec = np.asarray(v, dtype=float).reshape(
        3,
    )

    # Project velocity into image plane
    v_perp = v_vec - np.dot(v_vec, r_vec) * r_vec
    use_fallback = np.linalg.norm(v_perp) < 1e-6
    if not use_fallback and np.linalg.norm(v_vec) > 0:
        # also treat near-collinearity as degenerate
        col = abs(np.dot(unit(v_vec), r_vec)) > np.cos(np.deg2rad(v_collinear_deg))
        use_fallback = use_fallback or col

    if use_fallback:
        y_dir = unitcross(r_vec, up_fallback)  # any consistent world-up
    else:
        y_dir = unit(v_perp)

    x_dir = unitcross(y_dir, r_vec)  # right
    y_dir = unitcross(r_vec, x_dir)  # re-orthogonalize

    R_GC = np.column_stack((x_dir, y_dir, r_vec))
    return R_GC


def centerPointingAttitudeWithBoresight(
    r,
    v,
    boresight_angles=(0.0, 0.0, 0.0),
    up_fallback=np.array([0, 1, 0]),
    v_collinear_deg=15.0,
):
    """
    Create center-pointing attitude with boresight offset for SLAM applications.

    Parameters:
    r -- position vector relative to target
    v -- velocity vector relative to target
    boresight_angles -- [roll, pitch, yaw] offsets in radians from center-pointing

    Returns:
    R_GC -- rotation matrix from target frame to camera frame with boresight offset
    """
    # First create the standard center-pointing attitude
    R_GC_nominal = centerPointingAttitude(
        r, v, up_fallback=up_fallback, v_collinear_deg=v_collinear_deg
    )

    # Create boresight rotation matrix
    # Apply rotations in order: roll (about x), pitch (about y), yaw (about z)
    roll, pitch, yaw = boresight_angles
    R_bore = au2R(yaw, [0, 0, 1]) @ au2R(pitch, [0, 1, 0]) @ au2R(roll, [1, 0, 0])

    # Apply boresight offset to the nominal center-pointing attitude
    return R_GC_nominal @ R_bore


def generateBoresightAngles(num_agents, offset_magnitude=0.05):
    """
    Generate different boresight angles for each agent to avoid degeneracy.

    Parameters:
    num_agents -- number of agents
    offset_magnitude -- magnitude of offset in radians (default ~3 degrees)

    Returns:
    boresight_offsets -- array of [roll, pitch, yaw] for each agent
    """
    boresight_offsets = np.zeros((num_agents, 3))

    if num_agents == 1:
        # Single agent: small pitch offset for parallax
        boresight_offsets[0] = [0, offset_magnitude, 0]
    elif num_agents == 2:
        # Two agents: opposite pitch offsets
        boresight_offsets[0] = [0, offset_magnitude, 0]
        boresight_offsets[1] = [0, -offset_magnitude, 0]
    else:
        # Multiple agents: distribute around a circle
        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents
            boresight_offsets[i] = [
                offset_magnitude * np.cos(angle),  # roll
                offset_magnitude * np.sin(angle),  # pitch
                0,  # yaw
            ]

    return boresight_offsets


def timeVaryingBoresight(t, base_offset, frequency=0.0, amplitude=0.01):
    """
    Create time-varying boresight offset for additional motion diversity.

    Parameters:
    t -- current time
    base_offset -- base boresight offset [roll, pitch, yaw]
    frequency -- oscillation frequency in Hz
    amplitude -- oscillation amplitude in radians

    Returns:
    offset -- time-varying boresight offset
    """
    omega = 2 * np.pi * frequency
    offset = base_offset.copy()
    offset[0] += amplitude * np.sin(omega * t)  # roll variation
    offset[1] += amplitude * np.cos(omega * t)  # pitch variation

    return offset


# Alternative: Simple fixed offset function
def addBoresightOffset(R_GC_nominal, offset_angles):
    """
    Add boresight offset to existing attitude matrix.

    Parameters:
    R_GC_nominal -- nominal center-pointing attitude matrix
    offset_angles -- [roll, pitch, yaw] offset angles in radians

    Returns:
    R_GC -- attitude matrix with boresight offset applied
    """
    R_offset = (
        au2R(offset_angles[2], np.array([0, 0, 1]))
        @ au2R(offset_angles[1], np.array([0, 1, 0]))
        @ au2R(offset_angles[0], np.array([1, 0, 0]))
    )

    return R_GC_nominal @ R_offset


def createHillFrame(r_X, v_X):
    # Create Hill Frame
    er = r_X / np.linalg.norm(r_X)
    ev = v_X / np.linalg.norm(v_X)
    eh = unitcross(er, ev)
    eh = eh / np.linalg.norm(eh)
    etheta = unitcross(eh, er)
    etheta = etheta / np.linalg.norm(etheta)
    R_XH = np.array([er, etheta, eh]).T
    return R_XH


def calcInitCondChaser(
    xdot_0, ydot_0, zdot_0, x_0, y_0, z_0, s_t_0, R_IH_0, omega_HI_I_0
):
    r_GO_I_0 = s_t_0[0:3]  # Target position
    v_GO_I_0 = s_t_0[3:6]  # Target velocity
    r_CG_H_0 = np.hstack([x_0, y_0, z_0])
    dr_CG_H_0 = np.hstack([xdot_0, ydot_0, zdot_0])
    r_CO_I_0 = np.dot(R_IH_0, r_CG_H_0) + r_GO_I_0
    v_CO_I_0 = (
        np.dot(R_IH_0, dr_CG_H_0)
        + np.cross(omega_HI_I_0, np.dot(R_IH_0, r_CG_H_0))
        + v_GO_I_0
    )
    return r_CO_I_0, v_CO_I_0


def propagate_orbit(mu_ref, s_I_0, tspan):
    s_I = np.zeros((len(tspan), 6))
    s_I[0] = s_I_0  # Initial state
    Phi = np.zeros((len(tspan), 6, 6))
    Phi[0] = np.eye(6)  # Initial state transition matrix
    Phi2 = np.zeros((len(tspan), 6, 6, 6))

    for i in range(1, len(tspan)):
        r, v, Phi[i], Phi2[i] = calcSTT(s_I_0[:3], s_I_0[3:], mu_ref, tspan[i])
        s_I[i] = np.hstack([r, v])  # from the initial condition, get the state(r,v)

    return s_I


# Defined by Newton - Euler equations
# Define the system of equations

def calcOmegaDotNE(omega, J, invJ):
    return -np.dot(invJ, np.dot(sk(omega), np.dot(J, omega)))


def solve_ne_equation(nbSteps, delta_t, omega_0, R_0, J):
    s = 5  # CG approach
    a = np.array(
        [
            [0, 0, 0, 0, 0],
            [0.8177227988124852, 0, 0, 0, 0],
            [0.3199876375476427, 0.0659864263556022, 0, 0, 0],
            [0.9214417194464946, 0.4997857776773573, -1.0969984448371582, 0, 0],
            [
                0.3552358559023322,
                0.2390958372307326,
                1.3918565724203246,
                -1.1092979392113565,
                0,
            ],
        ]
    )
    b = np.array(
        [
            0.1370831520630755,
            -0.0183698531564020,
            0.7397813985370780,
            -0.1907142565505889,
            0.3322195591068374,
        ]
    )

    omega = np.zeros((nbSteps, 3))
    omega_step = np.zeros((s, 3))
    R = [np.eye(3)] * nbSteps
    k = np.zeros((nbSteps, 3))
    k_step = np.zeros((s, 3))

    omega[0] = omega_0
    R[0] = R_0
    invJ = np.linalg.inv(J)
    for k in range(0, nbSteps - 1):
        for i in range(0, s):
            # calculate the intermediate steps
            omega_step[i] = omega[k] + sum(
                a[i][j] * delta_t * k_step[j] for j in range(0, i)
            )
            k_step[i] = calcOmegaDotNE(omega_step[i], J, invJ)

        omega[k + 1] = omega[k] + delta_t * sum(b[i] * k_step[i] for i in range(0, s))
        R_temp = R[k]
        for i in range(0, s):
            R_temp = R_temp @ expm(sk(delta_t * b[i] * omega_step[i]))
        R[k + 1] = R_temp

    return omega, R


# Helper functions for the non orbital trajectory generator
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

def axes_to_quaternion(x_axis, y_axis, z_axis):
    # Form the rotation matrix from the axes
    R = np.column_stack((x_axis, y_axis, z_axis))
    # Convert to quaternion
    quaternion =R2q(R)
    return quaternion

def cartesian_to_spherical(x, y, z):
    # Calculate azimuth angle (phi)
    azimuth = np.arctan2(y, x)
    
    # Calculate elevation angle (theta)
    elevation = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    
    return azimuth, elevation