"""Fossen kinematics primitives for 6-DOF rigid-body marine vehicles.

Ported from xplorer_mini_python_utils/kinematics.py; ROS bindings stripped.
Conventions follow Fossen, "Handbook of Marine Craft Hydrodynamics and Motion
Control" (2011): nu = [u, v, w, p, q, r], eta = [x, y, z, phi, theta, psi].
"""

import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix from a 3-vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def m2c(M: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Coriolis-centripetal matrix from the inertia matrix and body velocity.

    Fossen formulation; M is 6x6, nu is 6-vector.
    """
    M11, M21 = M[:3, :3], M[3:, :3]
    M12, M22 = M[:3, 3:], M[3:, 3:]
    nu1, nu2 = nu[:3], nu[3:]

    C_rb = np.zeros((6, 6))
    C_rb[:3, 3:] = -skew_symmetric(M11 @ nu1 + M12 @ nu2)
    C_rb[3:, :3] = -skew_symmetric(M11 @ nu1 + M12 @ nu2)
    C_rb[3:, 3:] = -skew_symmetric(M21 @ nu1 + M22 @ nu2)
    return C_rb


def gvect(W: float, B: float, theta: float, phi: float,
          r_g: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """6-vector of restoring forces (gravity + buoyancy)."""
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_th, c_th = np.sin(theta), np.cos(theta)
    return np.array([
        (W - B) * s_th,
        -(W - B) * c_th * s_phi,
        -(W - B) * c_th * c_phi,
        -(r_g[1] * W - r_b[1] * B) * c_th * c_phi + (r_g[2] * W - r_b[2] * B) * c_th * s_phi,
        (r_g[2] * W - r_b[2] * B) * s_th + (r_g[0] * W - r_b[0] * B) * c_th * c_phi,
        -(r_g[0] * W - r_b[0] * B) * c_th * s_phi - (r_g[1] * W - r_b[1] * B) * s_th,
    ])


def eulerang(phi: float, theta: float, psi: float):
    """6x6 Euler-angle Jacobian J(eta) and its 3x3 sub-blocks (Fossen eq. 2.40).

    Returns (J, R_zyx, T_zyx) where J = blockdiag(R_zyx, T_zyx).
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    if cth == 0:
        raise ValueError("Euler angle singularity at theta = +/- pi/2")

    R_zyx = np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi,  spsi * sphi + cpsi * cphi * sth],
        [spsi * cth,  cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth,        cth * sphi,                       cth * cphi],
    ])
    T_zyx = np.array([
        [1, sphi * sth / cth,  cphi * sth / cth],
        [0, cphi,              -sphi],
        [0, sphi / cth,         cphi / cth],
    ])
    J = np.block([[R_zyx, np.zeros((3, 3))], [np.zeros((3, 3)), T_zyx]])
    return J, R_zyx, T_zyx