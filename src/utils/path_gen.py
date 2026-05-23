"""Reference trajectory generators for closed-loop simulation."""

from typing import Tuple

import numpy as np


def figure8_path(
    N: int,
    start_point,
    end_point,
    dt: float,
    tfinal: float,
    width_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 6-DOF figure-8 (Lemniscate of Gerono) trajectory.

    When `start_point == end_point`, traces N closed loops centred at the
    starting pose. Otherwise traces N+0.5 lobes from start to end.
    Yaw follows the tangent direction; roll/pitch interpolate linearly.

    Args:
        N:           number of figure-8 lobes (full rounds when closed)
        start_point: 6-vector [x, y, z, roll, pitch, yaw]
        end_point:   6-vector [x, y, z, roll, pitch, yaw]
        dt:          sample interval [s]
        tfinal:      total trajectory duration [s]
        width_ratio: lateral-to-longitudinal extent ratio (default 0.5)
    """
    if tfinal <= 0:
        raise ValueError("tfinal must be positive.")
    if not (0 < dt <= tfinal):
        raise ValueError("dt must be positive and <= tfinal.")

    t_span = np.arange(0, tfinal + dt, dt)
    eta_d = np.zeros((len(t_span), 6))

    start_pos = np.array(start_point[:3], dtype=float)
    end_pos = np.array(end_point[:3], dtype=float)
    start_orient = np.array(start_point[3:], dtype=float)
    end_orient = np.array(end_point[3:], dtype=float)

    vector = end_pos - start_pos
    dist = np.linalg.norm(vector[:2])

    if dist < 0.05:
        # Closed loop: start == end
        A = 3.0
        alpha = start_orient[2]
        W = A * width_ratio
        center = start_pos.copy()
        center[0] += A * np.cos(alpha)
        center[1] += A * np.sin(alpha)
        total_theta = 2 * np.pi * N
    else:
        center = (start_pos + end_pos) / 2.0
        A = dist / 2.0
        alpha = np.arctan2(vector[1], vector[0])
        W = A * width_ratio
        total_theta = np.pi + (2 * np.pi * N)

    for i, t in enumerate(t_span):
        s = t / tfinal if tfinal > 0 else 0.0
        theta = total_theta * s

        x_l = -A * np.cos(theta)
        y_l = -W * np.sin(2 * theta)

        x = center[0] + x_l * np.cos(alpha) - y_l * np.sin(alpha)
        y = center[1] + x_l * np.sin(alpha) + y_l * np.cos(alpha)
        z = start_pos[2] + (end_pos[2] - start_pos[2]) * s
        phi = start_orient[0] + (end_orient[0] - start_orient[0]) * s
        theta_pitch = start_orient[1] + (end_orient[1] - start_orient[1]) * s

        # Tangent yaw from instantaneous velocity
        dx_l = A * np.sin(theta)
        dy_l = -2 * W * np.cos(2 * theta)
        dx = dx_l * np.cos(alpha) - dy_l * np.sin(alpha)
        dy = dx_l * np.sin(alpha) + dy_l * np.cos(alpha)
        psi = np.arctan2(dy, dx) if np.linalg.norm([dx, dy]) > 1e-6 else alpha

        eta_d[i, :] = [x, y, z, phi, theta_pitch, psi]

    return eta_d, t_span