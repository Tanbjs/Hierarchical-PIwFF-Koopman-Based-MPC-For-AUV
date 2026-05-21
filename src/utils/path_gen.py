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
        y_l = W * np.sin(2 * theta)

        x = center[0] + x_l * np.cos(alpha) - y_l * np.sin(alpha)
        y = center[1] + x_l * np.sin(alpha) + y_l * np.cos(alpha)
        z = start_pos[2] + (end_pos[2] - start_pos[2]) * s
        phi = start_orient[0] + (end_orient[0] - start_orient[0]) * s
        theta_pitch = start_orient[1] + (end_orient[1] - start_orient[1]) * s

        # Tangent yaw from instantaneous velocity
        dx_l = A * np.sin(theta)
        dy_l = 2 * W * np.cos(2 * theta)
        dx = dx_l * np.cos(alpha) - dy_l * np.sin(alpha)
        dy = dx_l * np.sin(alpha) + dy_l * np.cos(alpha)
        psi = np.arctan2(dy, dx) if np.linalg.norm([dx, dy]) > 1e-6 else alpha

        eta_d[i, :] = [x, y, z, phi, theta_pitch, psi]

    return eta_d, t_span


def zigzag_path(
    num_zigs: int,
    start_point,
    end_point,
    dt: float,
    tfinal: float,
    amplitude: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 6-DOF zig-zag trajectory between two poses.

    The vehicle travels along the straight line from `start_point` to
    `end_point` while sinusoidally weaving sideways with the given
    `amplitude` and `num_zigs` full periods. Yaw is set to the path
    tangent (so the vehicle always faces along its instantaneous velocity);
    roll and pitch interpolate linearly between endpoints.

    Args:
        num_zigs:    number of full sinusoidal periods over the trajectory
        start_point: 6-vector [x, y, z, roll, pitch, yaw] in NED
        end_point:   6-vector [x, y, z, roll, pitch, yaw] in NED
        dt:          sample interval [s]
        tfinal:      total trajectory duration [s]
        amplitude:   sideways deviation amplitude [m]

    Returns:
        eta_d:  (N, 6) array of reference poses
        t:      (N,)   time vector starting at 0
    """
    if tfinal <= 0:
        raise ValueError("tfinal must be positive.")
    if not (0 < dt <= tfinal):
        raise ValueError("dt must be positive and <= tfinal.")
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative.")
    if num_zigs <= 0:
        raise ValueError("num_zigs must be positive.")

    t = np.arange(0, tfinal + dt, dt)
    N = len(t)

    start_pos = np.array(start_point[:3], dtype=float)
    end_pos = np.array(end_point[:3], dtype=float)
    start_orient = np.array(start_point[3:], dtype=float)
    end_orient = np.array(end_point[3:], dtype=float)

    main_path_vector = end_pos - start_pos
    main_path_length = np.linalg.norm(main_path_vector)
    main_path_direction = (
        main_path_vector / main_path_length
        if main_path_length > 0
        else np.array([1.0, 0.0, 0.0])
    )

    # Perpendicular in the xy-plane; falls back to +y for a purely vertical path.
    perp_vector = np.array([-main_path_direction[1], main_path_direction[0], 0.0])
    norm_perp = np.linalg.norm(perp_vector)
    if norm_perp == 0:
        perp_vector = np.array([0.0, 1.0, 0.0])
    else:
        perp_vector /= norm_perp

    position = np.zeros((N, 3))
    orientation = np.zeros((N, 3))
    angular_freq = (2 * np.pi * num_zigs) / tfinal
    centerline_vel = main_path_vector / tfinal if tfinal > 0 else np.zeros(3)

    for i, ti in enumerate(t):
        s = ti / tfinal if tfinal > 0 else 0.0

        # Centerline + sinusoidal deviation
        position[i] = start_pos + s * main_path_vector + (
            amplitude * np.sin(angular_freq * ti)
        ) * perp_vector

        # Roll/pitch: linear interpolation
        orientation[i, 0] = (1 - s) * start_orient[0] + s * end_orient[0]
        orientation[i, 1] = (1 - s) * start_orient[1] + s * end_orient[1]

        # Yaw: instantaneous tangent direction
        deviation_vel = amplitude * angular_freq * np.cos(angular_freq * ti) * perp_vector
        tangent = centerline_vel + deviation_vel
        if np.linalg.norm(tangent[:2]) > 1e-6:
            orientation[i, 2] = np.arctan2(tangent[1], tangent[0])
        else:
            orientation[i, 2] = np.arctan2(main_path_direction[1], main_path_direction[0])

    return np.hstack([position, orientation]), t
