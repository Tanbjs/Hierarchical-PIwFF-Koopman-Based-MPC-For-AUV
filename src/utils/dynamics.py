"""6-DOF nonlinear AUV dynamics in NED, plus RK4 integrator and YAML loader.

Ported from xplorer_mini_sim_ws/.../notebook/simulation.py. The dynamics
follow Fossen's standard form:

    eta_dot = J(eta) @ nu
    M @ nu_dot = tau - C(nu) @ nu - D(nu) @ nu - g(eta)

where M includes rigid-body and added-mass terms, D is linear + quadratic
drag, g is the restoring force, and tau is the body-frame control wrench.
"""

from pathlib import Path

import numpy as np
import yaml

from .kinematics import eulerang, gvect, m2c


class UUVParams:
    """AUV physical parameters loaded from params/xplorer_mini.yaml.

    Expected layout: top-level `auv_params:` mapping containing weight,
    buoyancy, mass, r_g, r_b, rigid_body_mass, added_mass, linear_drag,
    nonlinear_drag.
    """

    def __init__(self, yaml_data: dict):
        raw = yaml_data["auv_params"]
        self.W = float(raw["weight"])
        self.B = float(raw["buoyancy"])
        self.m = float(raw["mass"])
        self.r_g = np.array(raw["r_g"], dtype=float)
        self.r_b = np.array(raw["r_b"], dtype=float)
        self.M_total = (np.array(raw["rigid_body_mass"]).reshape(6, 6)
                        + np.array(raw["added_mass"]).reshape(6, 6))
        self.D_l = np.array(raw["linear_drag"]).reshape(6, 6)
        self.D_nl = np.array(raw["nonlinear_drag"]).reshape(6, 6)


def load_params(param_file: str | Path) -> UUVParams:
    with open(param_file, "r") as f:
        return UUVParams(yaml.safe_load(f))


def get_effective_buoyancy(z: float, B_max: float, robot_height: float = 0.3) -> float:
    """Linear partial-submergence model in NED (z positive downward, z=0 at surface)."""
    top_z = z - 0.5 * robot_height        # lower z value (above the vehicle)
    bottom_z = z + 0.5 * robot_height     # higher z value (below the vehicle)
    if top_z >= 0.0:
        return B_max                       # fully submerged
    if bottom_z <= 0.0:
        return 0.0                          # fully emerged
    return B_max * bottom_z / robot_height  # partial submergence


def f_dyn(x: np.ndarray, u: np.ndarray, params: UUVParams) -> np.ndarray:
    """Continuous-time 6-DOF rigid-body dynamics. x = [eta, nu] (12), u = tau (6).

    Returns x_dot in NED.
    """
    eta, nu = x[0:6], x[6:12]
    B_eff = get_effective_buoyancy(eta[2], params.B, robot_height=0.3)
    J_eta, _, _ = eulerang(eta[3], eta[4], eta[5])
    sum_forces = (
        u
        - m2c(params.M_total, nu) @ nu
        - (params.D_l + params.D_nl @ np.diag(np.abs(nu))) @ nu
        - gvect(params.W, B_eff, eta[4], eta[3], params.r_g, params.r_b)
    )
    return np.concatenate(((J_eta @ nu).flatten(),
                           np.linalg.solve(params.M_total, sum_forces).flatten()))


def rk4_step(sys, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Classical RK4 step for x_dot = sys(x, u)."""
    k1 = sys(x, u)
    k2 = sys(x + 0.5 * dt * k1, u)
    k3 = sys(x + 0.5 * dt * k2, u)
    k4 = sys(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)