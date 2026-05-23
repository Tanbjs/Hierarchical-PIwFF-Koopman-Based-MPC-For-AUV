"""Shared utilities: AUV kinematics, 6-DOF dynamics, IEEE Access plotting."""

from .dynamics import UUVParams, f_dyn, get_effective_buoyancy, load_params, rk4_step
from .kinematics import eulerang, gvect, m2c, skew_symmetric
from .path_gen import figure8_path
from .plot import (
    COL1, COL2, GRID_FIGSIZE,
    COLOR_DMDC, COLOR_EDMDC, COLOR_NONLINEAR, PHASE_LINE_COLOR,
    LABELS_ETA_ANG, LABELS_ETA_LIN, LABELS_NU_ANG, LABELS_NU_LIN,
    LABELS_TAU_F, LABELS_TAU_T,
    add_panel_labels, apply_ieee_style, format_figure,
)

__all__ = [
    # dynamics
    "UUVParams", "f_dyn", "get_effective_buoyancy", "load_params", "rk4_step",
    # kinematics
    "eulerang", "gvect", "m2c", "skew_symmetric",
    # reference trajectories
    "figure8_path",
    # plot
    "COL1", "COL2", "GRID_FIGSIZE",
    "COLOR_DMDC", "COLOR_EDMDC", "COLOR_NONLINEAR", "PHASE_LINE_COLOR",
    "LABELS_ETA_ANG", "LABELS_ETA_LIN", "LABELS_NU_ANG", "LABELS_NU_LIN",
    "LABELS_TAU_F", "LABELS_TAU_T",
    "add_panel_labels", "apply_ieee_style", "format_figure",
]
