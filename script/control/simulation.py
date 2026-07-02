"""Closed-loop cascade simulation harness + 3-case study runner.

Library functions:
  - `run_cascade_simulation(...)`  -- roll out the controller against f_dyn.
  - `plot_response(...)`            -- IEEE Access PDF figures from history.
  - `build_reference()`             -- figure-8 + hover reference trajectory.

Run as a script (`python script/control/simulation.py`) to execute all three
study cases on the figure-8 reference (paper protocol) and write per-case
PDFs under result/control/<Case_Name>/:

  Case 1  Model accuracy without preview     -- DMDc vs eDMDc, nominal MPC
  Case 2  Preview impact, nominal MPC        -- DMDc, with vs without preview
  Case 3  Controller comparison              -- PID / Nominal-MPC / Offset-Free
                                                MPC, DMDc with preview
"""

import logging
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Try to register matplotlib's 3D projection. The 3D path figure (fig6) is
# optional -- the other 7 figures don't depend on it.
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_3D = True
except ImportError:
    _HAS_3D = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Silence fontTools subsetting logs spammed by matplotlib's PDF font embedding.
logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
sim_logger = logging.getLogger("Simulation")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sysid import FittedModel
from utils import (
    COL1, COL2, GRID_FIGSIZE,
    LABELS_ETA_ANG, LABELS_ETA_LIN, LABELS_NU_ANG, LABELS_NU_LIN,
    LABELS_TAU_F, LABELS_TAU_T,
    add_panel_labels, apply_ieee_style, f_dyn, figure8_path,
    format_figure, load_params, rk4_step,
)
from controllers.utils.controller import (
    create_position_controller, create_velocity_controller,
)
from controllers.utils.kinematic import cal_eta_err_with_ssa
from controllers.utils.virtual import (
    generate_virtual_reference_ff_pi, generate_virtual_reference_pid,
)


# =============================================================================
# Plotting (IEEE Access typography, 8 figures: state, error, control, 3D path,
# metrics tables). Self-contained; works as soon as histories are provided.
# =============================================================================

def plot_response(histories, labels, eta_ref_full, save_dir=None):
    # ========================================================================
    # IEEE Access Specifications
    # ========================================================================
    # Column widths (inches): 1-col = 3.5", 2-col = 7.16", max depth = 8.5"
    # Min resolution: 300 dpi (color/grayscale), 600 dpi (line art)
    # Accepted formats: EPS, PDF, PS, TIFF, PNG (PDF preferred for vector)
    apply_ieee_style()

    # ================= Data Sorting & Q1 Paper Colors =================
    def get_config(lbl):
        lbl_lower = str(lbl).lower()

        if 'pid' in lbl_lower: ctrl_p = 0
        elif 'nominal' in lbl_lower: ctrl_p = 1
        elif 'offset-free' in lbl_lower: ctrl_p = 2
        else: ctrl_p = 3

        if 'edmdc' in lbl_lower: mod_p = 1
        elif 'dmdc' in lbl_lower: mod_p = 0
        else: mod_p = 2

        prev_p = 1 if 'with preview' in lbl_lower else 0

        keys = (ctrl_p, mod_p, prev_p)

        if ctrl_p == 0:
            color = '#758A93'
        elif ctrl_p == 1 and mod_p == 0 and prev_p == 0:
            color = '#7FCBC4'
        elif ctrl_p == 1 and mod_p == 0 and prev_p == 1:
            color = '#3A78B8'
        elif ctrl_p == 1 and mod_p == 1 and prev_p == 0:
            color = '#8FB8DE'
        elif ctrl_p == 1 and mod_p == 1 and prev_p == 1:
            color = '#2A9D8F'
        elif ctrl_p == 2 and mod_p == 0 and prev_p == 0:
            color = '#CC6B5C'
        elif ctrl_p == 2 and mod_p == 0 and prev_p == 1:
            color = '#C5172E'
        elif ctrl_p == 2 and mod_p == 1 and prev_p == 0:
            color = '#D9A055'
        elif ctrl_p == 2 and mod_p == 1 and prev_p == 1:
            color = '#FF7444'
        else:
            color = '#B0B0B0'

        # PID slightly thicker as baseline; others uniform (IEEE thin lines)
        linewidth = 1.0 if ctrl_p == 0 else 0.9
        return keys, color, '-', linewidth

    # Sort by (Controller -> Model -> Preview)
    sorted_data = sorted(zip(labels, histories), key=lambda x: get_config(x[0])[0])
    labels = [x[0] for x in sorted_data]
    histories = [x[1] for x in sorted_data]

    styles = [get_config(lbl) for lbl in labels]
    colors     = [s[1] for s in styles]
    linewidths = [s[3] for s in styles]
    label_map  = {l: l for l in labels}
    # =================================================================

    t = np.array(histories[0]['t'])
    eta_ref = eta_ref_full[:len(t), :]

    # Optimization: convert to numpy arrays once
    for h in histories:
        if not isinstance(h['x'], np.ndarray):
            h['x'] = np.array(h['x'])
        if 'nu_cmd_b' in h and not isinstance(h['nu_cmd_b'], np.ndarray):
            h['nu_cmd_b'] = np.array(h['nu_cmd_b'])
        if 'tau' in h and not isinstance(h['tau'], np.ndarray):
            h['tau'] = np.array(h['tau'])

    # ========================================================================
    # Fig 1: Position Tracking (eta)
    # ========================================================================
    fig1, axs1 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)

    for i in range(3):
        ref_ang_rad = np.unwrap(eta_ref[:, i+3]) if i == 2 else eta_ref[:, i+3]
        axs1[i, 0].plot(t, eta_ref[:, i], 'k--', linewidth=1.0, label='Ref' if i == 0 else "")
        axs1[i, 1].plot(t, np.rad2deg(ref_ang_rad), 'k--', linewidth=1.0, label='Ref' if i == 0 else "")
        for idx, history in enumerate(histories):
            eta = history['x'][:, 0:6]
            state_ang_rad = np.unwrap(eta[:, i+3]) if i == 2 else eta[:, i+3]
            axs1[i, 0].plot(t, eta[:, i], color=colors[idx], linewidth=linewidths[idx],
                            label=label_map[labels[idx]] if i == 0 else "")
            axs1[i, 1].plot(t, np.rad2deg(state_ang_rad), color=colors[idx], linewidth=linewidths[idx],
                            label=label_map[labels[idx]] if i == 0 else "")
        axs1[i, 0].set_ylabel(LABELS_ETA_LIN[i]); axs1[i, 0].grid(True)
        axs1[i, 1].set_ylabel(LABELS_ETA_ANG[i]); axs1[i, 1].grid(True)

    axs1[2, 0].set_xlabel('Time (s)'); axs1[2, 1].set_xlabel('Time (s)')
    axs1[2, 0].set_ylim(1.9, 2.1)
    add_panel_labels(axs1.flat)
    format_figure(fig1, axs1, bottom_margin=0.15)

    # ========================================================================
    # Fig 2: Position Tracking Error (e_eta)
    # ========================================================================
    fig2, axs2 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)
    labels_eeta_lin = [r'$e_x$ (m)', r'$e_y$ (m)', r'$e_z$ (m)']
    labels_eeta_ang = [r'$e_\phi$ (°)', r'$e_\theta$ (°)', r'$e_\psi$ (°)']

    for i in range(3):
        for idx, history in enumerate(histories):
            eta = history['x'][:, 0:6]
            e_lin = eta_ref[:, i] - eta[:, i]
            e_ang_rad = (eta_ref[:, i+3] - eta[:, i+3] + np.pi) % (2 * np.pi) - np.pi
            axs2[i, 0].plot(t, e_lin, color=colors[idx], linewidth=linewidths[idx],
                            label=label_map[labels[idx]] if i == 0 else "")
            axs2[i, 1].plot(t, np.rad2deg(e_ang_rad), color=colors[idx], linewidth=linewidths[idx])
        axs2[i, 0].set_ylabel(labels_eeta_lin[i]); axs2[i, 0].grid(True)
        axs2[i, 1].set_ylabel(labels_eeta_ang[i]); axs2[i, 1].grid(True)

    axs2[2, 0].set_xlabel('Time (s)'); axs2[2, 1].set_xlabel('Time (s)')
    add_panel_labels(axs2.flat)
    format_figure(fig2, axs2, bottom_margin=0.13)

    # ========================================================================
    # Fig 3: Velocity Tracking (nu)
    # ========================================================================
    fig3, axs3 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)

    for i in range(3):
        for idx, history in enumerate(histories):
            nu = history['x'][:, 6:12]
            nu_cmd = history['nu_cmd_b']
            nu_deg = nu.copy()
            nu_deg[:, 3:6] = np.rad2deg(nu_deg[:, 3:6])
            nu_cmd_deg = nu_cmd.copy()
            nu_cmd_deg[:, 3:6] = np.rad2deg(nu_cmd_deg[:, 3:6])
            axs3[i, 0].plot(t, nu_cmd[:, i], '--', color=colors[idx], linewidth=0.7, alpha=0.5,
                            label=f'{label_map[labels[idx]]} (Cmd)' if i == 0 else "")
            axs3[i, 0].plot(t, nu[:, i], '-', color=colors[idx], linewidth=linewidths[idx],
                            label=label_map[labels[idx]] if i == 0 else "")
            axs3[i, 1].plot(t, nu_cmd_deg[:, i+3], '--', color=colors[idx], linewidth=0.7, alpha=0.4)
            axs3[i, 1].plot(t, nu_deg[:, i+3], '-', color=colors[idx], linewidth=linewidths[idx])
        axs3[i, 0].set_ylabel(LABELS_NU_LIN[i]); axs3[i, 0].grid(True)
        axs3[i, 1].set_ylabel(LABELS_NU_ANG[i]); axs3[i, 1].grid(True)

    axs3[2, 0].set_xlabel('Time (s)'); axs3[2, 1].set_xlabel('Time (s)')
    add_panel_labels(axs3.flat)
    format_figure(fig3, axs3, bottom_margin=0.20)

    # ========================================================================
    # Fig 4: Velocity Tracking Error (e_nu)
    # ========================================================================
    fig4, axs4 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)
    labels_enu_lin = [r'$e_u$ (m/s)', r'$e_v$ (m/s)', r'$e_w$ (m/s)']
    labels_enu_ang = [r'$e_p$ (°/s)', r'$e_q$ (°/s)', r'$e_r$ (°/s)']

    for i in range(3):
        for idx, history in enumerate(histories):
            nu = history['x'][:, 6:12]
            nu_cmd = history['nu_cmd_b']
            e_nu_lin = nu_cmd[:, i] - nu[:, i]
            e_nu_ang = np.rad2deg(nu_cmd[:, i+3] - nu[:, i+3])
            axs4[i, 0].plot(t, e_nu_lin, color=colors[idx], linewidth=linewidths[idx],
                            label=label_map[labels[idx]] if i == 0 else "")
            axs4[i, 1].plot(t, e_nu_ang, color=colors[idx], linewidth=linewidths[idx])
        axs4[i, 0].set_ylabel(labels_enu_lin[i]); axs4[i, 0].grid(True)
        axs4[i, 1].set_ylabel(labels_enu_ang[i]); axs4[i, 1].grid(True)

    axs4[2, 0].set_xlabel('Time (s)'); axs4[2, 1].set_xlabel('Time (s)')
    add_panel_labels(axs4.flat)
    format_figure(fig4, axs4, bottom_margin=0.15, ncol=len(labels))

    # ========================================================================
    # Fig 5: Generalized Torque (tau)
    # ========================================================================
    fig5, axs5 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)

    for i in range(3):
        for idx, history in enumerate(histories):
            tau = history['tau']
            axs5[i, 0].plot(t, tau[:, i], color=colors[idx], linewidth=linewidths[idx],
                            label=label_map[labels[idx]] if i == 0 else "")
            axs5[i, 1].plot(t, tau[:, i+3], color=colors[idx], linewidth=linewidths[idx])
        axs5[i, 0].set_ylabel(LABELS_TAU_F[i]); axs5[i, 0].grid(True)
        axs5[i, 1].set_ylabel(LABELS_TAU_T[i]); axs5[i, 1].grid(True)

    axs5[2, 0].set_xlabel('Time (s)'); axs5[2, 1].set_xlabel('Time (s)')
    add_panel_labels(axs5.flat)
    format_figure(fig5, axs5, bottom_margin=0.15, ncol=len(labels))

    # ========================================================================
    # Fig 6: 3D Path Tracking (single column, IEEE Access COL1 = 3.5")
    # Skipped when mpl_toolkits.mplot3d isn't importable (system mpl mismatch).
    # ========================================================================
    fig6 = None
    if _HAS_3D:
        fig6 = plt.figure(figsize=(COL1, 3.4))
        ax6 = fig6.add_subplot(111, projection='3d')
        ax6.plot(eta_ref[:, 0], eta_ref[:, 1], eta_ref[:, 2],
                 'k--', linewidth=1.0, label='Reference')
        for idx, history in enumerate(histories):
            eta = history['x'][:, 0:6]
            ax6.plot(eta[:, 0], eta[:, 1], eta[:, 2],
                     color=colors[idx], linewidth=linewidths[idx],
                     label=label_map[labels[idx]])

        for axis in [ax6.xaxis, ax6.yaxis, ax6.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor('lightgray')

        ax6.view_init(elev=25, azim=-55)
        ax6.set_box_aspect([1.6, 1.0, 0.7])
        ax6.locator_params(axis='x', nbins=4)
        ax6.locator_params(axis='y', nbins=3)
        ax6.locator_params(axis='z', nbins=4)
        ax6.set_xlabel(r'$x$ (m)', fontsize=8, labelpad=-2)
        ax6.set_ylabel(r'$y$ (m)', fontsize=8, labelpad=-2)
        ax6.set_zlabel('')
        fig6.text(0.92, 0.78, r'$z$ (m)', fontsize=8, ha='center', va='center')
        ax6.set_zlim(1.5, 2.5)
        ax6.invert_zaxis()
        ax6.tick_params(labelsize=6, pad=0)

        handles6, labels6 = ax6.get_legend_handles_labels()
        legend_ncol = min(2, len(handles6))
        fig6.legend(handles6, labels6, loc='lower center', ncol=legend_ncol,
                    bbox_to_anchor=(0.5, 0.10),
                    columnspacing=1.0, handletextpad=0.4,
                    frameon=True, edgecolor='black', framealpha=1.0)
        ax6.set_position([-0.05, 0.18, 1.05, 0.82])

    # ========================================================================
    # Metrics & Tables (Fig 7 & 8, double column)
    # ========================================================================
    # Paper reports RMSE and MaxAE only (MAE dropped).
    metrics_eta, metrics_nu = [], []
    for history in histories:
        eta = history['x'][:, 0:6]
        e_lin = eta_ref[:, :3] - eta[:, :3]
        e_ang_rad = (eta_ref[:, 3:6] - eta[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
        e_eta = np.hstack((e_lin, np.rad2deg(e_ang_rad)))
        metrics_eta.append({'rmse':  np.sqrt(np.mean(e_eta**2, axis=0)),
                            'maxae': np.max(np.abs(e_eta), axis=0)})
        nu = history['x'][:, 6:12]
        nu_cmd = history['nu_cmd_b']
        e_nu = np.hstack((nu_cmd[:, :3] - nu[:, :3], np.rad2deg(nu_cmd[:, 3:6] - nu[:, 3:6])))
        metrics_nu.append({'rmse':  np.sqrt(np.mean(e_nu**2, axis=0)),
                           'maxae': np.max(np.abs(e_nu), axis=0)})

    def create_metric_figure(title_main, cols, metrics_data):
        fig, axs = plt.subplots(2, 1, figsize=(COL2, 3.4))
        fig.suptitle(title_main, fontsize=8, fontweight='bold')
        metric_keys   = ['rmse', 'maxae']
        metric_titles = ['RMSE', 'MaxAE']
        colors_bg = ['#d9ead3', '#f4cccc']
        highlight_color, highlight_text_color = '#fff2cc', '#d62728'
        for i, key in enumerate(metric_keys):
            cell_text = [[f"{m[key][col_idx]:.4f}" for col_idx in range(6)] for m in metrics_data]
            tbl = axs[i].table(cellText=cell_text, rowLabels=labels, colLabels=cols,
                               loc='center', cellLoc='center',
                               rowColours=['#f2f2f2']*len(labels),
                               colColours=[colors_bg[i]]*6)
            for col_idx in range(6):
                col_vals = [m[key][col_idx] for m in metrics_data]
                min_idx  = np.argmin(col_vals)
                best_cell = tbl[min_idx + 1, col_idx]
                best_cell.set_facecolor(highlight_color)
                best_cell.get_text().set_weight('bold')
                best_cell.get_text().set_color(highlight_text_color)
            tbl.scale(1, 1.6); tbl.set_fontsize(7)
            axs[i].axis('off')
            axs[i].set_title(metric_titles[i], pad=4, fontsize=7, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.subplots_adjust(top=0.88, hspace=0.5)
        return fig

    fig7 = create_metric_figure(r'$\eta$ Tracking Error Metrics',
                                ['x (m)', 'y (m)', 'z (m)', r'$\phi$ (°)', r'$\theta$ (°)', r'$\psi$ (°)'],
                                metrics_eta)
    fig8 = create_metric_figure(r'$\nu$ Tracking Error Metrics',
                                ['u (m/s)', 'v (m/s)', 'w (m/s)', 'p (°/s)', 'q (°/s)', 'r (°/s)'],
                                metrics_nu)

    # ========================================================================
    # Export -- PDF (vector, IEEE-preferred) at 300 dpi
    # ========================================================================
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        figs  = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]
        names = ["1_position_response", "2_position_error", "3_velocity_response",
                 "4_velocity_error", "5_control_effort", "6_3d_path",
                 "7_table_position_metrics", "8_table_velocity_metrics"]
        for f, name in zip(figs, names):
            if f is None:
                continue  # 3D fig skipped when mpl_toolkits.mplot3d unavailable
            f.savefig(save_path / f"{name}.pdf", format='pdf',
                      bbox_inches='tight', pad_inches=0.05, dpi=300)
        plt.close('all')
    else:
        plt.show()


# =============================================================================
# Cascade closed-loop simulator (PIwFF outer + MPC inner, plant = f_dyn in NED)
# =============================================================================

def run_cascade_simulation(auv_params, pose_ctrl, vel_ctrl, eta_ref_full,
                           *,
                           N_horizon: int = 10,
                           pose_ctrl_type: str = 'pid',
                           use_ff: bool = False,
                           use_filter: bool = False,
                           alpha_ff: float = 1.0,
                           use_preview: bool = False,
                           vel_ctrl_type: str = 'mpc',
                           t_end: float = 120.0,
                           dt: float = 0.1,
                           record_mpc_time: bool = False) -> dict:
    """Roll out the cascade controller against the AUV plant.

    All controllers, references, and plant integration are in NED.
    Returns a history dict with keys: 't', 'x' (12-DOF [eta | nu]), 'tau',
    'nu_cmd_b', 'mpc_time'.
    """
    num_steps = int(t_end / dt)
    history = {'t': [], 'x': [], 'tau': [], 'nu_cmd_b': [], 'mpc_time': []}

    sys = lambda x, u: f_dyn(x, u, auv_params)
    is_mpc = 'mpc' in vel_ctrl_type

    x_curr = np.concatenate((eta_ref_full[0, :].flatten(), np.zeros(6)))
    eta_ref_prev = eta_ref_full[0]
    eta_dot_ref_filtered = np.zeros(6)

    history['t'].append(0.0)
    history['x'].append(x_curr.copy())
    history['tau'].append(np.zeros(6))
    history['nu_cmd_b'].append(np.zeros(6))

    for i in range(num_steps):
        eta, nu = x_curr[0:6], x_curr[6:12]

        end_idx = min(i + N_horizon, len(eta_ref_full))
        eta_ref_window = eta_ref_full[i:end_idx, :]
        if len(eta_ref_window) < N_horizon:
            padding = np.repeat([eta_ref_window[-1]], N_horizon - len(eta_ref_window), axis=0)
            eta_ref_window = np.vstack((eta_ref_window, padding))

        # --- A. Position control (outer loop) ---
        if use_ff:
            # SSA-aware diff so the yaw-wrap at figure-8 lobe crossings doesn't
            # inject a 2-pi/dt spike into the feedforward term.
            eta_ref_dot_raw = cal_eta_err_with_ssa(eta_ref_window[0], eta_ref_prev) / dt
            eta_dot_ref_filtered = (
                (alpha_ff * eta_ref_dot_raw) + (1.0 - alpha_ff) * eta_dot_ref_filtered
                if use_filter else eta_ref_dot_raw
            )
            eta_ref_prev = eta_ref_window[0]
            nu_cmd_b = pose_ctrl.compute_control(eta, eta_ref_window[0], dt, eta_dot_ref_filtered)
        else:
            nu_cmd_b = pose_ctrl.compute_control(eta, eta_ref_window[0], dt)

        # --- B. Velocity control (inner loop) ---
        if is_mpc:
            if use_preview:
                # Roll the outer-loop forward N_horizon steps to build a nu reference window
                if pose_ctrl_type == 'pid':
                    nu_ref_window = generate_virtual_reference_pid(
                        eta_ref_window, eta, nu_cmd_b, N_horizon, dt, pose_ctrl, pose_ctrl,
                    )
                else:
                    nu_ref_window = generate_virtual_reference_ff_pi(
                        eta_ref_window, eta, nu_cmd_b, use_filter, eta_dot_ref_filtered,
                        alpha_ff, N_horizon, dt, pose_ctrl, pose_ctrl, use_ff,
                    )
                ref_in = nu_ref_window
            else:
                ref_in = nu_cmd_b

            t0 = time.perf_counter() if record_mpc_time else None
            tau_cmd = vel_ctrl.compute_control(nu, ref_in)
            if t0 is not None:
                history['mpc_time'].append(time.perf_counter() - t0)
        else:
            tau_cmd = vel_ctrl.compute_control(nu, nu_cmd_b, dt)

        tau_cmd = np.clip(np.asarray(tau_cmd).flatten(), -200, 200)

        # --- C. Plant integration (RK4) + yaw wrap ---
        x_curr = rk4_step(sys, x_curr, tau_cmd, dt)
        x_curr[3:6] = (x_curr[3:6] + np.pi) % (2 * np.pi) - np.pi

        history['t'].append((i + 1) * dt)
        history['x'].append(x_curr.copy())
        history['tau'].append(tau_cmd.copy())
        history['nu_cmd_b'].append(nu_cmd_b.copy())

    return history


# =============================================================================
# Reference trajectory + study cases (paper protocol)
# =============================================================================

DT = 0.1
TFINAL = 150.0
HOLD_TIME = 40.0
PARAMS = ROOT / "params" / "control"


def build_reference() -> np.ndarray:
    """Figure-8 (N=1, closed loop at z=2) followed by 40 s of station-keeping."""
    eta_ref, _ = figure8_path(
        N=1,
        start_point=np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        end_point=np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        dt=DT,
        tfinal=TFINAL,
    )
    hold_steps = int(HOLD_TIME / DT)
    hold = np.tile(eta_ref[-1, :], (hold_steps, 1))
    return np.vstack((eta_ref, hold))


STUDY_CASES = {
    "Case_1_Model_Accuracy_without_Preview": [
        {"label": "PIwFF -- Nominal MPC without preview (DMDc)",
         "config": PARAMS / "dmdc" / "without_preview" / "ffpi_stdmpc_gain.yaml",
         "model": "dmdc"},
        {"label": "PIwFF -- Nominal MPC without preview (eDMDc)",
         "config": PARAMS / "edmdc" / "without_preview" / "ffpi_stdmpc_gain.yaml",
         "model": "edmdc"},
    ],
    "Case_2_Preview_Impact_Nominal_MPC": [
        {"label": "PIwFF -- Nominal MPC without preview (DMDc)",
         "config": PARAMS / "dmdc" / "without_preview" / "ffpi_stdmpc_gain.yaml",
         "model": "dmdc"},
        {"label": "PIwFF -- Nominal MPC with preview (DMDc)",
         "config": PARAMS / "dmdc" / "with_preview" / "ffpi_stdmpc_gain.yaml",
         "model": "dmdc"},
    ],
    "Case_3_Controller_Comparison": [
        {"label": "PID -- PID",
         "config": PARAMS / "dpid_gain.yaml",
         "model": "dmdc"},  
        {"label": "PIwFF -- Nominal MPC with preview (DMDc)",
         "config": PARAMS / "dmdc" / "with_preview" / "ffpi_stdmpc_gain.yaml",
         "model": "dmdc"},
        {"label": "PIwFF -- Offset-Free MPC with preview (DMDc)",
         "config": PARAMS / "dmdc" / "with_preview" / "ffpi_intmpc_gain.yaml",
         "model": "dmdc"},
    ],
}


def cascade_kwargs_from(cfg: dict) -> dict:
    """Translate a controller yaml dict into run_cascade_simulation kwargs."""
    pose = cfg["position_controller"]
    vel = cfg["velocity_controller"]
    return dict(
        N_horizon=int(vel.get("params", {}).get("N_horizon", 10)),
        pose_ctrl_type=pose.get("type", "pid"),
        use_ff=pose.get("use_feedforward", False),
        use_filter=pose.get("use_filter", False),
        alpha_ff=pose.get("alpha_ff", 1.0),  # source default (filter no-op when 1.0)
        use_preview=vel.get("use_preview", False),
        vel_ctrl_type=vel.get("type", "pid"),
    )


def sanitize_node_name(label: str) -> str:
    """acados / casadi function names must be valid C identifiers with no
    consecutive underscores."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", label)).strip("_")


def main() -> None:
    t_sim = TFINAL + HOLD_TIME
    eta_ref_full = build_reference()
    print(f"reference: figure-8 N=1, {eta_ref_full.shape}, total {t_sim:.0f}s")

    auv_params = load_params(ROOT / "params" / "xplorer_mini.yaml")
    models = {
        "dmdc": FittedModel.load(ROOT / "result" / "sysid" / "trained_model" / "dmdc"),
        "edmdc": FittedModel.load(ROOT / "result" / "sysid" / "trained_model" / "edmdc"),
    }
    print(f"models loaded: {list(models.keys())}")

    for case_name, cases in STUDY_CASES.items():
        print(f"\n{'=' * 80}\n {case_name}\n{'=' * 80}")
        histories, labels = [], []
        for case in cases:
            print(f"\n--- {case['label']} ---")
            cfg = yaml.safe_load(case["config"].read_text())
            node_name = sanitize_node_name(case["label"])
            pose_ctrl = create_position_controller(cfg["position_controller"])
            vel_ctrl = create_velocity_controller(
                cfg["velocity_controller"],
                model=models[case["model"]],
                node_name=node_name, dt=DT,
            )
            kw = cascade_kwargs_from(cfg)
            hist = run_cascade_simulation(
                auv_params, pose_ctrl, vel_ctrl, eta_ref_full,
                t_end=t_sim, dt=DT, **kw,
            )
            histories.append(hist)
            labels.append(case["label"])

        save_dir = ROOT / "result" / "control" / case_name
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_response(histories=histories, labels=labels,
                      eta_ref_full=eta_ref_full, save_dir=save_dir)
        print(f"\n  -> saved to {save_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
