"""Evaluation script for Koopman-based system identification models (DMDc / eDMDc).

Implements the multi-step prediction protocol described in the manuscript:
  - p-step ahead open-loop prediction (sliding-window multi-step)
  - Per-trajectory and ensemble-averaged metrics
  - One-step and multi-step prediction visualization

Naming convention follows the paper: "DMDc" and "eDMDc" (lowercase 'e').

Ported from xplorer_mini_sim_ws/.../notebook/prediction.py with MLflow / MinIO
and the nonlinear / kinematic baselines stripped out. Periodic rollout
(periodic_rollout_prediction) has been removed at the user's request.
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sim_logger = logging.getLogger("Validation")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sysid import PolynomialObservable, inject_euler_angles
from utils import f_dyn, load_params, rk4_step


# =============================================================================
# Fitted-model loader (local-disk equivalent of DMDcWrapper / EDMDcWrapper)
# =============================================================================

@dataclass
class FittedModel:
    """Local-disk equivalent of DMDcWrapper / EDMDcWrapper.

    `predict(x, u)` returns (x_next, y_next) in physical units:
    - x_next: propagated full state -- seed for the next predict() call
    - y_next: observation at k+1 in output_col space
    For EDMDc, exploits C = [I 0]: the first n_state components of z are x.
    """
    name: str
    A: np.ndarray
    B: np.ndarray
    scaler_x: object
    scaler_u: object
    scaler_y: object
    state_col: list
    input_col: list
    output_col: list
    observable: object  # PolynomialObservable | None
    n_state: int
    n_output: int

    @classmethod
    def load(cls, model_dir: Path, name: str) -> "FittedModel":
        A = np.load(model_dir / "A.npy")
        B = np.load(model_dir / "B.npy")
        sx = joblib.load(model_dir / "scaler_x.joblib")
        su = joblib.load(model_dir / "scaler_u.joblib")
        sy = joblib.load(model_dir / "scaler_y.joblib")
        cols = json.loads((model_dir / "columns.json").read_text())
        meta = json.loads((model_dir / "metadata.json").read_text())

        observable = None
        if "observable" in meta:
            cfg = meta["observable"]
            observable = PolynomialObservable(
                degree=int(cfg["degree"]),
                include_bias=bool(cfg["include_bias"]),
                interaction_only=bool(cfg["interaction_only"]),
            )
            observable.fit(np.zeros((1, int(meta["n_state"]))))

        return cls(
            name=name, A=A, B=B,
            scaler_x=sx, scaler_u=su, scaler_y=sy,
            state_col=cols["state"], input_col=cols["input"], output_col=cols["output"],
            observable=observable,
            n_state=int(meta["n_state"]),
            n_output=len(cols["output"]),
        )

    def predict(self, x: np.ndarray, u: np.ndarray):
        x_s = self.scaler_x.transform(np.atleast_2d(x))
        u_s = self.scaler_u.transform(np.atleast_2d(u))
        if self.observable is None:  # DMDc
            x_next_s = x_s @ self.A.T + u_s @ self.B.T
            y_next = self.scaler_y.inverse_transform(x_next_s[:, :self.n_output]).squeeze(0)
            x_next = self.scaler_x.inverse_transform(x_next_s).squeeze(0)
        else:  # EDMDc
            z_s = self.observable.transform(x_s)
            z_next_s = z_s @ self.A.T + u_s @ self.B.T
            y_next = self.scaler_y.inverse_transform(z_next_s[:, :self.n_output]).squeeze(0)
            x_next = self.scaler_x.inverse_transform(z_next_s[:, :self.n_state]).squeeze(0)
        return x_next, y_next


# =============================================================================
# Prediction routines (ported verbatim from the reference, periodic_rollout dropped)
# =============================================================================

def _call_predict(model, x_k, u_k):
    """Unified predict call. Returns (next_state, observation_at_k+1)."""
    return model.predict(x_k, u_k)


def one_step_prediction(model, x_curr, u_curr):
    """One-step-ahead prediction.

    For each k, computes hat{x}_{k+1} = f(x_k, u_k) and stores it at index k+1.
    The slot y_pred[0] is filled with x_curr[0] since no prediction exists for
    the initial step (zero contribution to RMSE).
    """
    N = x_curr.shape[0]
    y_pred = np.zeros((N, x_curr.shape[1]))
    y_pred[0, :] = x_curr[0, :]
    for i in range(u_curr.shape[0]):
        _, y_next = _call_predict(model, x_curr[i, :], u_curr[i, :])
        if i + 1 < N:
            y_pred[i + 1, :] = np.squeeze(y_next)
    return y_pred


def multi_step_prediction(model, step, x_curr, u_curr):
    """p-step ahead prediction (sliding window).

    For every t, starts from ground-truth x_curr[t] and propagates p steps
    to produce y_pred[t+p].  The first p entries are filled with ground truth
    because no p-step prediction is available there.
    """
    N = x_curr.shape[0]
    Nu = u_curr.shape[0]
    y_pred = np.zeros((N, x_curr.shape[1]))
    y_pred[:step, :] = x_curr[:step, :]
    for t in range(N - step):
        x_k = x_curr[t, :].copy()
        for j in range(step):
            if t + j < Nu:
                next_state, y_next = _call_predict(model, x_k, u_curr[t + j, :])
                x_k = np.squeeze(next_state)
        y_pred[t + step, :] = np.squeeze(y_next)
    return y_pred


def nonlinear_one_step_prediction(x_full, u_curr, params, dt=0.1):
    """One-step nonlinear prediction using f_dyn + RK4 (everything in NED).

    x_full is the 12-DOF state [eta (6) | nu (6)] in NED; u_curr is tau (6).
    At every step k the full ground-truth state is used as the integration
    seed, so prediction error does not accumulate (matches the manuscript's
    one-step protocol).
    """
    N = x_full.shape[0]
    y_pred = np.zeros((N, 6))
    y_pred[0, :] = x_full[0, 6:12]
    sys = lambda x, u: f_dyn(x, u, params)
    for i in range(u_curr.shape[0]):
        x_next = rk4_step(sys, x_full[i, :], u_curr[i, :], dt)
        if i + 1 < N:
            y_pred[i + 1, :] = x_next[6:12]
    return y_pred


def nonlinear_multi_step_prediction(step, x_full, u_curr, params, dt=0.1):
    """p-step nonlinear prediction (sliding window) using f_dyn + RK4 in NED.

    For every t, starts from ground-truth x_full[t] and propagates p steps
    via the nonlinear dynamics. Stores the predicted nu at y_pred[t+p]; the
    first p entries are filled with ground truth.
    """
    N = x_full.shape[0]
    Nu = u_curr.shape[0]
    y_pred = np.zeros((N, 6))
    y_pred[:step, :] = x_full[:step, 6:12]
    sys = lambda x, u: f_dyn(x, u, params)
    for t in range(N - step):
        x_k = x_full[t, :].copy()
        for j in range(step):
            if t + j < Nu:
                x_k = rk4_step(sys, x_k, u_curr[t + j, :], dt)
        y_pred[t + step, :] = x_k[6:12]
    return y_pred


# =============================================================================
# Metrics (verbatim from the reference)
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Per-channel RMSE (the only metric reported in manuscript Table 2)."""
    Ns = y_true.shape[0]
    err = y_true - y_pred
    return np.sqrt(np.sum(err ** 2, axis=0) / Ns)


# =============================================================================
# Reporting (SVG metric tables, ported verbatim)
# =============================================================================

MODEL_NAMES = ['Nonlinear', 'DMDc', 'eDMDc']  # paper convention: lowercase 'e'

# Phase boundary from manuscript Table 1: maneuvering 0-60s, hovering 60-100s.
PHASE_SPLIT_TIME = 60.0


def _resolve_time_array(traj: pd.DataFrame) -> np.ndarray:
    """Resolve and zero-shift the time axis (seconds) from a trajectory DataFrame.

    Prefers explicit `header.stamp.sec + 1e-9 * header.stamp.nanosec`. Falls
    back to `time` / `t` (assumed seconds). For `timestamp`, auto-detects ns
    encoding (int64 ROS timestamps) and rescales to seconds.
    """
    if 'header.stamp.sec' in traj.columns and 'header.stamp.nanosec' in traj.columns:
        t = (traj['header.stamp.sec'].astype(float).to_numpy()
             + traj['header.stamp.nanosec'].astype(float).to_numpy() * 1e-9)
        return t - t[0] if len(t) else t
    for col in ('time', 't'):
        if col in traj.columns:
            t = traj[col].to_numpy(dtype=float)
            return t - t[0] if len(t) else t
    if 'timestamp' in traj.columns:
        t = traj['timestamp'].to_numpy(dtype=float)
        t = t - t[0] if len(t) else t
        # ROS timestamps in nanoseconds -- rescale once shifted to start at 0.
        if len(t) > 0 and t[-1] > 1e6:
            t = t * 1e-9
        return t
    return np.arange(len(traj), dtype=float)


def _phase_split_index(time: np.ndarray, split_t: float = PHASE_SPLIT_TIME) -> int:
    """Return index k such that time[:k] is maneuvering, time[k:] is hovering."""
    return int(np.searchsorted(time, split_t, side='left'))


def save_rmse_table_as_svg(df, main_title, file_path):
    """Render a single RMSE table as SVG (mirrors manuscript Table 2 layout)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    fig.patch.set_facecolor('#ffffff')

    ax.axis('off')
    ax.set_title(main_title, fontsize=14, weight='bold',
                 fontfamily='sans-serif', pad=10, color='#2c3e50')

    df_disp = df.copy()
    for col in df_disp.columns:
        if col != 'Model':
            df_disp[col] = df_disp[col].apply(lambda x: f"{x:.4f}")

    table = ax.table(cellText=df_disp.values, colLabels=df_disp.columns,
                     loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # RMSE: lower is better -- bold/colour the best cell in each column.
    data_cols = [c for c in df.columns if c != 'Model']
    best_vals = {}
    for c in data_cols:
        vals = df[c].values
        best_vals[c] = np.min(vals) if not np.allclose(vals, vals[0]) else None

    for (row, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontfamily='sans-serif')
            cell.set_facecolor('#1e3d59')
        else:
            cell.set_text_props(fontfamily='sans-serif', color='#333333')
            cell.set_facecolor('#f4f7f6' if row % 2 == 0 else '#ffffff')
            if col_idx > 0:
                col_name = df.columns[col_idx]
                current_val = df.iloc[row - 1][col_name]
                best = best_vals.get(col_name)
                if best is not None and current_val == best:
                    cell.set_text_props(weight='bold', color='#16a085')

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(file_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    sim_logger.info(f"Saved RMSE Table to {file_path}")


def _build_rmse_df(model_names, model_rmse_list):
    """Build a single RMSE DataFrame with paper units."""
    cols = ['u [m/s]', 'v [m/s]', 'w [m/s]', 'p [°/s]', 'q [°/s]', 'r [°/s]']
    data = {'Model': model_names}
    for i, col in enumerate(cols):
        data[col] = [m[i] for m in model_rmse_list]
    return pd.DataFrame(data)


def report_metrics_separate_phase(results, save_dir: Path, step: int):
    """Per-phase metrics partitioned at t=PHASE_SPLIT_TIME (paper Table 2).

    Maneuvering: t in [0, PHASE_SPLIT_TIME).
    Hovering:    t in [PHASE_SPLIT_TIME, T_end].
    """
    n_one_man, d_one_man, e_one_man = [], [], []
    n_ms_man,  d_ms_man,  e_ms_man  = [], [], []
    n_one_hov, d_one_hov, e_one_hov = [], [], []
    n_ms_hov,  d_ms_hov,  e_ms_hov  = [], [], []

    for result in results:
        traj = result['trajectory']
        traj_name = result['traj_name']
        true_state = traj[[c for c in traj.columns
                          if c.startswith('odom_filtered.twist.twist')]].values

        time = _resolve_time_array(traj)
        k = _phase_split_index(time, PHASE_SPLIT_TIME)

        # Maneuvering phase: t < PHASE_SPLIT_TIME
        ts_man = true_state[:k]
        n_one_m = calculate_metrics(ts_man, result['nonlinear_one_step_pred'][:k])
        d_one_m = calculate_metrics(ts_man, result['dmdc_one_step_pred'][:k])
        e_one_m = calculate_metrics(ts_man, result['edmdc_one_step_pred'][:k])
        m_lo, m_hi = step, max(step, k)
        n_ms_m = calculate_metrics(true_state[m_lo:m_hi],
                                   result['nonlinear_multi_step_pred'][m_lo:m_hi])
        d_ms_m = calculate_metrics(true_state[m_lo:m_hi],
                                   result['dmdc_multi_step_pred'][m_lo:m_hi])
        e_ms_m = calculate_metrics(true_state[m_lo:m_hi],
                                   result['edmdc_multi_step_pred'][m_lo:m_hi])

        # Hovering phase: t >= PHASE_SPLIT_TIME
        ts_hov = true_state[k:]
        n_one_h = calculate_metrics(ts_hov, result['nonlinear_one_step_pred'][k:])
        d_one_h = calculate_metrics(ts_hov, result['dmdc_one_step_pred'][k:])
        e_one_h = calculate_metrics(ts_hov, result['edmdc_one_step_pred'][k:])
        h_lo = max(step, k)
        n_ms_h = calculate_metrics(true_state[h_lo:],
                                   result['nonlinear_multi_step_pred'][h_lo:])
        d_ms_h = calculate_metrics(true_state[h_lo:],
                                   result['dmdc_multi_step_pred'][h_lo:])
        e_ms_h = calculate_metrics(true_state[h_lo:],
                                   result['edmdc_multi_step_pred'][h_lo:])

        n_one_man.append(n_one_m); d_one_man.append(d_one_m); e_one_man.append(e_one_m)
        n_ms_man.append(n_ms_m);   d_ms_man.append(d_ms_m);   e_ms_man.append(e_ms_m)
        n_one_hov.append(n_one_h); d_one_hov.append(d_one_h); e_one_hov.append(e_one_h)
        n_ms_hov.append(n_ms_h);   d_ms_hov.append(d_ms_h);   e_ms_hov.append(e_ms_h)

        traj_dir = save_dir / traj_name
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Per-trajectory tables: maneuvering phase
        save_rmse_table_as_svg(
            _build_rmse_df(MODEL_NAMES, [n_one_m, d_one_m, e_one_m]),
            "One-Step Prediction Metrics (Maneuvering Phase, t < 60s)",
            traj_dir / 'metrics_one_step_maneuvering.svg')
        save_rmse_table_as_svg(
            _build_rmse_df(MODEL_NAMES, [n_ms_m, d_ms_m, e_ms_m]),
            f"Multi-Step (p={step}) Metrics (Maneuvering Phase, t < 60s)",
            traj_dir / 'metrics_multi_step_maneuvering.svg')

        # Per-trajectory tables: hovering phase
        save_rmse_table_as_svg(
            _build_rmse_df(MODEL_NAMES, [n_one_h, d_one_h, e_one_h]),
            "One-Step Prediction Metrics (Hovering Phase, t >= 60s)",
            traj_dir / 'metrics_one_step_hovering.svg')
        save_rmse_table_as_svg(
            _build_rmse_df(MODEL_NAMES, [n_ms_h, d_ms_h, e_ms_h]),
            f"Multi-Step (p={step}) Metrics (Hovering Phase, t >= 60s)",
            traj_dir / 'metrics_multi_step_hovering.svg')

    def avg_metrics(rmse_list):
        return np.mean(rmse_list, axis=0)

    # Maneuvering averages (matches paper Table 2 'Maneuvering' columns)
    save_rmse_table_as_svg(
        _build_rmse_df(MODEL_NAMES,
            [avg_metrics(n_one_man), avg_metrics(d_one_man), avg_metrics(e_one_man)]),
        "Overall Average Metrics: One-Step Prediction (Maneuvering Phase)",
        save_dir / 'overall_metrics_one_step_maneuvering.svg')
    save_rmse_table_as_svg(
        _build_rmse_df(MODEL_NAMES,
            [avg_metrics(n_ms_man), avg_metrics(d_ms_man), avg_metrics(e_ms_man)]),
        f"Overall Average Metrics: Multi-Step (p={step}) (Maneuvering Phase)",
        save_dir / 'overall_metrics_multi_step_maneuvering.svg')

    # Hovering averages (matches paper Table 2 'Hovering' columns)
    save_rmse_table_as_svg(
        _build_rmse_df(MODEL_NAMES,
            [avg_metrics(n_one_hov), avg_metrics(d_one_hov), avg_metrics(e_one_hov)]),
        "Overall Average Metrics: One-Step Prediction (Hovering Phase)",
        save_dir / 'overall_metrics_one_step_hovering.svg')
    save_rmse_table_as_svg(
        _build_rmse_df(MODEL_NAMES,
            [avg_metrics(n_ms_hov), avg_metrics(d_ms_hov), avg_metrics(e_ms_hov)]),
        f"Overall Average Metrics: Multi-Step (p={step}) (Hovering Phase)",
        save_dir / 'overall_metrics_multi_step_hovering.svg')


# =============================================================================
# Plotting (IEEE Access style, ported verbatim minus the nonlinear baseline,
# periodic rollout, kinematic eta plots, and NED display flips since our data
# is already on disk in NED)
# =============================================================================

def plot_predictions(results, save_dir: Path, step: int):
    """Generate 3x2 prediction plots for each trajectory (IEEE Access style)."""
    COL2 = 7.16

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'lines.linewidth': 0.9,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.4,
        'grid.linestyle': ':',
        'grid.alpha': 0.7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    labels_nu_lin = [r'$u$ (m/s)', r'$v$ (m/s)', r'$w$ (m/s)']
    labels_nu_ang = [r'$p$ (°/s)', r'$q$ (°/s)', r'$r$ (°/s)']

    COLOR_NONLINEAR = '#2CA02C'
    COLOR_DMDC      = '#3A78B8'
    COLOR_EDMDC     = '#D62728'
    PHASE_LINE_COLOR = '#555555'
    GRID_FIGSIZE = (COL2, 5.8)

    def add_panel_labels(axs_flat):
        for ax, lbl in zip(axs_flat, 'abcdefghij'):
            has_xlabel = bool(ax.get_xlabel())
            y_pos = -0.45 if has_xlabel else -0.22
            ax.text(0.5, y_pos, f'({lbl})', transform=ax.transAxes,
                    fontsize=8, va='top', ha='center')

    def format_figure(fig, axs, bottom_margin, ncol=None):
        fig.align_ylabels(axs)
        handles, labels_l = axs[0, 0].get_legend_handles_labels()
        if ncol is None:
            ncol = len(handles)
        fig.tight_layout(rect=[0.02, bottom_margin, 1, 1.0])
        if fig._suptitle is not None:
            fig._suptitle.set_y(0.99)
        fig.subplots_adjust(top=0.954, hspace=0.55)
        fig.legend(handles, labels_l, loc='upper center', ncol=ncol,
                   bbox_to_anchor=(0.5, bottom_margin + 0.04),
                   columnspacing=1.5, handletextpad=0.5, fontsize=7,
                   frameon=True, edgecolor='black')

    for result in results:
        traj = result['trajectory']
        traj_name = result['traj_name']
        traj_dir = save_dir / traj_name
        traj_dir.mkdir(parents=True, exist_ok=True)

        time = _resolve_time_array(traj)

        true_state = traj[[c for c in traj.columns
                          if c.startswith('odom_filtered.twist.twist')]].values

        def _add_phase_line(ax, label_first=False):
            if len(time) == 0 or not (time[0] <= PHASE_SPLIT_TIME <= time[-1]):
                return
            ax.axvline(PHASE_SPLIT_TIME, color=PHASE_LINE_COLOR,
                       linestyle='--', linewidth=0.8, alpha=0.8,
                       label=f'Phase Split (t = {PHASE_SPLIT_TIME:g}s)' if label_first else "")

        def create_3x2_plot(pred_nonlinear, pred_dmdc, pred_edmdc, title, filename):
            fig, axs = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)
            for i in range(3):
                ax_l = axs[i, 0]
                ax_l.plot(time, true_state[:, i], 'k--', linewidth=1.0,
                          label='Ground Truth' if i == 0 else "")
                ax_l.plot(time, pred_nonlinear[:, i], color=COLOR_NONLINEAR, linewidth=0.9,
                          label='Nonlinear' if i == 0 else "")
                ax_l.plot(time, pred_dmdc[:, i], color=COLOR_DMDC, linewidth=0.9,
                          label='DMDc' if i == 0 else "")
                ax_l.plot(time, pred_edmdc[:, i], color=COLOR_EDMDC, linewidth=0.9,
                          label='eDMDc' if i == 0 else "")
                _add_phase_line(ax_l, label_first=(i == 0))
                ax_l.set_ylabel(labels_nu_lin[i]); ax_l.grid(True)

                ax_a = axs[i, 1]
                ax_a.plot(time, true_state[:, i + 3], 'k--', linewidth=1.0)
                ax_a.plot(time, pred_nonlinear[:, i + 3], color=COLOR_NONLINEAR, linewidth=0.9)
                ax_a.plot(time, pred_dmdc[:, i + 3], color=COLOR_DMDC, linewidth=0.9)
                ax_a.plot(time, pred_edmdc[:, i + 3], color=COLOR_EDMDC, linewidth=0.9)
                _add_phase_line(ax_a, label_first=False)
                ax_a.set_ylabel(labels_nu_ang[i]); ax_a.grid(True)

            axs[2, 0].set_xlabel('Time (s)'); axs[2, 1].set_xlabel('Time (s)')
            add_panel_labels(axs.flat)
            format_figure(fig, axs, bottom_margin=0.15, ncol=5)

            file_path = traj_dir / filename
            fig.savefig(file_path, format='pdf', bbox_inches='tight',
                        pad_inches=0.05, dpi=300)
            plt.close(fig)
            return file_path

        path1 = create_3x2_plot(
            result['nonlinear_one_step_pred'],
            result['dmdc_one_step_pred'], result['edmdc_one_step_pred'],
            'One-Step Prediction',
            'one_step_prediction.pdf')
        sim_logger.info(f"Saved One-Step Plot to {path1}")

        path2 = create_3x2_plot(
            result['nonlinear_multi_step_pred'],
            result['dmdc_multi_step_pred'], result['edmdc_multi_step_pred'],
            f'Multi-Step Prediction (p = {step})',
            'multi_step_prediction.pdf')
        sim_logger.info(f"Saved multi-Step Plot to {path2}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, default=10,
                        help="prediction horizon p (default: 10)")
    parser.add_argument("--model-root", type=Path,
                        default=ROOT / "result" / "sysid" / "trained_model")
    parser.add_argument("--test-root", type=Path,
                        default=ROOT / "data" / "split" / "test")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "result" / "sysid" / "validation")
    parser.add_argument("--auv-params", type=Path,
                        default=ROOT / "params" / "xplorer_mini.yaml",
                        help="AUV physical parameters for the nonlinear baseline")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="sampling interval for nonlinear RK4 integration (default: 0.1s)")
    args = parser.parse_args()

    dmdc_model  = FittedModel.load(args.model_root / "dmdc",  name="dmdc")
    edmdc_model = FittedModel.load(args.model_root / "edmdc", name="edmdc")
    auv_params = load_params(args.auv_params)

    test_paths = sorted(args.test_root.rglob("*.csv"))
    if not test_paths:
        raise FileNotFoundError(
            f"No test CSVs under {args.test_root}. Run preprocess.py first."
        )

    test_trajectories = []
    for path in test_paths:
        rel = path.relative_to(args.test_root)
        traj_name = "_".join(rel.parts[:-2])
        df = pd.read_csv(path)
        df = inject_euler_angles(df)
        test_trajectories.append({'name': traj_name, 'data': df})
        sim_logger.info(f"Loaded test trial {traj_name}")

    results = []
    step = args.step
    for traj_dict in test_trajectories:
        traj_name = traj_dict['name']
        traj = traj_dict['data'].copy()

        twist_cols = [c for c in traj.columns
                      if c.startswith('odom_filtered.twist.twist')]
        x_nu = traj[twist_cols].values   # (N, 6) nu -- Koopman input (NED)
        u    = traj[[c for c in traj.columns if c.startswith('est_tau')]].values

        # eta in NED: [x, y, z, roll, pitch, yaw] from pose position + injected Euler.
        eta_pos = traj[['odom_filtered.pose.pose.position.x',
                        'odom_filtered.pose.pose.position.y',
                        'odom_filtered.pose.pose.position.z']].to_numpy()
        eta_ang = traj[['odom_filtered.pose.pose.orientation.euler.roll',
                        'odom_filtered.pose.pose.orientation.euler.pitch',
                        'odom_filtered.pose.pose.orientation.euler.yaw']].to_numpy()
        eta = np.hstack([eta_pos, eta_ang])           # (N, 6)
        x_full = np.hstack([eta, x_nu])               # (N, 12) -- nonlinear baseline input

        # Length sanity check
        if len(u) > len(x_nu):
            sim_logger.warning(
                f"[{traj_name}] u has more rows than x ({len(u)} > {len(x_nu)}); "
                f"truncating u.")
            u = u[:len(x_nu)]

        nonlinear_one_step_pred  = nonlinear_one_step_prediction(x_full, u, auv_params, args.dt)
        nonlinear_multi_step_pred = nonlinear_multi_step_prediction(step, x_full, u, auv_params, args.dt)
        dmdc_one_step_pred  = one_step_prediction(dmdc_model,  x_nu, u)
        edmdc_one_step_pred = one_step_prediction(edmdc_model, x_nu, u)
        dmdc_multi_step_pred  = multi_step_prediction(dmdc_model,  step, x_nu, u)
        edmdc_multi_step_pred = multi_step_prediction(edmdc_model, step, x_nu, u)

        # Convert angular velocity components from rad/s to deg/s for reporting
        if len(twist_cols) >= 6:
            traj.loc[:, twist_cols[3:6]] = np.rad2deg(traj[twist_cols[3:6]].values)
            for arr in (nonlinear_one_step_pred, nonlinear_multi_step_pred,
                        dmdc_one_step_pred, edmdc_one_step_pred,
                        dmdc_multi_step_pred, edmdc_multi_step_pred):
                arr[:, 3:6] = np.rad2deg(arr[:, 3:6])

        results.append({
            'traj_name': traj_name,
            'trajectory': traj,
            'nonlinear_one_step_pred':  nonlinear_one_step_pred,
            'nonlinear_multi_step_pred': nonlinear_multi_step_pred,
            'dmdc_one_step_pred':    dmdc_one_step_pred,
            'edmdc_one_step_pred':   edmdc_one_step_pred,
            'dmdc_multi_step_pred':  dmdc_multi_step_pred,
            'edmdc_multi_step_pred': edmdc_multi_step_pred,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    report_metrics_separate_phase(results, save_dir=args.output, step=step)
    plot_predictions(results, save_dir=args.output, step=step)


if __name__ == "__main__":
    main()