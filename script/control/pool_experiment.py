"""Pool-experiment comparison harness (paper §VI figure-8 study).

Reads the released ROS2 pool logs (rosbag2 / mcap) for the three closed-loop
controllers and writes the IEEE Access comparison PDFs -- the real-hardware
counterpart to `simulation.py`, which produces the same figures from the
Fossen sim.

The mcap files embed their own ROS2 message schema, so no ROS2 install or the
`xplorer_mini_common_interfaces` package is needed -- only `mcap-ros2` (PyPI).

Run as a script (paper protocol, defaults to data/control/pool_experiment -> result/control/pool_experiment):

  python script/control/pool_experiment.py
  python script/control/pool_experiment.py --data-dir <dir> --out <dir>
"""

import argparse
import io
import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mcap_ros2.reader import read_ros2_messages
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utils.plot import apply_ieee_style  # noqa: E402  (needs src on sys.path)

# ==========================================
# 1. Helper Functions & Data Mapping
# ==========================================
def flatten_msg(msg_obj, prefix=""):
    items = {}
    for slot in dir(msg_obj):
        if slot.startswith('_') or callable(getattr(msg_obj, slot)): continue
        val = getattr(msg_obj, slot)
        key = f"{prefix}{slot}"
        if hasattr(val, '__slots__'): items.update(flatten_msg(val, prefix=f"{key}."))
        else: items[key] = val
    return items

def build_topic_map(df):
    ref_prefix, act_prefix, cmd_prefix = None, None, None
    for col in df.columns:
        if col.endswith('ref_nav.position.x') or col.endswith('ref_filtered.position.x'):
            ref_prefix = col.replace('.position.x', '')
        elif col.endswith('odom_filtered.pose.pose.position.x'):
            act_prefix = col.replace('.pose.pose.position.x', '')
        elif col.endswith('nu_cmd.linear.x'):
            cmd_prefix = col.replace('.linear.x', '')
            
    if not ref_prefix or not act_prefix: return None
    
    tmap = {
        'ref_pos_x': f'{ref_prefix}.position.x', 'ref_pos_y': f'{ref_prefix}.position.y', 'ref_pos_z': f'{ref_prefix}.position.z', 'ref_ori': f'{ref_prefix}.orientation',
        'act_pos_x': f'{act_prefix}.pose.pose.position.x', 'act_pos_y': f'{act_prefix}.pose.pose.position.y', 'act_pos_z': f'{act_prefix}.pose.pose.position.z', 'act_ori': f'{act_prefix}.pose.pose.orientation',
        'cmd_u': f'{cmd_prefix}.linear.x', 'cmd_v': f'{cmd_prefix}.linear.y', 'cmd_w': f'{cmd_prefix}.linear.z',
        'cmd_p': f'{cmd_prefix}.angular.x', 'cmd_q': f'{cmd_prefix}.angular.y', 'cmd_r': f'{cmd_prefix}.angular.z',
        'act_u': f'{act_prefix}.twist.twist.linear.x', 'act_v': f'{act_prefix}.twist.twist.linear.y', 'act_w': f'{act_prefix}.twist.twist.linear.z',
        'act_p': f'{act_prefix}.twist.twist.angular.x', 'act_q': f'{act_prefix}.twist.twist.angular.y', 'act_r': f'{act_prefix}.twist.twist.angular.z',
    }

    for col in df.columns:
        if col.endswith('est_tau.force.x'):
            tau_p = col.replace('.force.x', '')
            tmap.update({
                'tau_x': f'{tau_p}.force.x',  'tau_y': f'{tau_p}.force.y',  'tau_z': f'{tau_p}.force.z',
                'tau_k': f'{tau_p}.torque.x', 'tau_m': f'{tau_p}.torque.y', 'tau_n': f'{tau_p}.torque.z'
            })
            break
        elif col.endswith('wrench.force.x'):
            tau_p = col.replace('.wrench.force.x', '')
            tmap.update({
                'tau_x': f'{tau_p}.wrench.force.x',  'tau_y': f'{tau_p}.wrench.force.y',  'tau_z': f'{tau_p}.wrench.force.z',
                'tau_k': f'{tau_p}.wrench.torque.x', 'tau_m': f'{tau_p}.wrench.torque.y', 'tau_n': f'{tau_p}.wrench.torque.z'
            })
            break
            
    return tmap

# ==========================================
# 2. Data Loading & Statistical Filtering
# ==========================================
def load_and_sync_data(base_dir, controllers, path_type):
    base_path = Path(base_dir).expanduser().resolve()
    raw_storage = {}
    
    print("\n[*] Initializing Data Pipeline...")
    for ctrl in controllers:
        mcap_files = list((base_path / ctrl / path_type).rglob('*.mcap'))
        if not mcap_files: 
            print(f"    [!] Skip: {ctrl} (MCAP not found)")
            continue
        
        with open(mcap_files[0], 'rb') as f: mcap_stream = io.BytesIO(f.read())
            
        data_by_topic = defaultdict(list)
        for msg in read_ros2_messages(mcap_stream):
            topic = getattr(msg, 'topic', getattr(msg.channel, 'topic', 'unknown')).strip('/')
            flat = flatten_msg(msg.ros_msg); flat["_log_time"] = msg.log_time
            data_by_topic[topic].append(flat)
            
        dfs = [pd.DataFrame(recs).set_index('_log_time').rename(columns=lambda c: f"{t}.{c}") for t, recs in data_by_topic.items()]
        df = pd.concat(dfs, axis=1).sort_index()
        raw_storage[ctrl] = {'df': df, 'len': len(df)}
        print(f"    - {ctrl}: {len(df)} samples")

    if not raw_storage: return {}

    mean_len = np.mean([v['len'] for v in raw_storage.values()])
    filtered_storage = {k: v for k, v in raw_storage.items() if v['len'] >= mean_len * 0.5} # Tolerance added
    min_len = int(min(v['len'] for v in filtered_storage.values()))
    
    print(f"\n[*] Syncing Time Domain (Min Samples = {min_len})")
    
    comp_data = {}
    for ctrl, data in filtered_storage.items():
        df = data['df'].iloc[:min_len].copy()
        df = df.ffill().bfill().reset_index(names='_log_time')
        
        tmap = build_topic_map(df)
        if tmap:
            # Note: Converted to RADIANS to match the plotting backend requirements
            for ori in ['ref_ori', 'act_ori']:
                prefix = ori.split('_')[0]
                cols = [f"{tmap[ori]}.x", f"{tmap[ori]}.y", f"{tmap[ori]}.z", f"{tmap[ori]}.w"]
                if all(c in df.columns for c in cols):
                    e = R.from_quat(df[cols].to_numpy()).as_euler('xyz', degrees=False) 
                    df.loc[:, f'{prefix}_roll'] = e[:, 0]
                    df.loc[:, f'{prefix}_pitch'] = e[:, 1]
                    df.loc[:, f'{prefix}_yaw'] = e[:, 2]
            comp_data[ctrl] = {'df': df, 'tmap': tmap}
    return comp_data

# ==========================================
# 3. Enhanced Visualization & Metrics
# ==========================================
def _ctrl_color(name):
    n = name.lower()
    if 'pid' in n and 'nominal' not in n and 'offset' not in n:
        ctrl_p = 0
    elif 'offset' in n:
        ctrl_p = 2
    else:
        ctrl_p = 1
    mod_p  = 1 if 'edmdc' in n.replace('-', '') else 0
    prev_p = 1 if 'with preview' in n else 0

    palette = {
        (0, 0, 0): '#758A93',
        (1, 0, 0): '#7FCBC4',
        (1, 0, 1): '#3A78B8',
        (1, 1, 0): '#8FB8DE',
        (1, 1, 1): '#2A9D8F',
        (2, 0, 0): '#CC6B5C',
        (2, 0, 1): '#C5172E',
        (2, 1, 0): '#D9A055',
        (2, 1, 1): '#FF7444',
    }
    return palette.get((ctrl_p, mod_p, prev_p), '#B0B0B0')

def plot_journal_style(comp_data, save_dir=None):
    # ========================================================================
    # IEEE Access Specifications
    # ========================================================================
    # Column widths (inches): 1-col = 3.5", 2-col = 7.16", max depth = 8.5"
    # Min resolution: 300 dpi (color/grayscale), 600 dpi (line art)
    # Accepted formats: EPS, PDF, PS, TIFF, PNG (PDF preferred for vector)
    COL1, COL2 = 3.5, 7.16

    # IEEE-compliant rcParams (Times New Roman, IEEE typography) -- single
    # source of truth in src/utils/plot.py, shared with simulation.py / validate.py.
    apply_ieee_style()

    labels = list(comp_data.keys())
    colors = [_ctrl_color(lbl) for lbl in labels]

    # ------------------------------------------------------------------------
    # Use original (verbose) controller names in legends and tables.
    # label_map is kept as identity for code consistency.
    # ------------------------------------------------------------------------
    label_map = {l: l for l in labels}
    short_labels = labels

    # ------------------------------------------------------------------------
    # Extract base time
    # ------------------------------------------------------------------------
    base_ctrl = labels[0]
    df_b = comp_data[base_ctrl]['df']
    t_raw = df_b['_log_time'].values.astype(np.int64)
    t = (t_raw - t_raw[0]) / 1e9

    def get_arr(df, tmap, key):
        return df[tmap[key]].values if key in tmap and tmap[key] in df.columns else np.zeros(len(df))

    def get_arr_direct(df, key):
        return df[key].values if key in df.columns else np.zeros(len(df))

    def add_panel_labels(axs_flat):
        # IEEE Access spec: 8 pt Times New Roman, (a) (b) (c) format,
        # centered below each subfigure. Bottom row needs extra offset
        # to clear x-tick labels and the x-axis label.
        for ax, lbl in zip(axs_flat, 'abcdefghij'):
            has_xlabel = bool(ax.get_xlabel())
            y_pos = -0.45 if has_xlabel else -0.22
            ax.text(0.5, y_pos, f'({lbl})', transform=ax.transAxes,
                    fontsize=8, va='top', ha='center')

    def format_figure(fig, axs, bottom_margin, ncol=None):
        # Auto-cap ncol at 2 so the legend never forces the figure wider
        # than the configured figsize. Long verbose labels need ≤2 columns
        # at COL2=7.16" with fontsize=7.
        fig.align_ylabels(axs)
        handles, labels_l = axs[0, 0].get_legend_handles_labels()
        if ncol is None:
            ncol = min(2, len(handles))
        else:
            ncol = min(ncol, len(handles))
        # tight_layout for L/R/B margins only (top=1.0 — overridden below)
        fig.tight_layout(rect=[0.02, bottom_margin, 1, 1.0])
        # Manually pin suptitle near top and push axes top right under it.
        # 8 pt text ≈ 0.022 fig-coord in 5.8" fig (8/72/5.8); 2 mm ≈ 0.0136.
        # suptitle y=0.99 (va='top') → text bottom ≈ 0.968
        # plot top = 0.968 - 0.0136 ≈ 0.954
        if fig._suptitle is not None:
            fig._suptitle.set_y(0.99)
        fig.subplots_adjust(top=0.954, hspace=0.55)
        # loc='upper center' anchors the legend's TOP edge — increasing y
        # pulls the legend up, closer to the bottom-row panel labels.
        fig.legend(handles, labels_l, loc='upper center', ncol=ncol,
                   bbox_to_anchor=(0.5, bottom_margin + 0.04),
                   columnspacing=1.5, handletextpad=0.5,
                   frameon=True, edgecolor='black')

    # ------------------------------------------------------------------------
    # Reference & state arrays (NWU -> NED, full 6-DOF)
    # Sign flip applies identically to world-frame η, body-frame ν, and τ
    # because the WU↔ED axis swap is the same in either frame.
    # ------------------------------------------------------------------------
    NED_SIGN = np.array([1, -1, -1, 1, -1, -1])

    eta_ref = np.column_stack([get_arr(df_b, comp_data[base_ctrl]['tmap'], f'ref_pos_{x}') for x in ['x','y','z']] +
                              [get_arr_direct(df_b, f'ref_{a}') for a in ['roll','pitch','yaw']])
    eta_ref *= NED_SIGN

    ctrl_eta = {}
    for _ctrl in labels:
        _df, _tmap = comp_data[_ctrl]['df'], comp_data[_ctrl]['tmap']
        _eta = np.column_stack([get_arr(_df, _tmap, f'act_pos_{x}') for x in ['x', 'y', 'z']] +
                               [get_arr_direct(_df, f'act_{a}') for a in ['roll', 'pitch', 'yaw']])
        _eta *= NED_SIGN
        ctrl_eta[_ctrl] = _eta

    # IEEE-tuned figure size for 3x2 grids:
    #   width = COL2 (7.16") -> two ~3.4" sub-panels
    #   height = 5.8" -> three ~1.6" rows + suptitle + legend area
    GRID_FIGSIZE = (COL2, 5.8)

    # ========================================================================
    # Fig 1: Position Tracking (eta)
    # ========================================================================
    fig1, axs1 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)
    # fig1.suptitle(r'Position Tracking ($\eta$)', fontsize=8)
    labels_eta_lin = [r'$x$ (m)', r'$y$ (m)', r'$z$ (m)']
    labels_eta_ang = [r'$\phi$ (°)', r'$\theta$ (°)', r'$\psi$ (°)']

    for i in range(3):
        ref_ang_rad = np.unwrap(eta_ref[:, i+3]) if i == 2 else eta_ref[:, i+3]
        axs1[i, 0].plot(t, eta_ref[:, i], 'k--', linewidth=1.0, label='Ref' if i==0 else "")
        axs1[i, 1].plot(t, np.rad2deg(ref_ang_rad), 'k--', linewidth=1.0, label='Ref' if i==0 else "")
        for idx, ctrl in enumerate(labels):
            eta = ctrl_eta[ctrl]
            state_ang_rad = np.unwrap(eta[:, i+3]) if i == 2 else eta[:, i+3]
            axs1[i, 0].plot(t, eta[:, i], color=colors[idx], label=label_map[ctrl] if i==0 else "")
            axs1[i, 1].plot(t, np.rad2deg(state_ang_rad), color=colors[idx], label=label_map[ctrl] if i==0 else "")
        axs1[i, 0].set_ylabel(labels_eta_lin[i]); axs1[i, 0].grid(True)
        axs1[i, 1].set_ylabel(labels_eta_ang[i]); axs1[i, 1].grid(True)

    axs1[2, 0].set_xlabel('Time (s)'); axs1[2, 1].set_xlabel('Time (s)')
    axs1[2, 0].set_ylim(1.9, 2.1)
    add_panel_labels(axs1.flat)
    format_figure(fig1, axs1, bottom_margin=0.15)

    # ========================================================================
    # Fig 2 & 3: Velocity Tracking & Error (nu)
    # ========================================================================
    fig2, axs2 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)
    # fig2.suptitle(r'Velocity Tracking ($\nu$)', fontsize=8)
    fig3, axs3 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)
    # fig3.suptitle(r'Velocity Tracking Error ($e_\nu$)', fontsize=8)
    labels_nu_lin  = [r'$u$ (m/s)', r'$v$ (m/s)', r'$w$ (m/s)']
    labels_nu_ang  = [r'$p$ (°/s)', r'$q$ (°/s)', r'$r$ (°/s)']
    labels_enu_lin = [r'$e_u$ (m/s)', r'$e_v$ (m/s)', r'$e_w$ (m/s)']
    labels_enu_ang = [r'$e_p$ (°/s)', r'$e_q$ (°/s)', r'$e_r$ (°/s)']

    metrics_eta, metrics_nu = [], []
    lin_k, ang_k = ['u','v','w'], ['p','q','r']

    for idx, ctrl in enumerate(labels):
        df, tmap = comp_data[ctrl]['df'], comp_data[ctrl]['tmap']
        eta = ctrl_eta[ctrl]
        nu     = np.column_stack([get_arr(df, tmap, f'act_{k}') for k in lin_k] + [get_arr(df, tmap, f'act_{k}') for k in ang_k]) * NED_SIGN
        nu_cmd = np.column_stack([get_arr(df, tmap, f'cmd_{k}') for k in lin_k] + [get_arr(df, tmap, f'cmd_{k}') for k in ang_k]) * NED_SIGN

        e_lin     = eta_ref[:, :3] - eta[:, :3]
        e_ang_rad = (eta_ref[:, 3:6] - eta[:, 3:6] + np.pi) % (2*np.pi) - np.pi
        e_eta = np.hstack((e_lin, np.rad2deg(e_ang_rad)))
        metrics_eta.append({'rmse': np.sqrt(np.mean(e_eta**2, axis=0)), 'mae': np.mean(np.abs(e_eta), axis=0), 'maxae': np.max(np.abs(e_eta), axis=0)})

        e_nu = np.hstack((nu_cmd[:, :3] - nu[:, :3], np.rad2deg(nu_cmd[:, 3:6] - nu[:, 3:6])))
        metrics_nu.append({'rmse': np.sqrt(np.mean(e_nu**2, axis=0)), 'mae': np.mean(np.abs(e_nu), axis=0), 'maxae': np.max(np.abs(e_nu), axis=0)})

        for i in range(3):
            axs2[i, 0].plot(t, nu_cmd[:, i], '--', color=colors[idx], linewidth=0.7, alpha=0.5, label=f'{label_map[ctrl]} (Cmd)' if i==0 else "")
            axs2[i, 0].plot(t, nu[:, i], '-', color=colors[idx], label=label_map[ctrl] if i==0 else "")
            axs2[i, 1].plot(t, np.rad2deg(nu_cmd[:, i+3]), '--', color=colors[idx], linewidth=0.7, alpha=0.4)
            axs2[i, 1].plot(t, np.rad2deg(nu[:, i+3]), '-', color=colors[idx])
            axs3[i, 0].plot(t, nu_cmd[:, i] - nu[:, i], color=colors[idx], label=label_map[ctrl] if i==0 else "")
            axs3[i, 1].plot(t, np.rad2deg(nu_cmd[:, i+3] - nu[:, i+3]), color=colors[idx])

    for i in range(3):
        axs2[i, 0].set_ylabel(labels_nu_lin[i]);  axs2[i, 0].grid(True)
        axs2[i, 1].set_ylabel(labels_nu_ang[i]);  axs2[i, 1].grid(True)
        axs3[i, 0].set_ylabel(labels_enu_lin[i]); axs3[i, 0].grid(True)
        axs3[i, 1].set_ylabel(labels_enu_ang[i]); axs3[i, 1].grid(True)

    axs2[2, 0].set_xlabel('Time (s)'); axs2[2, 1].set_xlabel('Time (s)')
    axs3[2, 0].set_xlabel('Time (s)'); axs3[2, 1].set_xlabel('Time (s)')
    add_panel_labels(axs2.flat)
    add_panel_labels(axs3.flat)
    format_figure(fig2, axs2, bottom_margin=0.20)
    format_figure(fig3, axs3, bottom_margin=0.15, ncol=len(labels))

    # ========================================================================
    # Fig 4: Generalized Torque (tau)
    # ========================================================================
    fig4, axs4 = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharex=True)
    # fig4.suptitle(r'Generalized Torque ($\tau$)', fontsize=8)
    labels_tau_f = [r'$X$ (N)', r'$Y$ (N)', r'$Z$ (N)']
    labels_tau_t = [r'$K$ (N$\cdot$m)', r'$M$ (N$\cdot$m)', r'$N$ (N$\cdot$m)']
    tau_k_lin, tau_k_ang = ['x','y','z'], ['k','m','n']

    for idx, ctrl in enumerate(labels):
        df, tmap = comp_data[ctrl]['df'], comp_data[ctrl]['tmap']
        tau = np.column_stack([get_arr(df, tmap, f'tau_{k}') for k in tau_k_lin] +
                              [get_arr(df, tmap, f'tau_{k}') for k in tau_k_ang]) * NED_SIGN
        for i in range(3):
            axs4[i, 0].plot(t, tau[:, i],   color=colors[idx], label=label_map[ctrl] if i==0 else "")
            axs4[i, 1].plot(t, tau[:, i+3], color=colors[idx])

    for i in range(3):
        axs4[i, 0].set_ylabel(labels_tau_f[i]); axs4[i, 0].grid(True)
        axs4[i, 1].set_ylabel(labels_tau_t[i]); axs4[i, 1].grid(True)

    axs4[2, 0].set_xlabel('Time (s)'); axs4[2, 1].set_xlabel('Time (s)')
    add_panel_labels(axs4.flat)
    format_figure(fig4, axs4, bottom_margin=0.15, ncol=len(labels))

    # ========================================================================
    # Fig 5: 3D Path Tracking (single column, IEEE Access COL1 = 3.5")
    # ========================================================================
    # Figure height tuned to bring the cube top close to the suptitle:
    # 3D axes always render the cube with large internal padding inside
    # the bbox, so a tall figure leaves a big visible gap above the cube.
    # Shorter figure -> cube fills more vertical area -> gap matches 3x2 figs.
    fig5 = plt.figure(figsize=(COL1, 3.4))
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.plot(eta_ref[:, 0], eta_ref[:, 1], eta_ref[:, 2],
             'k--', linewidth=1.0, label='Reference')
    for idx, ctrl in enumerate(labels):
        eta = ctrl_eta[ctrl]
        n = ctrl.lower()
        lw = 1.0 if 'pid' in n and 'nominal' not in n and 'offset' not in n else 0.9
        ax5.plot(eta[:, 0], eta[:, 1], eta[:, 2],
                 color=colors[idx], linewidth=lw, label=label_map[ctrl])

    # Transparent panes (avoid gray background per IEEE multicolor guideline)
    for axis in [ax5.xaxis, ax5.yaxis, ax5.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('lightgray')

    # View tuned for single-column width: slightly higher elevation and a
    # less extreme azimuth keeps all axis labels visible at narrow width.
    ax5.view_init(elev=25, azim=-55)
    # Box aspect: less elongated than the COL2 version (was [2.0, 1.0, 0.6])
    # so the figure-8 still reads at 3.5" width without crushing z-depth.
    ax5.set_box_aspect([1.6, 1.0, 0.7])
    # Sparse ticks — at COL1 width, 3D tick labels overlap quickly.
    ax5.locator_params(axis='x', nbins=4)
    ax5.locator_params(axis='y', nbins=3)
    ax5.locator_params(axis='z', nbins=4)
    ax5.set_xlabel(r'$x$ (m)', fontsize=8, labelpad=-2)
    ax5.set_ylabel(r'$y$ (m)', fontsize=8, labelpad=-2)
    # Use a 2D figure-coord annotation for z-label: matplotlib's 3D zlabel
    # is frequently clipped by bbox_inches='tight'. Position is tuned to sit
    # just right of the z-tick labels at this view/box configuration.
    ax5.set_zlabel('')
    fig5.text(0.92, 0.78, r'$z$ (m)', fontsize=8, ha='center', va='center')
    ax5.set_zlim(1.5, 2.5)
    ax5.invert_zaxis()
    ax5.tick_params(labelsize=6, pad=0)
    # Use fig.suptitle (matches 3x2 plots) so title-to-axes gap is consistent.
    # fig5.suptitle('3D Path Tracking', fontsize=8)
    # fig5._suptitle.set_y(0.99)

    # Legend below the plot — 2 columns to match the other figures' layout.
    handles5, labels5 = ax5.get_legend_handles_labels()
    legend_ncol = min(2, len(handles5))
    fig5.legend(handles5, labels5, loc='lower center', ncol=legend_ncol,
                bbox_to_anchor=(0.5, 0.10),
                columnspacing=1.0, handletextpad=0.4,
                frameon=True, edgecolor='black', framealpha=1.0)

    # Bottom 0.18 leaves room for the 2-row legend at fig height 2.6".
    ax5.set_position([-0.05, 0.18, 1.05, 0.82])

    # ========================================================================
    # Fig 6: 2D Path Tracking (X-Y, single column, IEEE Access COL1 = 3.5")
    # ========================================================================
    # Height sized for plot (~2.7") + 2-col legend area (~0.7").
    fig6, ax6 = plt.subplots(figsize=(COL1, 3.4))
    ax6.plot(eta_ref[:, 0], eta_ref[:, 1], 'k--', linewidth=1.0,
             label='Reference', zorder=2)

    for idx, ctrl in enumerate(labels):
        eta = ctrl_eta[ctrl]
        px, py = eta[:, 0], eta[:, 1]
        ax6.plot(px, py, color=colors[idx], linewidth=0.8,
                 label=label_map[ctrl], zorder=3)

    ax6.set_xlabel(r'$x$ (m)')
    ax6.set_ylabel(r'$y$ (m)')
    ax6.set_aspect('equal')
    ax6.grid(True)
    # ax6.set_title('2D Path Tracking', fontsize=8, pad=5.7)  # ≈ 2 mm

    # Legend below the plot — 2 columns, matching Fig 5 layout.
    # Position the legend's top ~5 mm below the xlabel "x (m)".
    # 5 mm / fig_height(3.4") = 0.197"/3.4" ≈ 0.058 fig-coords.
    handles6, labels6 = ax6.get_legend_handles_labels()
    legend_ncol = min(2, len(handles6))
    fig6.tight_layout(rect=[0, 0.18, 1, 1])
    fig6.legend(handles6, labels6, loc='upper center', ncol=legend_ncol,
                bbox_to_anchor=(0.56, 0.25),
                columnspacing=1.0, handletextpad=0.4,
                frameon=True, edgecolor='black', framealpha=1.0)

    # ========================================================================
    # Fig 7 & 8: Metric Tables (double column)
    # ========================================================================
    def create_metric_figure(title_main, cols, metrics_data):
        fig, axs = plt.subplots(3, 1, figsize=(COL2, 5.0))
        # fig.suptitle(title_main, fontsize=8)
        metric_keys   = ['rmse', 'mae', 'maxae']
        metric_titles = ['RMSE', 'MAE', 'MaxAE']
        colors_bg = ['#d9ead3', '#cfe2f3', '#f4cccc']
        highlight_color, highlight_text_color = '#fff2cc', '#d62728'
        for i, key in enumerate(metric_keys):
            cell_text = [[f"{m[key][col_idx]:.4f}" for col_idx in range(6)] for m in metrics_data]
            tbl = axs[i].table(cellText=cell_text, rowLabels=short_labels, colLabels=cols,
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
            # axs[i].set_title(metric_titles[i], pad=4, fontsize=7, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 1.0])
        # Pin suptitle and axes-top so gap is ~2 mm
        # 8 pt text ≈ 0.022 fig-coord in 5.0" fig (8/72/5.0); 2 mm ≈ 0.0157
        if fig._suptitle is not None:
            fig._suptitle.set_y(0.99)
        fig.subplots_adjust(top=0.952)
        return fig

    fig8 = create_metric_figure(r'$\eta$ Error Metrics',
                                ['x (m)', 'y (m)', 'z (m)', r'$\phi$ (°)', r'$\theta$ (°)', r'$\psi$ (°)'],
                                metrics_eta)
    fig9 = create_metric_figure(r'$\nu$ Error Metrics',
                                ['u (m/s)', 'v (m/s)', 'w (m/s)', 'p (°/s)', 'q (°/s)', 'r (°/s)'],
                                metrics_nu)

    # ========================================================================
    # Export — PDF (vector, IEEE-preferred) at 300 dpi
    # ========================================================================
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        figs  = [fig1, fig2, fig3, fig4, fig5, fig6, fig8, fig9]
        names = ["1_position_response", "2_velocity_response", "3_velocity_error",
                 "4_control_effort", "5_3d_path", "6_2d_path",
                 "7_table_position_metrics", "8_table_velocity_metrics"]
        for f, name in zip(figs, names):
            f.savefig(save_path / f"{name}.pdf", format='pdf',
                      bbox_inches='tight', pad_inches=0.05, dpi=300)
        plt.close('all')
    else:
        plt.show()

# ==========================================
# 4. Main Execution
# ==========================================
CONTROLLERS = [
    "PID -- PID",
    "PIwFF -- Nominal MPC with preview (DMDc)",
    "PIwFF -- Offset-Free MPC with preview (DMDc)",
]
TARGET_PATH = "figure8"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "control" / "pool_experiment",
                        help="root of the pool logs (default: data/control/pool_experiment)")
    parser.add_argument("--out", type=Path, default=ROOT / "result" / "control" / "pool_experiment",
                        help="output dir for the PDFs (default: result/control/pool_experiment)")
    args = parser.parse_args()

    pool_dir = args.data_dir.expanduser().resolve()
    result_dir = args.out.expanduser().resolve()

    comp_data = load_and_sync_data(pool_dir, CONTROLLERS, TARGET_PATH)

    if comp_data:
        print("\n[*] Generating Journal Style Plots & Metrics...")
        plot_journal_style(comp_data, save_dir=result_dir)
        print(f"\n[Success] All graphs & tables exported to: {result_dir}")
    else:
        sys.exit(f"[!] No pool logs found under {pool_dir}")