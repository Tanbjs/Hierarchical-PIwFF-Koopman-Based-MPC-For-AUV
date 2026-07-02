"""Shared plotting infrastructure for IEEE Access figures.

Used by script/sysid/validate.py (open-loop prediction plots) and
script/control/simulation.py (closed-loop tracking plots) -- keeps the
typography, colors, and layout consistent across the manuscript.
"""

import matplotlib.pyplot as plt

# IEEE Access page widths (inches)
COL1 = 3.5            # single column
COL2 = 7.16           # double column
GRID_FIGSIZE = (COL2, 5.8)

# Paper-convention colors
COLOR_NONLINEAR  = '#2CA02C'   # green
COLOR_DMDC       = '#3A78B8'   # blue
COLOR_EDMDC      = '#D62728'   # red
PHASE_LINE_COLOR = '#555555'   # grey (vertical phase-split line)

# 6-DOF axis labels (Fossen convention)
LABELS_ETA_LIN = [r'$x$ (m)', r'$y$ (m)', r'$z$ (m)']
LABELS_ETA_ANG = [r'$\phi$ (°)', r'$\theta$ (°)', r'$\psi$ (°)']
LABELS_NU_LIN  = [r'$u$ (m/s)', r'$v$ (m/s)', r'$w$ (m/s)']
LABELS_NU_ANG  = [r'$p$ (°/s)', r'$q$ (°/s)', r'$r$ (°/s)']
LABELS_TAU_F   = [r'$X$ (N)', r'$Y$ (N)', r'$Z$ (N)']
LABELS_TAU_T   = [r'$K$ (N$\cdot$m)', r'$M$ (N$\cdot$m)', r'$N$ (N$\cdot$m)']


def apply_ieee_style() -> None:
    """Set matplotlib rcParams for IEEE Access (Times serif, 8 pt, 300 dpi)."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 8,
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


def add_panel_labels(axs_flat) -> None:
    """Place (a), (b), (c), ... centered below each subplot (8 pt, Times)."""
    for ax, lbl in zip(axs_flat, 'abcdefghij'):
        has_xlabel = bool(ax.get_xlabel())
        y_pos = -0.45 if has_xlabel else -0.22
        ax.text(0.5, y_pos, f'({lbl})', transform=ax.transAxes,
                fontsize=8, va='top', ha='center')


def format_figure(fig, axs, bottom_margin: float, ncol: int | None = None) -> None:
    """Align ylabels, pin suptitle near top, draw bottom-center legend.

    `ncol`: legend column count. If None, defaults to min(2, n_handles) so
    long verbose labels (e.g. controller comparison) don't force the
    figure wider than the configured figsize. Pass an explicit int to
    override (e.g. ncol=5 for GT + 3 models + phase split).
    """
    fig.align_ylabels(axs)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if ncol is None:
        ncol = min(2, len(handles))
    else:
        ncol = min(ncol, len(handles))
    fig.tight_layout(rect=[0.02, bottom_margin, 1, 1.0])
    if fig._suptitle is not None:
        fig._suptitle.set_y(0.99)
    fig.subplots_adjust(top=0.954, hspace=0.55)
    fig.legend(handles, labels, loc='upper center', ncol=ncol,
               bbox_to_anchor=(0.5, bottom_margin + 0.04),
               columnspacing=1.5, handletextpad=0.5,
               frameon=True, edgecolor='black')
