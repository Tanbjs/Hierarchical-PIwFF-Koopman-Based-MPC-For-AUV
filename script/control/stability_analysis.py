"""Closed-loop stability / eigen-analysis of the hierarchical cascade.

For BOTH loops this script computes the closed-loop eigenvalues together with
the RIGHT eigenvector (v, mode shape) and LEFT eigenvector (w) of each
eigenvalue, plus the time constant and spectral-radius stability verdict.

    Outer loop  -- FF-PI position controller on the kinematic integrator.
                   At the hover linearization (J = I, unsaturated) the assembled
                   12-state closed loop is the paper's A_o form:
                       A_o = [[ I - Ts K_P, -Ts K_I],
                              [   Ts I     ,    I   ]]
                   state [pose(6); integral(6)], Ts = dt, K_P = diag(kp),
                   K_I = diag(ki). Same controller as pi.py up to the integral
                   sign convention (similarity transform, identical spectrum).

    Inner loop  -- offset-free integral Koopman MPC (ffpi_intmpc) and nominal
                   MPC (ffpi_stdmpc). The MPC terminal cost is the DARE solution,
                   so the unconstrained MPC equals the infinite-horizon LQR; the
                   closed loop is (A - B K) for the nominal form and
                   (A_tilde - B_tilde K_tilde) for the augmented offset-free form
                   (unsaturated regime, no active constraints).

Definitions:
    A_cl v_k = lambda_k v_k         (right eigenvector v_k = column k of V)
    w_k^T A_cl = lambda_k w_k^T     (left eigenvector  w_k^T = row k of inv(V))
    w_k^T v_j = delta_kj            (biorthogonality, verified at runtime)
    tau_k = -dt / ln|lambda_k|      (1/e time constant)

Eigenvalues are modes of the COUPLED closed loop; v_k tells how the mode spreads
across the physical states (mode shape), w_k tells how the states excite it.

Run:
    python script/control/stability_analysis.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

try:
    from scipy.linalg import solve_discrete_are
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sysid import FittedModel  # noqa: E402

DT = 0.1
PARAMS = ROOT / "params" / "control"
MODEL_DIR = ROOT / "result" / "sysid" / "trained_model"

VEL_LABELS = ["u", "v", "w", "p", "q", "r"]
POSE_LABELS = ["x", "y", "z", "phi", "theta", "psi"]


# ---------------------------------------------------------------------------
# Core linear-algebra helpers
# ---------------------------------------------------------------------------
def dlqr(A, B, Q, R):
    """Discrete infinite-horizon LQR. Returns (K, P) with u = -K x."""
    A, B, Q, R = map(np.atleast_2d, (A, B, Q, R))
    if _HAS_SCIPY:
        P = solve_discrete_are(A, B, Q, R)
    else:  # fallback fixed-point iteration
        P = Q.copy()
        for _ in range(100_000):
            K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
            P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
            if np.max(np.abs(P_next - P)) < 1e-12:
                P = P_next
                break
            P = P_next
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return K, P


def tau_from_eig(lam, dt=DT):
    """1/e time constant of a discrete eigenvalue: tau = -dt / ln|lambda|."""
    mag = abs(lam)
    if mag <= 0.0 or mag >= 1.0:
        return np.inf
    return -dt / np.log(mag)


@dataclass
class EigResult:
    """Eigen-analysis of one closed-loop matrix."""
    name: str
    state_labels: list
    A_cl: np.ndarray
    eigvals: np.ndarray            # (n,)
    right_vecs: np.ndarray         # (n, n) columns = right eigenvectors v_k
    left_vecs: np.ndarray          # (n, n) rows    = left eigenvectors w_k^T (= inv(V))
    order: np.ndarray = field(default=None)  # indices sorted by |lambda| ascending

    @property
    def spectral_radius(self):
        return float(np.max(np.abs(self.eigvals)))

    @property
    def is_stable(self):
        return self.spectral_radius < 1.0


def eigen_analysis(A_cl, state_labels, name):
    """Eigenvalues + right (v) and left (w) eigenvectors of A_cl."""
    w, V = np.linalg.eig(A_cl)
    Vinv = np.linalg.inv(V)                       # rows are left eigenvectors w_k^T
    order = np.argsort(np.abs(w))                 # ascending |lambda| (fast -> slow)
    return EigResult(name=name, state_labels=state_labels, A_cl=A_cl,
                       eigvals=w, right_vecs=V, left_vecs=Vinv, order=order)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_model_matrices(method="dmdc"):
    """Return (A, B, C) of the Koopman model. For DMDc, C = [I 0] (output=state)."""
    model = FittedModel.load(MODEL_DIR / method)
    A, B = np.asarray(model.A), np.asarray(model.B)
    C = np.eye(model.n_output, A.shape[0])
    return A, B, C


def load_gains(config_path):
    return yaml.safe_load(Path(config_path).read_text())


# ---------------------------------------------------------------------------
# Outer loop (FF-PI on kinematics) -- assembled 12-state A_o (paper form)
# ---------------------------------------------------------------------------
def outer_loop(kp, ki, dt=DT):
    """Outer FF-PI closed loop in the paper's A_o form (hover, unsaturated).

        A_o = [[ I - Ts K_P, -Ts K_I],
               [    Ts I    ,    I   ]]
    state = [eta(6); xi(6)], K_P = diag(kp), K_I = diag(ki), Ts = dt.
    """
    kp, ki = np.asarray(kp, float), np.asarray(ki, float)
    Kp, Ki, I6 = np.diag(kp), np.diag(ki), np.eye(6)
    A_o = np.block([[I6 - dt * Kp, -dt * Ki],
                    [dt * I6,        I6     ]])
    labels = POSE_LABELS + [f"int_{p}" for p in POSE_LABELS]
    return eigen_analysis(A_o, labels, "outer (A_o)")


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------
def inner_nominal(A, B, C, Q_diag, R_diag):
    """Nominal MPC closed loop = (A - B K),  K = dlqr(A, B, C'QC, R)."""
    Qz = C.T @ np.diag(Q_diag) @ C
    K, _ = dlqr(A, B, Qz, np.diag(R_diag))
    return eigen_analysis(A - B @ K, list(VEL_LABELS), "inner nominal (A - B K)")


def inner_offset_free(A, B, C, Q_diag, Qi_diag, R_diag, dt=DT):
    """Offset-free integral MPC closed loop = (A_tilde - B_tilde K_tilde)."""
    nz, nu, ny = A.shape[0], B.shape[1], C.shape[0]
    A_aug = np.block([[A, np.zeros((nz, ny))],
                      [C * dt, np.eye(ny)]])
    B_aug = np.block([[B], [np.zeros((ny, nu))]])
    Qz = C.T @ np.diag(Q_diag) @ C
    Q_aug = np.block([[Qz, np.zeros((nz, ny))],
                      [np.zeros((ny, nz)), np.diag(Qi_diag)]])
    K_aug, _ = dlqr(A_aug, B_aug, Q_aug, np.diag(R_diag))
    labels = list(VEL_LABELS) + [f"int_{d}" for d in VEL_LABELS]
    return eigen_analysis(A_aug - B_aug @ K_aug, labels, "inner offset-free (A_t - B_t K_t)")


# ---------------------------------------------------------------------------
# Reporting (console)
# ---------------------------------------------------------------------------
def _fmt_vec(vec, width=6, prec=2):
    return "[" + ", ".join(f"{x:>{width}.{prec}f}" for x in np.real(vec)) + "]"


def print_eig_report(res):
    """Print eigenvalues with their right (v) and left (w) eigenvectors."""
    print("\n" + "=" * 90)
    print(f"  {res.name}   (spectral radius = {res.spectral_radius:.4f}  ->  "
          f"{'STABLE' if res.is_stable else 'UNSTABLE'})")
    print("=" * 90)
    print("  state order: [" + ", ".join(res.state_labels) + "]")
    print("  (v = right eigenvector / mode shape, w = left eigenvector; "
          "both normalized max|.|=1, slow->fast)")
    print("  time constant:  tau = -dt / ln|lambda|   (dt = %.2f s)" % DT)
    for k in res.order:
        lam = res.eigvals[k]
        v = np.real(res.right_vecs[:, k]); v = v / np.max(np.abs(v))
        wv = np.real(res.left_vecs[k, :]); wv = wv / np.max(np.abs(wv))
        im = f"{lam.imag:+.4f}j" if abs(lam.imag) > 1e-9 else ""
        print(f"\n  lambda = {lam.real:+.4f}{im}   |lambda| = {abs(lam):.4f}   "
              f"tau = {tau_from_eig(lam):.2f} s")
        print(f"     v = {_fmt_vec(v)}")
        print(f"     w = {_fmt_vec(wv)}")


def verify_biorthogonality(res, tol=1e-8):
    """Check w_k^T v_j = delta_kj  (i.e. inv(V) @ V = I)."""
    err = np.max(np.abs(res.left_vecs @ res.right_vecs - np.eye(res.eigvals.size)))
    print(f"  [check] biorthogonality  max|inv(V)V - I| = {err:.2e}  -> "
          f"{'OK' if err < tol else 'FAIL'}")
    return err < tol


# ---------------------------------------------------------------------------
# PDF report (pole-map z-plane + per-eigenvalue v / w listing)
# ---------------------------------------------------------------------------
def _draw_unit_circle(ax):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.6", lw=1.0, zorder=1)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")


def _pole_scatter(ax, res):
    _draw_unit_circle(ax)
    ax.scatter(np.real(res.eigvals), np.imag(res.eigvals), s=42,
               color="#1f77b4", edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_title(f"{res.name}\n$\\rho$ = {res.spectral_radius:.4f} "
                 f"({'stable' if res.is_stable else 'UNSTABLE'})", fontsize=9)


def _vw_text(res):
    """Monospace block listing each eigenvalue with its v and w vectors."""
    lines = [f"state order: [{', '.join(res.state_labels)}]", ""]
    for k in res.order:
        lam = res.eigvals[k]
        v = np.real(res.right_vecs[:, k]); v = v / np.max(np.abs(v))
        wv = np.real(res.left_vecs[k, :]); wv = wv / np.max(np.abs(wv))
        im = f"{lam.imag:+.4f}j" if abs(lam.imag) > 1e-9 else ""
        lines.append(f"lambda={lam.real:+.4f}{im}  |lambda|={abs(lam):.4f}  "
                     f"tau={tau_from_eig(lam):.2f}s")
        lines.append("  v=" + _fmt_vec(v))
        lines.append("  w=" + _fmt_vec(wv))
        lines.append("")
    return "\n".join(lines)


def _formula_text():
    return (
        "SAMPLING\n"
        f"   dt = Ts = {DT} s\n\n"
        "MODAL TIME CONSTANT (1/e decay of coordinate z_k)\n"
        "   tau = -dt / ln|lambda|\n\n"
        "EIGEN-DECOMPOSITION   A_cl = V Lambda V^-1\n"
        "   right eigenvector v_k (col V):    A_cl v_k   = lambda_k v_k\n"
        "   left  eigenvector w_k^T (row V^-1): w_k^T A_cl = lambda_k w_k^T\n\n"
        "STATE SOLUTION\n"
        "   continuous: x(t) = sum_k v_k (w_k^T x0) e^(lambda_c,k t)\n"
        "   discrete:   x[n] = sum_k v_k (w_k^T x0) lambda_k^n\n\n"
        "OUTER FF-PI (hover, J = I)\n"
        "   A_o = [[ I - Ts*Kp, -Ts*Ki],\n"
        "          [   Ts*I    ,    I  ]]   Kp=diag(kp), Ki=diag(ki)\n\n"
        "INNER NOMINAL MPC\n"
        "   K    = dlqr(A, B, C^T Q C, R)\n"
        "   A_cl = A - B K\n\n"
        "INNER OFFSET-FREE MPC (augmented)\n"
        "   A~ = [[A, 0],[C*Ts, I]]   B~ = [[B],[0]]   Q~ = diag(C^T Q C, Qi)\n"
        "   K~   = dlqr(A~, B~, Q~, R)\n"
        "   A_cl = A~ - B~ K~\n\n"
        "LQR / DARE\n"
        "   P = A^T P A - (A^T P B)(R + B^T P B)^-1 (B^T P A) + Q\n"
        "   K = (R + B^T P B)^-1 (B^T P A)"
    )


def _params_text(params):
    a2s = lambda M: np.array2string(np.asarray(M), precision=4, suppress_small=True,
                                    max_line_width=120)
    return (
        f"SAMPLING:  dt = {params['dt']} s\n\n"
        "OUTER FF-PI GAINS (per DOF: surge sway heave roll pitch yaw)\n"
        f"   kp = {np.asarray(params['kp'])}\n"
        f"   ki = {np.asarray(params['ki'])}\n\n"
        "INNER MPC WEIGHTS (diagonal: u v w p q r)\n"
        f"   Q   = {np.asarray(params['Q'])}\n"
        f"   Qi  = {np.asarray(params['Qi'])}\n"
        f"   R   = {np.asarray(params['R'])}\n\n"
        f"DMDc MODEL  (C = I,  open-loop spectral radius = {params['rhoA']:.4f})\n"
        f"A =\n{a2s(params['A'])}\n\n"
        f"B =\n{a2s(params['B'])}"
    )


FORMULA_MD = r"""## Formula

Discrete-time, sampling period $T_s = dt = %.2f$ s. DMDc output matrix $C = I$.

**Time constant (1/e decay)**

$$\tau = -\frac{dt}{\ln|\lambda|}$$

**Eigen-decomposition**

$$A_{cl} = V\,\Lambda\,V^{-1},\qquad A_{cl}\,v_k = \lambda_k v_k,\qquad w_k^{\top} A_{cl} = \lambda_k w_k^{\top}$$

- $v_k$ = right eigenvector (column $k$ of $V$) = mode shape — how mode $k$ spreads across the states.
- $w_k^{\top}$ = left eigenvector (row $k$ of $V^{-1}$) — how the states excite mode $k$.

**State solution**

$$x(t) = \sum_k v_k\,(w_k^{\top} x_0)\,e^{\lambda_{c,k}\,t}\qquad\text{(continuous)}$$

$$x[n] = \sum_k v_k\,(w_k^{\top} x_0)\,\lambda_k^{\,n}\qquad\text{(discrete)}$$

**Outer loop — FF-PI (hover, $J = I$, unsaturated)**

$$A_o = \begin{bmatrix} I - T_s K_P & -T_s K_I \\ T_s I & I \end{bmatrix},\qquad K_P = \mathrm{diag}(k_p),\quad K_I = \mathrm{diag}(k_i)$$

**Inner loop — nominal MPC**

$$K = \mathrm{dlqr}(A,\,B,\,C^{\top} Q\,C,\,R),\qquad A_{cl} = A - B K$$

**Inner loop — offset-free MPC (augmented)**

$$\tilde{A} = \begin{bmatrix} A & 0 \\ C\,T_s & I \end{bmatrix},\quad \tilde{B} = \begin{bmatrix} B \\ 0 \end{bmatrix},\quad \tilde{Q} = \begin{bmatrix} C^{\top} Q\,C & 0 \\ 0 & Q_i \end{bmatrix}$$

$$\tilde{K} = \mathrm{dlqr}(\tilde{A},\,\tilde{B},\,\tilde{Q},\,R),\qquad A_{cl} = \tilde{A} - \tilde{B}\,\tilde{K}$$

**LQR / DARE**

$$P = A^{\top} P A - (A^{\top} P B)(R + B^{\top} P B)^{-1}(B^{\top} P A) + Q,\qquad K = (R + B^{\top} P B)^{-1} B^{\top} P A$$

**Stability**: asymptotically stable iff $\rho(A_{cl}) = \max_k|\lambda_k| < 1$.
""" % DT


def _md_params(params):
    a2s = lambda M: np.array2string(np.asarray(M), precision=4, suppress_small=True,
                                    max_line_width=120)
    g = lambda x: ", ".join(f"{v:g}" for v in np.asarray(x))
    return (
        "## Parameters\n\n"
        f"- Sampling: `dt = {params['dt']} s`\n"
        f"- Outer FF-PI gains (surge, sway, heave, roll, pitch, yaw):\n"
        f"    - `kp = [{g(params['kp'])}]`\n"
        f"    - `ki = [{g(params['ki'])}]`\n"
        f"- Inner MPC weights (diagonal u, v, w, p, q, r):\n"
        f"    - `Q  = [{g(params['Q'])}]`\n"
        f"    - `Qi = [{g(params['Qi'])}]`\n"
        f"    - `R  = [{g(params['R'])}]`\n"
        f"- DMDc model: `C = I`, open-loop spectral radius `rho(A) = {params['rhoA']:.4f}`\n\n"
        f"```\nA =\n{a2s(params['A'])}\n\nB =\n{a2s(params['B'])}\n```\n"
    )


def _md_eig_table(res):
    rows = ["| # | $\\lambda$ | $\\lvert\\lambda\\rvert$ | $\\tau$ [s] |",
            "|---|---|---|---|"]
    for i, k in enumerate(res.order, 1):
        lam = res.eigvals[k]
        im = f"{lam.imag:+.4f}j" if abs(lam.imag) > 1e-9 else ""
        rows.append(f"| {i} | ${lam.real:+.4f}{im}$ | {abs(lam):.4f} | {tau_from_eig(lam):.2f} |")
    return "\n".join(rows)


def _md_vw_block(res):
    out = ["```", "state order: [" + ", ".join(res.state_labels) + "]", ""]
    for k in res.order:
        lam = res.eigvals[k]
        v = np.real(res.right_vecs[:, k]); v = v / np.max(np.abs(v))
        wv = np.real(res.left_vecs[k, :]); wv = wv / np.max(np.abs(wv))
        im = f"{lam.imag:+.4f}j" if abs(lam.imag) > 1e-9 else ""
        out.append(f"lambda = {lam.real:+.4f}{im}   |lambda| = {abs(lam):.4f}   "
                   f"tau = {tau_from_eig(lam):.2f} s")
        out.append("  v = " + _fmt_vec(v))
        out.append("  w = " + _fmt_vec(wv))
        out.append("")
    out.append("```")
    return "\n".join(out)


def _save_pole_png(results, png_path):
    """Pole-map z-plane figure for embedding; returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig = plt.figure(figsize=(9, 8))
    if len(results) == 3:
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)
        axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1])]
    else:
        n = len(results)
        axes = [fig.add_subplot((n + 1) // 2, 2, i + 1) for i in range(n)]
    for ax, res in zip(axes, results):
        _pole_scatter(ax, res)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_md_report(results, params, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    png = out_path.with_name(out_path.stem + "_poles.png")
    has_png = _save_pole_png(results, png)

    md = ["# Closed-loop Stability / Eigenvalue Analysis",
          "",
          f"Hierarchical PIwFF-Koopman MPC, discrete-time (`dt = {DT} s`). "
          "All quantities derived from the identified model and controller gains "
          "(unsaturated / hover regime; no new experiments).",
          ""]

    # stability summary
    md.append("## Stability summary\n")
    md.append("| loop | matrix | spectral radius $\\rho$ | verdict |")
    md.append("|---|---|---|---|")
    for res in results:
        md.append(f"| {res.name} | $A_{{cl}}$ | {res.spectral_radius:.4f} | "
                  f"{'STABLE' if res.is_stable else 'UNSTABLE'} |")
    md.append("")

    if has_png:
        md.append("## Pole map (z-plane)\n")
        md.append(f"![pole map]({png.name})\n")

    md.append(_md_params(params))
    md.append(FORMULA_MD)

    for res in results:
        md.append(f"## {res.name}\n")
        md.append(f"Spectral radius $\\rho = {res.spectral_radius:.4f}$  "
                  f"({'STABLE' if res.is_stable else 'UNSTABLE'}).\n")
        md.append(_md_eig_table(res))
        md.append("")
        md.append("Right ($v$) and left ($w$) eigenvectors (normalized max$|\\cdot|=1$, "
                  "slow $\\to$ fast):\n")
        md.append(_md_vw_block(res))
        md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[md] stability report written to {out_path}"
          + (f"  (+ {png.name})" if has_png else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("#" * 90)
    print(f"#  Closed-loop stability / eigenvalue analysis  (dt = {DT:.2f} s)")
    print("#" * 90)

    A, B, C = load_model_matrices("dmdc")
    int_cfg = load_gains(PARAMS / "dmdc" / "with_preview" / "ffpi_intmpc_gain.yaml")
    std_cfg = load_gains(PARAMS / "dmdc" / "with_preview" / "ffpi_stdmpc_gain.yaml")
    vc_i = int_cfg["velocity_controller"]["params"]
    vc_s = std_cfg["velocity_controller"]["params"]
    pc = int_cfg["position_controller"]["params"]   # outer gains identical for both

    # Cascade loops: outer (A_o), inner nominal MPC, inner offset-free MPC.
    outer = outer_loop(np.asarray(pc["kp"]), np.asarray(pc["ki"]))
    inner_nom = inner_nominal(A, B, C, vc_s["Q"], vc_s["R_abs"])
    inner_off = inner_offset_free(A, B, C, vc_i["Q"], vc_i["Qi"], vc_i["R_abs"])

    for res in (outer, inner_nom, inner_off):
        verify_biorthogonality(res)
        print_eig_report(res)

    params = dict(dt=DT, kp=pc["kp"], ki=pc["ki"],
                  Q=vc_i["Q"], Qi=vc_i["Qi"], R=vc_i["R_abs"],
                  A=A, B=B, rhoA=float(np.max(np.abs(np.linalg.eigvals(A)))))

    print("\n" + "#" * 90)
    print("#  PARAMETERS")
    print("#" * 90)
    print(_params_text(params))
    print("\n" + "#" * 90)
    print("#  FORMULAS")
    print("#" * 90)
    print(_formula_text())

    generate_md_report([outer, inner_nom, inner_off], params,
                       ROOT / "result" / "control" / "stability_analysis.md")


if __name__ == "__main__":
    main()
