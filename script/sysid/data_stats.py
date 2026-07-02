"""Quick mean/variance of tau (est_tau) and state (odom twist) over a split.

State / input groups mirror params/sysid/*.yaml:
    state (twist) : odom_filtered.twist.twist.{linear,angular}.{x,y,z}
    input  (tau)  : est_tau.{force,torque}.{x,y,z}

Data under data/sysid/split/ is already NED + smoothed (preprocess.py output), i.e.
exactly what the Koopman fit consumes.

Usage: python script/sysid/data_stats.py [--root data/sysid/split/train]
                                         [--md report.md]

With no --md, the tables are written to
result/sysid/data_stats_<split>.md (split = last path segment of --root).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def pick(cols, words):
    return [c for c in cols if all(w in c for w in words)]


def render_md(md_path, header, tables):
    """Write each (title, DataFrame) as a GitHub-flavoured Markdown table."""
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data statistics", "", f"`{header}`", ""]
    for title, out in tables:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(out.to_markdown(floatfmt=".6f"))
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sysid/split/train")
    ap.add_argument("--md", default=None,
                    help="write the tables to this Markdown file "
                         "(default: result/sysid/data_stats_<split>.md)")
    args = ap.parse_args()

    root = ROOT / args.root
    md_path = args.md or ROOT / "result" / "sysid" / f"data_stats_{root.name}.md"
    paths = sorted(root.rglob("*.csv"))
    if not paths:
        raise SystemExit(f"No CSVs under {root}")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, axis=0, ignore_index=True)

    numeric = set(df.select_dtypes(include="number").columns)
    tau_cols = [c for c in pick(df.columns, ["est_tau"]) if c in numeric]
    state_cols = [c for c in pick(df.columns, ["odom", "twist"]) if c in numeric]

    def report(title, cols):
        sub = df[cols].astype(float)
        out = pd.DataFrame({
            "mean": sub.mean(),
            "variance": sub.var(ddof=1),
            "std": sub.std(ddof=1),
            "min": sub.min(),
            "max": sub.max(),
        })
        print(f"\n=== {title}  ({len(cols)} cols, N={len(sub)} samples) ===")
        with pd.option_context("display.float_format", lambda x: f"{x: .6f}",
                               "display.width", 200, "display.max_columns", None):
            print(out)
        return out

    # Report angular twist (p, q, r) in deg/s instead of rad/s. Converting the
    # raw samples up front makes every stat -- incl. variance ((180/pi)^2 scale)
    # -- come out in the right unit automatically.
    ang_cols = [c for c in state_cols if "angular" in c]
    df[ang_cols] = np.rad2deg(df[ang_cols])

    header = (f"Root: {root.relative_to(ROOT)}   trials: {len(paths)}   "
              f"rows: {len(df)}")
    print(header)
    tau_out = report("tau  (est_tau force/torque) [N, N*m]", tau_cols)
    state_out = report("state (odom twist linear/angular) [m/s, deg/s]", state_cols)

    out_md = render_md(md_path, header, [
        ("tau  (est_tau force/torque) [N, N*m]", tau_out),
        ("state (odom twist linear/angular) [m/s, deg/s]", state_out),
    ])
    print(f"\nWrote Markdown -> {out_md}")


if __name__ == "__main__":
    main()
