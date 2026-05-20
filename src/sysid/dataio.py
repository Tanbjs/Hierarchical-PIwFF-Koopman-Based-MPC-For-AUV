"""Local-filesystem loader for the Xplorer-mini closed-loop CSV dataset.

Mirrors the load sequence from the kmc reference pipeline (S3-backed):
    discover -> read CSV -> inject Euler angles from quaternions.

Each trial CSV follows the rosbag2 export schema, with flat columns such as:
    ref_filtered.orientation.{x,y,z,w}
    odom_filtered.pose.pose.orientation.{x,y,z,w}
    odom_filtered.twist.twist.{linear,angular}.{x,y,z}
    est_tau.{force,torque}.{x,y,z}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


def discover_trials(root: str | Path = "data/dataset") -> list[Path]:
    """Return all trial CSVs under `root`, sorted alphabetically for determinism."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    paths = sorted(root.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return paths


def _extract_ordered_quat(
    df: pd.DataFrame, prefix: str, pattern: str = "orientation"
) -> np.ndarray:
    """Extract quaternion columns in strict [x, y, z, w] order."""
    candidates = df.filter(like=prefix).filter(like=pattern).columns
    cols = []
    for q_comp in ("x", "y", "z", "w"):
        col = next((c for c in candidates if c.endswith(f".{q_comp}")), None)
        if col is None:
            raise ValueError(
                f"Missing quaternion component '.{q_comp}' for prefix '{prefix}'"
            )
        cols.append(col)
    return df[cols].to_numpy()


def inject_euler_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Add roll/pitch/yaw columns derived from quaternion columns.

    Adds columns for both reference and odometry orientations:
        ref_filtered.orientation.euler.{roll,pitch,yaw}
        odom_filtered.pose.pose.orientation.euler.{roll,pitch,yaw}
    Modifies the DataFrame in-place and also returns it.
    """
    targets = [
        ("ref", "ref_filtered.orientation.euler"),
        ("odom", "odom_filtered.pose.pose.orientation.euler"),
    ]
    for prefix, out_prefix in targets:
        try:
            quats = _extract_ordered_quat(df, prefix)
        except ValueError as e:
            logger.warning("Skipping Euler conversion for prefix '%s': %s", prefix, e)
            continue
        euler = R.from_quat(quats).as_euler("xyz", degrees=False)
        df[f"{out_prefix}.roll"] = euler[:, 0]
        df[f"{out_prefix}.pitch"] = euler[:, 1]
        df[f"{out_prefix}.yaw"] = euler[:, 2]
    return df


def load_trial(path: str | Path, inject_euler: bool = True) -> pd.DataFrame:
    """Read a single trial CSV and (optionally) inject Euler angles."""
    df = pd.read_csv(path)
    if inject_euler:
        df = inject_euler_angles(df)
    return df


def load_dataset(
    root: str | Path = "data/dataset",
    inject_euler: bool = True,
) -> list[tuple[str, pd.DataFrame]]:
    """Discover and load all trial CSVs.

    Returns a list of (key, DataFrame), sorted by key, matching the shape of
    AUVDataFetchService.load_dataset() from the kmc reference pipeline.
    The key is the relative path under `root.parent`.
    """
    root = Path(root)
    paths = discover_trials(root)

    trials = []
    for p in paths:
        key = str(p.relative_to(root.parent))
        df = load_trial(p, inject_euler=inject_euler)
        trials.append((key, df))
        logger.info("Loaded %s -- shape %s", key, df.shape)
    return trials


def process_stage(
    src_root: str | Path,
    dst_root: str | Path,
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    rename: Callable[[Path], str] | None = None,
) -> list[Path]:
    """Walk src_root for CSVs, apply `transform` to each, write to dst_root.

    The relative folder layout under src_root is mirrored under dst_root. The
    filename is preserved by default; pass a `rename` callable to override it
    (used by the cleaned/ and smooth/ stages, which encode filter parameters
    into the output filename a la the kmc pipeline).

    Args:
        src_root:  Input root (e.g. "data/raw" or "data/cleaned").
        dst_root:  Output root (e.g. "data/cleaned", "data/smooth"). Created if missing.
        transform: A function `df -> df` applied to each trial (e.g. `hampel_filter`).
        rename:    Optional `Path -> str` returning the output filename (no parents).
                   When omitted, the input filename is reused.

    Returns:
        List of written CSV paths.
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    paths = discover_trials(src_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    written = []
    for p in paths:
        rel = p.relative_to(src_root)
        out_name = rename(p) if rename is not None else rel.name
        out_path = dst_root / rel.parent / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(p)
        df_out = transform(df)
        df_out.to_csv(out_path, index=False)

        logger.info("Stage %s -> %s (shape %s)", src_root.name, out_path, df_out.shape)
        written.append(out_path)

    return written