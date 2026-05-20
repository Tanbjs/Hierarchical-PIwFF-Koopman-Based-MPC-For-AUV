"""Preprocess stage: data/dataset -> data/raw -> data/cleaned -> data/smooth.

Step 1  dataset -> raw
    Copy every trial CSV from data/dataset/ to data/raw/, preserving the
    {depth}/{trial}/ folder layout and the original filename.

Step 2  raw -> cleaned
    Apply the vectorized Hampel outlier filter to every CSV under data/raw/:
        data/cleaned/{depth}/{trial}/hampel_window{w}_sigma{n}.csv

Step 3  cleaned -> smooth
    Apply a centered moving-average filter to every CSV under data/cleaned/:
        data/smooth/{depth}/{trial}/{cleaner_stem}/ma_window{w}.csv

The filename conventions mirror the kmc reference pipeline so multiple
cleaner/smoother configurations can coexist side-by-side without collisions.

Usage
    python script/sysid/preprocess.py
    python script/sysid/preprocess.py --hampel-window 5 --hampel-sigma 3 --ma-window 5
    python script/sysid/preprocess.py --skip-raw       # reuse existing data/raw/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sysid import discover_trials, hampel_filter, moving_average, process_stage

logger = logging.getLogger(__name__)


def copy_dataset_to_raw(src: Path, dst: Path) -> int:
    """Copy CSVs under `src` to `dst`, mirroring the folder layout."""
    paths = discover_trials(src)
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        rel = p.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out_path)
        logger.info("dataset -> raw : %s", out_path.relative_to(ROOT))
    return len(paths)


def build_cleaned_stage(
    src: Path,
    dst: Path,
    window_size: int,
    n_sigmas: float,
) -> int:
    """Apply Hampel filter; write data/cleaned/.../hampel_window{w}_sigma{n}.csv."""
    sigma_repr = int(n_sigmas) if float(n_sigmas).is_integer() else n_sigmas
    suffix = f"hampel_window{window_size}_sigma{sigma_repr}.csv"

    written = process_stage(
        src_root=src,
        dst_root=dst,
        transform=lambda df: hampel_filter(df, window_size=window_size, n_sigmas=n_sigmas),
        rename=lambda _p: suffix,
    )
    return len(written)


def build_smooth_stage(
    src: Path,
    dst: Path,
    window_size: int,
) -> int:
    """Apply MA filter; write data/smooth/.../<cleaner_stem>/ma_window{w}.csv."""
    smoother_suffix = f"ma_window{window_size}.csv"

    def rename(input_path: Path) -> str:
        # input_path.stem encodes the cleaner config, e.g. "hampel_window20_sigma20"
        return f"{input_path.stem}/{smoother_suffix}"

    written = process_stage(
        src_root=src,
        dst_root=dst,
        transform=lambda df: moving_average(df, window_size=window_size),
        rename=rename,
    )
    return len(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hampel-window", type=int, default=5,
                        help="Hampel half-window size (default: 5)")
    parser.add_argument("--hampel-sigma", type=float, default=3.0,
                        help="Hampel MAD-sigma threshold (default: 3)")
    parser.add_argument("--ma-window", type=int, default=5,
                        help="Moving-average window size (default: 5)")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip the dataset -> raw copy; reuse existing data/raw/")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = ROOT / "data" / "dataset"
    raw_dir = ROOT / "data" / "raw"
    cleaned_dir = ROOT / "data" / "cleaned"
    smooth_dir = ROOT / "data" / "smooth"

    if args.skip_raw:
        logger.info("Step 1/3  dataset -> raw  (skipped, reusing %s)", raw_dir)
    else:
        logger.info("Step 1/3  dataset -> raw")
        n = copy_dataset_to_raw(dataset_dir, raw_dir)
        logger.info("  copied %d trial CSVs into %s", n, raw_dir)

    logger.info(
        "Step 2/3  raw -> cleaned  (Hampel window=%d, n_sigmas=%g)",
        args.hampel_window, args.hampel_sigma,
    )
    n = build_cleaned_stage(
        src=raw_dir,
        dst=cleaned_dir,
        window_size=args.hampel_window,
        n_sigmas=args.hampel_sigma,
    )
    logger.info("  wrote %d filtered CSVs into %s", n, cleaned_dir)

    logger.info("Step 3/3  cleaned -> smooth  (MA window=%d)", args.ma_window)
    n = build_smooth_stage(
        src=cleaned_dir,
        dst=smooth_dir,
        window_size=args.ma_window,
    )
    logger.info("  wrote %d smoothed CSVs into %s", n, smooth_dir)


if __name__ == "__main__":
    main()
