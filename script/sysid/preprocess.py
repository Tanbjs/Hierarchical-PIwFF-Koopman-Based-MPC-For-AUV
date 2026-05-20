"""Preprocess: data/dataset -> raw -> cleaned -> smooth -> split.

Step 1  dataset -> raw
    Copy every trial CSV from data/dataset/ to data/raw/, preserving the
    {depth}/{trial}/ folder layout and the original filename.

Step 2  raw -> cleaned
    Apply the vectorized Hampel outlier filter to every CSV under data/raw/:
        data/cleaned/{depth}/{trial}/hampel_window{w}_sigma{n}.csv

Step 3  cleaned -> smooth
    Apply a centered moving-average filter to every CSV under data/cleaned/:
        data/smooth/{depth}/{trial}/{cleaner_stem}/ma_window{w}.csv

Step 4  smooth -> split
    Trial-level train/test split via sklearn.train_test_split applied
    sequentially (mirrors kmc's process.py:split_and_log_datasets). Output
    is a folder of relative symlinks pointing back into data/smooth/:
        data/split/{train,test}/{depth}/{trial}/{cleaner_stem}/ma_window{w}.csv
    2-way ratio -> {train, test}; 3-way ratio -> {train, val, test}.

The filename conventions mirror the kmc reference pipeline so multiple
cleaner/smoother configurations can coexist side-by-side without collisions.

Usage
    python script/sysid/preprocess.py
    python script/sysid/preprocess.py --hampel-window 5 --hampel-sigma 3 --ma-window 5
    python script/sysid/preprocess.py --skip-raw       # reuse existing data/raw/
    python script/sysid/preprocess.py --ratio 0.7 0.15 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import train_test_split

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


def split_paths(
    paths: Sequence[Path],
    ratio: Sequence[float],
    random_state: int,
) -> dict[str, list[Path]]:
    """Sequential sklearn split mirroring kmc's split_and_log_datasets.

    Operates on the sorted path list (sequence preserved across runs).

    ratio=[0.85, 0.15]        -> {"train", "test"}
    ratio=[0.7, 0.15, 0.15]   -> {"train", "val", "test"}
    """
    if len(ratio) not in (2, 3):
        raise ValueError(f"ratio must have 2 or 3 entries, got {ratio!r}")

    contexts = ["train", "val", "test"] if len(ratio) == 3 else ["train", "test"]

    total = float(sum(ratio))
    if abs(total - 1.0) > 1e-9:
        logger.warning("ratio %s does not sum to 1.0; normalizing.", ratio)
        ratio = [r / total for r in ratio]

    segments: list[list[Path]] = []
    remaining = list(paths)
    for i in range(len(ratio) - 1):
        relative_test_size = 1.0 - (ratio[i] / sum(ratio[i:]))
        main_part, split_part = train_test_split(
            remaining, test_size=relative_test_size, random_state=random_state
        )
        segments.append(sorted(main_part))
        remaining = list(split_part)
    segments.append(sorted(remaining))

    return dict(zip(contexts, segments))


def build_split_stage(
    src: Path,
    dst: Path,
    ratio: Sequence[float],
    seed: int,
    ma_window: int,
) -> dict[str, int]:
    """Symlink smooth CSVs into data/split/{context}/, mirroring layout.

    Idempotent (clears existing dst). Only links files matching the active
    ma_window (one smoother config per split).
    """
    if dst.exists():
        shutil.rmtree(dst)

    paths = sorted(src.rglob(f"ma_window{ma_window}.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSVs matched {src}/**/ma_window{ma_window}.csv -- "
            f"did the smooth stage run with --ma-window {ma_window}?"
        )

    splits = split_paths(paths, ratio=ratio, random_state=seed)
    counts = {}
    for context, plist in splits.items():
        for s in plist:
            rel = s.relative_to(src)
            out = dst / context / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(s, out.parent), out)
        counts[context] = len(plist)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hampel-window", type=int, default=5,
                        help="Hampel half-window size (default: 5)")
    parser.add_argument("--hampel-sigma", type=float, default=3.0,
                        help="Hampel MAD-sigma threshold (default: 3)")
    parser.add_argument("--ma-window", type=int, default=5,
                        help="Moving-average window size (default: 5)")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.85, 0.15],
                        help="Train/test (or train/val/test) ratio. Default: 0.85 0.15 "
                             "(matches kmc kaec configs).")
    parser.add_argument("--seed", type=int, default=42,
                        help="random_state for sklearn.train_test_split (default: 42, "
                             "matches kmc).")
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
    split_dir = ROOT / "data" / "split"

    if args.skip_raw:
        logger.info("Step 1/4  dataset -> raw  (skipped, reusing %s)", raw_dir)
    else:
        logger.info("Step 1/4  dataset -> raw")
        n = copy_dataset_to_raw(dataset_dir, raw_dir)
        logger.info("  copied %d trial CSVs into %s", n, raw_dir)

    logger.info(
        "Step 2/4  raw -> cleaned  (Hampel window=%d, n_sigmas=%g)",
        args.hampel_window, args.hampel_sigma,
    )
    n = build_cleaned_stage(
        src=raw_dir,
        dst=cleaned_dir,
        window_size=args.hampel_window,
        n_sigmas=args.hampel_sigma,
    )
    logger.info("  wrote %d filtered CSVs into %s", n, cleaned_dir)

    logger.info("Step 3/4  cleaned -> smooth  (MA window=%d)", args.ma_window)
    n = build_smooth_stage(
        src=cleaned_dir,
        dst=smooth_dir,
        window_size=args.ma_window,
    )
    logger.info("  wrote %d smoothed CSVs into %s", n, smooth_dir)

    logger.info(
        "Step 4/4  smooth -> split  (ratio=%s, seed=%d)", args.ratio, args.seed,
    )
    counts = build_split_stage(
        src=smooth_dir,
        dst=split_dir,
        ratio=args.ratio,
        seed=args.seed,
        ma_window=args.ma_window,
    )
    logger.info(
        "  linked %s into %s",
        ", ".join(f"{k}={v}" for k, v in counts.items()),
        split_dir,
    )


if __name__ == "__main__":
    main()
