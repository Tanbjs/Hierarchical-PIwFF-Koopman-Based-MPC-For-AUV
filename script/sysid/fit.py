"""Fit a DMDc or EDMDc Koopman model from the train split.

Pipeline (mirrors the kmc reference, MLflow stripped out):

    data/split/train/**/*.csv
        -> feature selection (state / input / output column groups)
        -> StandardScaler / MinMaxScaler fit on train
        -> stack one-step pairs (X_k, U_k, X_{k+1})
        -> DMDc or EDMDc regression
        -> dump A, B, scaler params, columns, config snapshot, metadata

The train/test split is produced upstream by script/sysid/preprocess.py
(Step 4); fit just reads it. Trained artifacts land in
result/sysid/trained_model/<method>/ (gitignored, reproducible).

Usage
    python script/sysid/fit.py --config params/sysid/dmdc.yaml
    python script/sysid/fit.py --config params/sysid/edmdc.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sysid import DMDc, EDMDc, PolynomialObservable, inject_euler_angles

logger = logging.getLogger(__name__)

TrialFeatures = tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]


# ---------------------------------------------------------------------------
# Data discovery and feature selection
# ---------------------------------------------------------------------------

def discover_smooth_trials(root: Path, glob: str) -> list[Path]:
    paths = sorted(root.glob(glob))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched {root}/{glob}")
    return paths


def load_trial(path: Path) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df = inject_euler_angles(df)
    key = str(path.relative_to(ROOT))
    return key, df


def match_columns(columns: Iterable[str], groups: Sequence[Sequence[str]]) -> list[str]:
    """Select columns whose name contains every word in at least one group."""
    selected = []
    for col in columns:
        if any(all(word in col for word in group) for group in groups):
            selected.append(col)
    return selected


def _priority(col: str) -> int:
    if "position" in col:
        return 0
    if "orientation" in col or "euler" in col:
        return 1
    if "twist" in col:
        return 2
    return 3


def resolve_columns(
    df: pd.DataFrame, cfg: dict
) -> tuple[list[str], list[str], list[str]]:
    numeric_cols = set(df.select_dtypes(include="number").columns)
    candidates = [c for c in df.columns if c in numeric_cols]
    state_col = match_columns(candidates, cfg["state"])
    input_col = match_columns(candidates, cfg["input"])
    output_col = match_columns(candidates, cfg["output"])

    if not state_col or not input_col or not output_col:
        raise ValueError(
            f"Empty selection: state={len(state_col)}, input={len(input_col)}, "
            f"output={len(output_col)} -- check feature_selection groups."
        )

    # Reorder state so output columns come first (EDMDc assumes C = [I 0]).
    sorted_state = sorted(state_col, key=_priority)
    output_first = [c for c in sorted_state if c in output_col]
    rest = [c for c in sorted_state if c not in output_col]
    reorded_state = output_first + rest
    output_col = [c for c in reorded_state if c in output_col]

    return reorded_state, input_col, output_col


def select_features(
    trials: list[tuple[str, pd.DataFrame]],
    state_col: list[str],
    input_col: list[str],
    output_col: list[str],
) -> list[TrialFeatures]:
    out: list[TrialFeatures] = []
    for key, df in trials:
        out.append((key, df[state_col], df[input_col], df[output_col]))
    return out


# ---------------------------------------------------------------------------
# Scaling and one-step stacking
# ---------------------------------------------------------------------------

def fit_scalers(train: list[TrialFeatures], cfg: dict):
    method = cfg.get("method", "standard")
    args = cfg.get("args", {}) or {}

    state_concat = pd.concat([t[1] for t in train], axis=0)
    input_concat = pd.concat([t[2] for t in train], axis=0)
    output_concat = pd.concat([t[3] for t in train], axis=0)

    if method == "standard":
        sx, su, sy = StandardScaler(**args), StandardScaler(**args), StandardScaler(**args)
    elif method == "minmax":
        feature_range = tuple(args.get("feature_range", [-1, 1]))
        sx = MinMaxScaler(feature_range=feature_range)
        su = MinMaxScaler(feature_range=feature_range)
        sy = MinMaxScaler(feature_range=feature_range)
    elif method == "none":
        return None, None, None
    else:
        raise ValueError(f"Unknown scaler method: {method!r}")

    sx.fit(state_concat)
    su.fit(input_concat)
    sy.fit(output_concat)
    return sx, su, sy


def apply_scalers(
    trials: list[TrialFeatures], sx, su, sy
) -> list[TrialFeatures]:
    if sx is None:
        return trials
    out: list[TrialFeatures] = []
    for key, x_df, u_df, y_df in trials:
        out.append((
            key,
            pd.DataFrame(sx.transform(x_df), columns=x_df.columns),
            pd.DataFrame(su.transform(u_df), columns=u_df.columns),
            pd.DataFrame(sy.transform(y_df), columns=y_df.columns),
        ))
    return out


def stack_one_step(trials: list[TrialFeatures]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xk, Uk, Xk1 = [], [], []
    for _, x_df, u_df, y_df in trials:
        Xk.append(x_df.iloc[:-1].to_numpy())
        Uk.append(u_df.iloc[:-1].to_numpy())
        Xk1.append(y_df.iloc[1:].to_numpy())
    return np.concatenate(Xk), np.concatenate(Uk), np.concatenate(Xk1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dmdc(train: list[TrialFeatures]) -> tuple[DMDc, dict]:
    Xk, Uk, Xk1 = stack_one_step(train)
    Omega = np.concatenate([Xk, Uk], axis=1)
    cond = float(np.linalg.cond(Omega.T @ Omega))

    model = DMDc().fit(Xk, Xk1, Uk)

    metadata = {
        "n_samples": int(Xk.shape[0]),
        "n_state": int(Xk.shape[1]),
        "n_input": int(Uk.shape[1]),
        "cond": cond,
    }
    logger.info("DMDc fit (OLS): %s", metadata)
    return model, metadata


def train_edmdc(train: list[TrialFeatures], cfg: dict) -> tuple[EDMDc, dict]:
    Xk, Uk, Xk1 = stack_one_step(train)

    obs_cfg = cfg.get("observable", {}) or {}
    if obs_cfg.get("type", "polynomial") != "polynomial":
        raise ValueError(f"Unsupported observable type: {obs_cfg.get('type')!r}")
    observable = PolynomialObservable(
        degree=int(obs_cfg.get("degree", 2)),
        include_bias=bool(obs_cfg.get("include_bias", False)),
        interaction_only=bool(obs_cfg.get("interaction_only", False)),
    )

    model = EDMDc(obs=observable).fit(Xk, Xk1, Uk)
    cond = float(np.linalg.cond(model.Omega.T @ model.Omega))

    metadata = {
        "observable": {
            "type": "polynomial",
            "degree": observable.degree,
            "include_bias": observable.include_bias,
            "interaction_only": observable.interaction_only,
            "feature_names": observable.get_output_names(),
        },
        "n_samples": int(Xk.shape[0]),
        "n_state": int(Xk.shape[1]),
        "n_lifted": int(model.A.shape[0]),
        "n_input": int(Uk.shape[1]),
        "cond": cond,
    }
    logger.info(
        "EDMDc fit (OLS): degree=%d, n_lifted=%d, samples=%d, cond=%.3e",
        observable.degree, metadata["n_lifted"], metadata["n_samples"], cond,
    )
    return model, metadata


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_scaler(path: Path, scaler) -> None:
    if scaler is None:
        return
    joblib.dump(scaler, path)


def save_artifacts(
    out_dir: Path,
    *,
    model,
    metadata: dict,
    state_col: list[str],
    input_col: list[str],
    output_col: list[str],
    scaler_x,
    scaler_u,
    scaler_y,
    config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "A.npy", model.A)
    np.save(out_dir / "B.npy", model.B)
    save_scaler(out_dir / "scaler_x.joblib", scaler_x)
    save_scaler(out_dir / "scaler_u.joblib", scaler_u)
    save_scaler(out_dir / "scaler_y.joblib", scaler_y)

    columns = {
        "state": state_col,
        "input": input_col,
        "output": output_col,
    }
    (out_dir / "columns.json").write_text(json.dumps(columns, indent=2))

    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    try:
        display_dir = out_dir.relative_to(ROOT)
    except ValueError:
        display_dir = out_dir
    logger.info("Saved model artifacts to %s", display_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config (see params/sysid/dmdc.yaml)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dir (default: result/sysid/trained_model/<method>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = yaml.safe_load(args.config.read_text())
    method = config["method"].lower()
    if method not in {"dmdc", "edmdc"}:
        raise ValueError(f"Unsupported method: {method!r}")

    train_root = ROOT / config.get("train_root", "data/split/train")
    paths = discover_smooth_trials(train_root, "**/*.csv")
    logger.info("Loaded %d train trials from %s", len(paths),
                train_root.relative_to(ROOT))
    if not (ROOT / "data" / "split" / "test").exists():
        logger.warning(
            "No data/split/test/ found -- run script/sysid/preprocess.py first."
        )

    train_trials = [load_trial(p) for p in paths]

    state_col, input_col, output_col = resolve_columns(
        train_trials[0][1], config["feature_selection"]
    )
    logger.info("Selected %d state / %d input / %d output columns",
                len(state_col), len(input_col), len(output_col))

    train_feat = select_features(train_trials, state_col, input_col, output_col)
    sx, su, sy = fit_scalers(train_feat, config.get("scaler", {"method": "standard"}))
    train_norm = apply_scalers(train_feat, sx, su, sy)

    if method == "dmdc":
        model, meta = train_dmdc(train_norm)
    else:
        model, meta = train_edmdc(train_norm, config)

    out_dir = args.output or (ROOT / "result" / "sysid" / "trained_model" / method)
    save_artifacts(
        out_dir,
        model=model,
        metadata=meta,
        state_col=state_col,
        input_col=input_col,
        output_col=output_col,
        scaler_x=sx, scaler_u=su, scaler_y=sy,
        config=config,
    )


if __name__ == "__main__":
    main()