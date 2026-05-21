"""Local-disk Koopman model loader (replaces kmc's DMDcWrapper / EDMDcWrapper).

Loads the artifacts written by script/sysid/fit.py:
    A.npy, B.npy, scaler_{x,u,y}.joblib, columns.json, metadata.json

Provides the duck-typed interface the controllers expect
(see src/controllers/core/kmc.py):
    .A, .B, .n_state, .n_output, .observable, .scaler_x/u/y
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .observable import PolynomialObservable


@dataclass
class FittedModel:
    """Disk-loaded Koopman fit. Used by validate.py and by the MPC controllers.

    `predict(x, u)` returns (x_next, y_next) in physical units:
    - x_next: propagated full state -- seed for the next predict() call
    - y_next: observation at k+1 in output_col space
    For EDMDc, exploits C = [I 0]: the first n_output components of z are y,
    and the first n_state components of z are x.
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
    def load(cls, model_dir: Path, name: str | None = None) -> "FittedModel":
        model_dir = Path(model_dir)
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
            name=name or model_dir.name,
            A=A, B=B,
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
