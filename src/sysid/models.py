"""Koopman-based linear models identified via DMDc and EDMDc.

This codebase fits ordinary least squares only (see params/sysid/*.yaml and the
manuscript -- ridge/lasso were dropped to keep the published pipeline aligned
with a single regression rule).
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from .observable import BaseObservable


class DMDc(BaseEstimator):
    """Dynamic Mode Decomposition with Control (DMDc).

    Fits x_{k+1} = A x_k + B u_k via ordinary least squares.
    """

    def __init__(self):
        self.A = None
        self.B = None
        self.n_features_in_ = None

    def fit(self, X1: ArrayLike, X2: ArrayLike, U: ArrayLike) -> "DMDc":
        X_current = np.asarray(X1)
        X_next = np.asarray(X2)
        U_current = np.asarray(U)

        self.n_features_in_ = X_current.shape[1]

        Omega = np.concatenate([X_current, U_current], axis=1)
        K = LinearRegression(fit_intercept=False).fit(Omega, X_next).coef_

        n_states = X_current.shape[1]
        self.A = K[:, :n_states]
        self.B = K[:, n_states:]
        return self

    def predict(self, X: ArrayLike, U: ArrayLike) -> np.ndarray:
        if self.A is None or self.B is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet."
            )

        X = np.asarray(X)
        U = np.asarray(U)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: expected {self.n_features_in_}, got {X.shape[1]}"
            )

        return X @ self.A.T + U @ self.B.T


class EDMDc(BaseEstimator):
    """Extended Dynamic Mode Decomposition with Control (EDMDc).

    Fits z_{k+1} = A z_k + B u_k via ordinary least squares, with
    z = psi(x). Assumes the observable includes the original state x as the
    first components of z, giving C = [I 0].
    """

    def __init__(self, obs: BaseObservable):
        super().__init__()
        self._obs_func = obs

        self.A = None
        self.B = None
        self.Omega = None

    def fit(self, X1: np.ndarray, X2: np.ndarray, U: np.ndarray) -> "EDMDc":
        X_current = np.asarray(X1)
        X_next = np.asarray(X2)
        U_current = np.asarray(U)

        Z_current = self._obs_func.fit_transform(X_current)
        Z_next = self._obs_func.transform(X_next)
        n_lifted = Z_current.shape[1]

        self.Omega = np.hstack((Z_current, U_current))
        K = LinearRegression(fit_intercept=False).fit(self.Omega, Z_next).coef_

        self.A = K[:, :n_lifted]
        self.B = K[:, n_lifted:]
        return self

    def predict(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        self._check_fitted()

        Z = self._obs_func.transform(X)
        Z_next = Z @ self.A.T + U @ self.B.T
        return Z_next

    def _check_fitted(self):
        if self.A is None or self.B is None:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")
