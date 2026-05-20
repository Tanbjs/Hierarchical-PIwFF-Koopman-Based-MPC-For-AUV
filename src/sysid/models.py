"""Koopman-based linear models identified via DMDc and EDMDc."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from .observable import BaseObservable


class DMDc(BaseEstimator):
    """Dynamic Mode Decomposition with Control (DMDc).

    Fits a discrete-time linear model: x_{k+1} = A x_k + B u_k.

    Regression methods:
        - 'ols':   Ordinary Least Squares.
        - 'ridge': L2 regularization, strength `alpha`.
        - 'lasso': L1 regularization, strength `alpha`.
    """

    def __init__(self, method: str = "ridge", alpha: float = 1.0):
        self.method = method
        self.alpha = alpha

        self.A = None
        self.B = None

        self.n_features_in_ = None

    def fit(self, X1: ArrayLike, X2: ArrayLike, U: ArrayLike) -> "DMDc":
        X_current = np.asarray(X1)
        X_next = np.asarray(X2)
        U_current = np.asarray(U)

        self.n_features_in_ = X_current.shape[1]

        Omega = np.concatenate([X_current, U_current], axis=1)

        if self.method == "ridge":
            regressor = Ridge(alpha=self.alpha, fit_intercept=False)
        elif self.method == "ols":
            regressor = LinearRegression(fit_intercept=False)
        elif self.method == "lasso":
            regressor = Lasso(alpha=self.alpha, fit_intercept=False)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Supported: ['ols', 'ridge', 'lasso']"
            )

        regressor.fit(Omega, X_next)

        K = regressor.coef_
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

    Enforces C = [I 0], assuming the observable function includes the original
    state x as the first components of z.

    Dynamics:  z_{k+1} = A z_k + B u_k
    """

    def __init__(
        self,
        obs: BaseObservable,
        method: str = "ols",
        alpha: float = 1e-5,
    ):
        super().__init__()
        self._obs_func = obs
        self.method = method
        self.alpha = alpha

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

        if self.method == "ridge":
            regressor = Ridge(alpha=self.alpha, fit_intercept=False)
        elif self.method == "ols":
            regressor = LinearRegression(fit_intercept=False)
        elif self.method == "lasso":
            regressor = Lasso(alpha=self.alpha, fit_intercept=False)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'ridge', 'ols', or 'lasso'."
            )

        regressor.fit(self.Omega, Z_next)

        K = regressor.coef_
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
