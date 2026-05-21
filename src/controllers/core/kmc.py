from abc import ABC

import numpy as np

from .model import KoopmanModel, LinearModel


class BaseKMC(ABC):
    """Base for Koopman-MPC controllers.

    Adapts a fitted Koopman model into the internal `KoopmanModel`
    representation used by MPC. The input `model` is duck-typed -- it must
    expose:
        .A           (n_lifted, n_lifted)
        .B           (n_lifted, n_input)
        .n_state     int -- dim of the original x
        .n_output    int -- dim of the observation y
        .observable  None for DMDc, a fitted PolynomialObservable for EDMDc
        .scaler_x, .scaler_u, .scaler_y  fitted sklearn scalers (or None)
    `FittedModel` from script/sysid/validate.py satisfies this interface.
    """

    def __init__(self, model):
        self.model = self._build(model)

    def _build(self, m) -> KoopmanModel:
        n_lifted = m.A.shape[0]

        # Lift: identity for DMDc, observable.transform for EDMDc.
        observable = getattr(m, "observable", None)
        if observable is None:
            def lift(x):
                return x
        else:
            def lift(x):
                if x.ndim == 1:
                    return observable.transform(x.reshape(1, -1)).flatten()
                return observable.transform(x)

        # Output matrix: first n_output components of z are x (C = [I 0]).
        C = np.zeros((m.n_output, n_lifted))
        C[:m.n_output, :m.n_output] = np.eye(m.n_output)

        return KoopmanModel(
            dyn=LinearModel(A=m.A, B=m.B, C=C, D=None),
            lift=lift,
            scaler_x=getattr(m, "scaler_x", None),
            scaler_u=getattr(m, "scaler_u", None),
            scaler_y=getattr(m, "scaler_y", None),
        )
