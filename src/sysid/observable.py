"""Koopman observable (lifting) functions used by EDMDc."""

from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


class BaseObservable(TransformerMixin):
    """Base class for Koopman observable functions."""

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        raise NotImplementedError("fit method not implemented.")

    def transform(self, X):
        raise NotImplementedError("transform method not implemented.")

    def get_output_names(self) -> list[str]:
        raise NotImplementedError("get_output_names method not implemented.")


class IdentityObservable(BaseObservable):
    """Identity lifting: psi(x) = x. Recovers plain DMDc behaviour from EDMDc."""

    def __init__(self):
        super().__init__()
        self.n_features_in_ = None

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X

    def get_output_names(self) -> list[str]:
        if self.n_features_in_ is None:
            raise RuntimeError("IdentityObservable must be fitted before calling get_output_names")
        return [f"x{i}" for i in range(self.n_features_in_)]


class PolynomialObservable(BaseObservable):
    """Polynomial lifting via sklearn's PolynomialFeatures."""

    def __init__(self, degree: int = 2, interaction_only: bool = False, include_bias: bool = False):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.poly_transformer_ = None
        self.n_input_features_ = None

    def fit(self, X, y=None):
        self.poly_transformer_ = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only,
        )
        self.poly_transformer_.fit(X)
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X):
        if self.poly_transformer_ is None:
            raise RuntimeError("PolynomialObservable must be fitted before calling transform.")
        return self.poly_transformer_.transform(X)

    def get_output_names(self) -> list[str]:
        if self.poly_transformer_ is None:
            raise RuntimeError("PolynomialObservable must be fitted before calling get_output_names.")
        input_names = [f"x{i}" for i in range(self.n_input_features_)]
        return self.poly_transformer_.get_feature_names_out(input_names).tolist()
