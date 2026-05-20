"""Koopman-based system identification library for the Xplorer-mini AUV."""

from .dataio import (
    discover_trials,
    inject_euler_angles,
    load_dataset,
    load_trial,
    process_stage,
    to_ned,
)
from .filters import (
    butterworth_lowpass,
    check_outliers_iqr,
    hampel_filter,
    moving_average,
)
from .models import DMDc, EDMDc
from .observable import (
    BaseObservable,
    IdentityObservable,
    PolynomialObservable,
)

__all__ = [
    "DMDc",
    "EDMDc",
    "BaseObservable",
    "IdentityObservable",
    "PolynomialObservable",
    "discover_trials",
    "load_trial",
    "load_dataset",
    "inject_euler_angles",
    "process_stage",
    "to_ned",
    "check_outliers_iqr",
    "hampel_filter",
    "moving_average",
    "butterworth_lowpass",
]