from abc import abstractmethod
from dataclasses import dataclass
import textwrap

import numpy as np
import tabulate

from ...core.kmc import BaseKMC
from ...core.model import KoopmanModel
from ...core.params import Weights, Bounds


@dataclass
class MPCParams:
    """
    Parameters for MPC setup.
        dt: Discretization time step
        N_horizon: Prediction horizon length
    """
    dt: float
    N_horizon: int
    weights: Weights
    bounds: Bounds


class KMPC(BaseKMC):
    
    """
    Base class for Koopman-based MPC controllers.
    Subclasses must implement the compute_control method to calculate the control input based on the current state and reference.
    
    Attributes:
        mpc_params: MPCParams - Parameters for MPC setup, including cost weights and horizon length.
        model: KoopmanModel - The Koopman model of the system, containing the system dynamics and lifting function.

    Methods:
        set_params(**kwargs): Update MPC parameters dynamically. Accepts any parameter defined in MPCParams or its nested Weights dataclass.
        compute_control(x, y_ref): Abstract method to compute the control input based on the current state x and reference output y_ref. Must be implemented by subclasses.
    """

    mpc_params: MPCParams
    model: KoopmanModel

    def __init__(self, model, mpc_params: MPCParams):

        super().__init__(model)
        self.mpc_params = mpc_params
        self.print_koopman_summary(model)

    def print_koopman_summary(self, model):
        """
        Detailed technical summary of Koopman structure and per-state scaler statistics
        using mathematical variable naming (x1, x2, tau1, tau2...). Discriminates
        DMDc vs EDMDc by whether the fitted model has an `observable` attribute.
        """
        # 1. Model Structure Table
        print(f"\n{'='*25} KOOPMAN MODEL SUMMARY {'='*25}")

        observable = getattr(model, "observable", None)
        if observable is None:
            wrapper_type = "DMDc"
            n_states = self.model.dyn.A.shape[0]
            obs_list = [f"x{i+1}" for i in range(n_states)]
        else:
            wrapper_type = "EDMDc"
            obs_list = observable.get_output_names()
        obs_func_str = textwrap.fill(", ".join(obs_list), width=60)

        model_summary = [
            ["Model Type", wrapper_type],
            ["Observables", obs_func_str],
            ["A (System Matrix)", f"{self.model.dyn.A.shape[0]}x{self.model.dyn.A.shape[1]}"],
            ["B (Input Matrix)", f"{self.model.dyn.B.shape[0]}x{self.model.dyn.B.shape[1]}"],
            ["C (Output Matrix)", f"{self.model.dyn.C.shape[0]}x{self.model.dyn.C.shape[1]}"],
            ["Horizon (N)", self.mpc_params.N_horizon],
            ["Sampling (dt)", f"{self.mpc_params.dt} s"]
        ]
        print("\n" + tabulate.tabulate(model_summary, headers=["Property", "Configuration"], tablefmt="fancy_grid"))

        # 2. Detailed Scaler Tables
        print(f"\n{'='*25} NORMALIZATION SUMMARY {'='*25}")
        scaler_configs = [
            ("STATE (X)", self.model.scaler_x, "x"), 
            ("INPUT (U)", self.model.scaler_u, "tau"),
            ("OUTPUT (Y)", getattr(self.model, 'scaler_y', None), "y")
        ]

        for label, scaler, sym in scaler_configs:
            if scaler is not None and hasattr(scaler, 'mean_'):
                rows = []
                has_feature_names = hasattr(scaler, 'feature_names_in_')
                
                for i in range(len(scaler.mean_)):
                    if has_feature_names:
                        var_name = scaler.feature_names_in_[i]
                    else:
                        var_name = f"{sym}{i+1}"
                        
                    rows.append([var_name, f"{scaler.mean_[i]:.6f}", f"{scaler.scale_[i]:.6f}"])
                
                print(f"\n{label} Scaler Details:", flush=True)
                print(tabulate.tabulate(rows, headers=["Feature Name", "Mean", "Std"], tablefmt="fancy_grid"), flush=True)
            else:
                print(f"\n{label} Scaler: Not Found / Identity Transformation", flush=True)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.mpc_params, key):
                setattr(self.mpc_params, key, value)
            elif hasattr(self.mpc_params.weights, key):
                setattr(self.mpc_params.weights, key, value)
            elif hasattr(self.mpc_params.bounds, key):
                setattr(self.mpc_params.bounds, key, value)
            else:
                raise ValueError(f"Invalid parameter name: {key}")

    @abstractmethod
    def compute_control(self, x: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement compute_control method.")


    
