"""Controller factories. Consume nested YAML dicts directly (no ROS shim)."""

from typing import TypeAlias, Union
import inspect

import numpy as np

from ..controller import mpc, pid
from ..core.params import Bounds, Weights


PositionControllerType: TypeAlias = Union[pid.PositionPIDController, pid.PositionFFPIController]
VelocityControllerType: TypeAlias = Union[pid.VelocityPIDController,
                                          mpc.ConstrainedStandardStateForm,
                                          mpc.ConstrainedStandardOutputForm,
                                          mpc.ConstrainedIntegralStateForm]


def _arr(val):
    return np.array(val) if val is not None else None


def _extract_from_signature(cls, source: dict) -> dict:
    """Pop and return keys from `source` whose names appear in cls.__init__."""
    keys = inspect.signature(cls).parameters.keys()
    return {k: np.array(source.pop(k)) for k in keys if k in source}


def create_position_controller(cfg: dict, logger=None) -> PositionControllerType:
    """Build outer-loop position controller from a yaml config dict.

    Expected shape (nested):
        {'type': 'pid' | 'ff_pi', 'params': {...kwargs...}, ...}
    """
    ctrl_type = cfg['type']
    params = dict(cfg.get('params', {}))

    if ctrl_type == 'pid':
        return pid.PositionPIDController(**params)
    if ctrl_type == 'ff_pi':
        return pid.PositionFFPIController(**params)
    raise ValueError(f"Unsupported position controller type: {ctrl_type!r}")


def create_velocity_controller(cfg: dict, model=None, node_name: str = None,
                               dt: float = None, logger=None) -> VelocityControllerType:
    """Build inner-loop velocity controller from a yaml config dict.

    Expected shape (nested):
        {'type': 'pid' | 'mpc_standard' | 'mpc_integral',
         'mode': 'state_form' | 'output_form',
         'use_preview': bool,
         'params': {N_horizon, Q, Qi, R_abs, R_rate, y_min/max, u_min/max, ...}}
    """
    ctrl_type = cfg['type']
    params = dict(cfg.get('params', {}))

    if ctrl_type == 'pid':
        return pid.VelocityPIDController(**params)

    if 'mpc' not in ctrl_type:
        raise ValueError(f"Unsupported velocity controller type: {ctrl_type!r}")

    mode = cfg.get('mode')
    use_preview = cfg.get('use_preview', False)

    n_horizon = int(params.pop('N_horizon', 10))
    weights_dict = _extract_from_signature(Weights, params)
    bounds_dict = _extract_from_signature(Bounds, params)

    weights = Weights(
        Q=np.diag(weights_dict['Q']) if 'Q' in weights_dict else None,
        Qi=np.diag(weights_dict['Qi']) if 'Qi' in weights_dict else None,
        R_abs=np.diag(weights_dict['R_abs']) if 'R_abs' in weights_dict else None,
        R_rate=np.diag(weights_dict['R_rate']) if 'R_rate' in weights_dict else None,
    )
    bounds = Bounds(
        x_min=_arr(bounds_dict.get('x_min')), x_max=_arr(bounds_dict.get('x_max')),
        y_min=_arr(bounds_dict.get('y_min')), y_max=_arr(bounds_dict.get('y_max')),
        u_min=_arr(bounds_dict.get('u_min')), u_max=_arr(bounds_dict.get('u_max')),
        du_min=_arr(bounds_dict.get('du_min')), du_max=_arr(bounds_dict.get('du_max')),
    )
    mpc_params = mpc.MPCParams(dt=dt, N_horizon=n_horizon, weights=weights, bounds=bounds)

    if 'standard' in ctrl_type:
        if mode == 'state_form':
            return mpc.ConstrainedStandardStateForm(model, mpc_params, node_name=node_name,
                                                    use_preview=use_preview, logger=logger)
        if mode == 'output_form':
            return mpc.ConstrainedStandardOutputForm(model, mpc_params, node_name=node_name,
                                                     use_preview=use_preview, logger=logger)
        raise ValueError(f"Standard MPC requires mode 'state_form' or 'output_form', got: {mode!r}")

    if 'integral' in ctrl_type:
        if mode == 'state_form':
            int_limit = _arr(params.get('int_limit'))
            return mpc.ConstrainedIntegralStateForm(model, mpc_params, node_name=node_name,
                                                    use_preview=use_preview, dt=dt,
                                                    int_limit=int_limit, logger=logger)
        raise NotImplementedError(f"Only state-form integral MPC is supported (got mode={mode!r}).")

    raise ValueError(f"Unsupported MPC ctrl_type: {ctrl_type!r}. "
                     "Supported substrings: 'standard', 'integral'.")
