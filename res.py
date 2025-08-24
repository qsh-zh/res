"""
A general framework for various sampling algorithm from a diffusion model.
Impl based on
* Refined Exponential Solver (RES) in https://arxiv.org/pdf/2308.02157
* also clude other impl, DDIM, DEIS, DPM-Solver, EDM sampler.
Most of sampling algorihtm, Runge-Kutta, Multi-step, etc, can be impl in this framework by \
    adding new step function in get_runge_kutta_fn or get_multi_step_fn.
"""

import math
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
import torch
from torch import Tensor
from dataclasses import dataclass


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(axis)

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y

def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x * y

##################################################### multi-step solver #####################################################

def phi1(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the first order phi function: (exp(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi1 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return (torch.expm1(t) / t).to(dtype=input_dtype)


def phi2(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the second order phi function: (phi1(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi2 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return ((phi1(t) - 1.0) / t).to(dtype=input_dtype)


def res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a residual-based 2nd order Runge-Kutta step.

    Args:
        x_s: Current state tensor.
        t: Target time tensor.
        s: Current time tensor.
        x0_s: Prediction at current time.
        s1: Intermediate time tensor.
        x0_s1: Prediction at intermediate time.

    Returns:
        Tensor: Updated state tensor.

    Raises:
        AssertionError: If step size is too small.
    """
    s = -torch.log(s)
    t = -torch.log(t)
    m = -torch.log(s1)

    dt = t - s
    assert not torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"
    assert not torch.any(torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"

    c2 = (m - s) / dt
    phi1_val, phi2_val = phi1(-dt), phi2(-dt)

    # Handle edge case where t = s = m
    b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
    b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

    return batch_mul(torch.exp(-dt), x_s) + batch_mul(dt, batch_mul(b1, x0_s) + batch_mul(b2, x0_s1))


def reg_x0_euler_step(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on x0 prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_s: Prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current prediction.
    """
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return batch_mul(coef_x0, x0_s) + batch_mul(coef_xs, x_s), x0_s


def reg_eps_euler_step(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, eps_s: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on epsilon prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        eps_s: Epsilon prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current x0 prediction.
    """
    return x_s + batch_mul(eps_s, t - s), x_s + batch_mul(eps_s, 0 - s)


def rk1_euler(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a first-order Runge-Kutta (Euler) step.

    Recommended for diffusion models with guidance or model undertrained
    Usually more stable at the cost of a bit slower convergence.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    x0_s = x0_fn(x_s, s)
    return reg_x0_euler_step(x_s, s, t, x0_s)


def rk2_mid_stable(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a stable second-order Runge-Kutta (midpoint) step.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    s1 = torch.sqrt(s * t)
    x_s1, _ = rk1_euler(x_s, s, s1, x0_fn)

    x0_s1 = x0_fn(x_s1, s1)
    return reg_x0_euler_step(x_s, s, t, x0_s1)


def rk2_mid(x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a second-order Runge-Kutta (midpoint) step.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    s1 = torch.sqrt(s * t)
    x_s1, x0_s = rk1_euler(x_s, s, s1, x0_fn)

    x0_s1 = x0_fn(x_s1, s1)

    return res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1), x0_s1


def rk_2heun_naive(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive second-order Runge-Kutta (Heun's method) step.
    Impl based on rho-rk-deis solvers, https://github.com/qsh-zh/deis
    Recommended for diffusion models without guidance and relative large NFE

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    x_t, x0_s = rk1_euler(x_s, s, t, x0_fn)
    eps_s = batch_mul(1.0 / s, x_t - x0_s)
    x0_t = x0_fn(x_t, t)
    eps_t = batch_mul(1.0 / t, x_t - x0_t)

    avg_eps = (eps_s + eps_t) / 2

    return reg_eps_euler_step(x_s, s, t, avg_eps)


def rk_2heun_edm(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive second-order Runge-Kutta (Heun's method) step.
    Impl based no EDM second order Heun method

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    x_t, x0_s = rk1_euler(x_s, s, t, x0_fn)
    x0_t = x0_fn(x_t, t)

    avg_x0 = (x0_s + x0_t) / 2

    return reg_x0_euler_step(x_s, s, t, avg_x0)


def rk_3kutta_naive(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive third-order Runge-Kutta step.
    Impl based on rho-rk-deis solvers, https://github.com/qsh-zh/deis
    Recommended for diffusion models without guidance and relative large NFE

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    c2, c3 = 0.5, 1.0
    a31, a32 = -1.0, 2.0
    b1, b2, b3 = 1.0 / 6, 4.0 / 6, 1.0 / 6

    delta = t - s

    s1 = c2 * delta + s
    s2 = c3 * delta + s
    x_s1, x0_s = rk1_euler(x_s, s, s1, x0_fn)
    eps_s = batch_mul(1.0 / s, x_s - x0_s)
    x0_s1 = x0_fn(x_s1, s1)
    eps_s1 = batch_mul(1.0 / s1, x_s1 - x0_s1)

    _eps = a31 * eps_s + a32 * eps_s1
    x_s2, _ = reg_eps_euler_step(x_s, s, s2, _eps)

    x0_s2 = x0_fn(x_s2, s2)
    eps_s2 = batch_mul(1.0 / s2, x_s2 - x0_s2)

    avg_eps = b1 * eps_s + b2 * eps_s1 + b3 * eps_s2
    return reg_eps_euler_step(x_s, s, t, avg_eps)


# key : order + name
RK_FNs = {
    "1euler": rk1_euler,
    "2mid": rk2_mid,
    "2mid_stable": rk2_mid_stable,
    "2heun_edm": rk_2heun_edm,
    "2heun_naive": rk_2heun_naive,
    "3kutta_naive": rk_3kutta_naive,
}


def get_runge_kutta_fn(name: str) -> Callable:
    """
    Get the specified Runge-Kutta function.

    Args:
        name: Name of the Runge-Kutta method.

    Returns:
        Callable: The specified Runge-Kutta function.

    Raises:
        RuntimeError: If the specified method is not supported.
    """
    if name in RK_FNs:
        return RK_FNs[name]
    methods = "\n\t".join(RK_FNs.keys())
    raise RuntimeError(f"Only support the following Runge-Kutta methods:\n\t{methods}")


def is_runge_kutta_fn_supported(name: str) -> bool:
    """
    Check if the specified Runge-Kutta function is supported.

    Args:
        name: Name of the Runge-Kutta method.

    Returns:
        bool: True if the method is supported, False otherwise.
    """
    return name in RK_FNs


##################################################### multi-step solver #####################################################
"""
Impl of multistep methods to solve the ODE in the diffusion model.
"""


def order2_fn(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_s: torch.Tensor, x0_preds: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    impl the second order multistep method in https://arxiv.org/pdf/2308.02157
    Adams Bashforth approach!
    """
    if x0_preds:
        x0_s1, s1 = x0_preds[0]
        x_t = res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1)
    else:
        x_t = reg_x0_euler_step(x_s, s, t, x0_s)[0]
    return x_t, [(x0_s, s)]


# key: method name, value: method function
# key: order + algorithm name
MULTISTEP_FNs = {
    "2ab": order2_fn,
}


def get_multi_step_fn(name: str) -> Callable:
    if name in MULTISTEP_FNs:
        return MULTISTEP_FNs[name]
    methods = "\n\t".join(MULTISTEP_FNs.keys())
    raise RuntimeError("Only support multistep method\n" + methods)


def is_multi_step_fn_supported(name: str) -> bool:
    """
    Check if the multistep method is supported.
    """
    return name in MULTISTEP_FNs


##################################################### sampler impl #####################################################

COMMON_SOLVER_OPTIONS = Literal["2ab", "2mid", "1euler"]


@dataclass
class SolverConfig:
    is_multi: bool = False
    rk: str = "2mid"
    multistep: str = "2ab"
    # following parameters control stochasticity, see EDM paper
    # BY default, we use deterministic with no stochasticity
    s_churn: float = 0.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.05
    s_noise: float = 1.0


@dataclass
class SolverTimestampConfig:
    nfe: int = 50
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False  # whether generate forward or backward timestamps


@dataclass
class SamplerConfig:
    solver: SolverConfig = SolverConfig()
    timestamps: SolverTimestampConfig = SolverTimestampConfig()
    sample_clean: bool = True  # whether run one last step to generate clean image


def get_rev_ts(
    t_min: float, t_max: float, num_steps: int, ts_order: Union[int, float], is_forward: bool = False
) -> torch.Tensor:
    """
    Generate a sequence of reverse time steps.

    Args:
        t_min (float): The minimum time value.
        t_max (float): The maximum time value.
        num_steps (int): The number of time steps to generate.
        ts_order (Union[int, float]): The order of the time step progression.
        is_forward (bool, optional): If True, returns the sequence in forward order. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the generated time steps in reverse or forward order.

    Raises:
        ValueError: If `t_min` is not less than `t_max`.
        TypeError: If `ts_order` is not an integer or float.
    """
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")

    if not isinstance(ts_order, (int, float)):
        raise TypeError("ts_order must be an integer or float")

    step_indices = torch.arange(num_steps + 1, dtype=torch.float64)
    time_steps = (
        t_max ** (1 / ts_order) + step_indices / num_steps * (t_min ** (1 / ts_order) - t_max ** (1 / ts_order))
    ) ** ts_order

    if is_forward:
        return time_steps.flip(dims=(0,))

    return time_steps


class Sampler(torch.nn.Module):
    def __init__(self, cfg: Optional[SamplerConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = SamplerConfig()
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        solver_option: str = "2ab",
    ) -> torch.Tensor:
        in_dtype = x_sigma_max.dtype

        def float64_x0_fn(x_B_StateShape: torch.Tensor, t_B: torch.Tensor) -> torch.Tensor:
            return x0_fn(x_B_StateShape.to(in_dtype), t_B.to(in_dtype)).to(torch.float64)

        is_multistep = is_multi_step_fn_supported(solver_option)
        is_rk = is_runge_kutta_fn_supported(solver_option)
        assert is_multistep or is_rk, f"Only support multistep or Runge-Kutta method, got {solver_option}"

        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=is_multistep,
            rk=solver_option,
            multistep=solver_option,
        )
        timestamps_cfg = SolverTimestampConfig(nfe=num_steps, t_min=sigma_min, t_max=sigma_max, order=rho)
        sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=True)

        return self._forward_impl(float64_x0_fn, x_sigma_max, sampler_cfg).to(in_dtype)

    @torch.no_grad()
    def _forward_impl(
        self,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noisy_input_B_StateShape: torch.Tensor,
        sampler_cfg: Optional[SamplerConfig] = None,
        callback_fns: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """
        Internal implementation of the forward pass.

        Args:
            denoiser_fn: Function to denoise the input.
            noisy_input_B_StateShape: Input tensor with noise.
            sampler_cfg: Configuration for the sampler.
            callback_fns: List of callback functions to be called during sampling.

        Returns:
            torch.Tensor: Denoised output tensor.
        """
        sampler_cfg = self.cfg if sampler_cfg is None else sampler_cfg
        solver_order = 1 if sampler_cfg.solver.is_multi else int(sampler_cfg.solver.rk[0])
        num_timestamps = sampler_cfg.timestamps.nfe // solver_order

        sigmas_L = get_rev_ts(
            sampler_cfg.timestamps.t_min, sampler_cfg.timestamps.t_max, num_timestamps, sampler_cfg.timestamps.order
        ).to(noisy_input_B_StateShape.device)

        denoised_output = differential_equation_solver(
            denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns
        )(noisy_input_B_StateShape)

        if sampler_cfg.sample_clean:
            # Override denoised_output with fully denoised version
            ones = torch.ones(denoised_output.size(0), device=denoised_output.device, dtype=denoised_output.dtype)
            denoised_output = denoiser_fn(denoised_output, sigmas_L[-1] * ones)

        return denoised_output


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, Any], Any], init_val: Any) -> Any:
    """
    Implements a for loop with a function.

    Args:
        lower: Lower bound of the loop (inclusive).
        upper: Upper bound of the loop (exclusive).
        body_fun: Function to be applied in each iteration.
        init_val: Initial value for the loop.

    Returns:
        The final result after all iterations.
    """
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def differential_equation_solver(
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigmas_L: torch.Tensor,
    solver_cfg: SolverConfig,
    callback_fns: Optional[List[Callable]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a differential equation solver function.

    Args:
        x0_fn: Function to compute x0 prediction.
        sigmas_L: Tensor of sigma values with shape [L,].
        solver_cfg: Configuration for the solver.
        callback_fns: Optional list of callback functions.

    Returns:
        A function that solves the differential equation.
    """
    num_step = len(sigmas_L) - 1

    if solver_cfg.is_multi:
        update_step_fn = get_multi_step_fn(solver_cfg.multistep)
    else:
        update_step_fn = get_runge_kutta_fn(solver_cfg.rk)

    eta = min(solver_cfg.s_churn / (num_step + 1), math.sqrt(1.2) - 1)

    def sample_fn(input_xT_B_StateShape: torch.Tensor) -> torch.Tensor:
        """
        Samples from the differential equation.

        Args:
            input_xT_B_StateShape: Input tensor with shape [B, StateShape].

        Returns:
            Output tensor with shape [B, StateShape].
        """
        ones_B = torch.ones(input_xT_B_StateShape.size(0), device=input_xT_B_StateShape.device, dtype=torch.float64)

        def step_fn(
            i_th: int, state: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
        ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
            input_x_B_StateShape, x0_preds = state
            sigma_cur_0, sigma_next_0 = sigmas_L[i_th], sigmas_L[i_th + 1]

            # algorithm 2: line 4-6
            if solver_cfg.s_t_min < sigma_cur_0 < solver_cfg.s_t_max:
                hat_sigma_cur_0 = sigma_cur_0 + eta * sigma_cur_0
                input_x_B_StateShape = input_x_B_StateShape + (
                    hat_sigma_cur_0**2 - sigma_cur_0**2
                ).sqrt() * solver_cfg.s_noise * torch.randn_like(input_x_B_StateShape)
                sigma_cur_0 = hat_sigma_cur_0

            if solver_cfg.is_multi:
                x0_pred_B_StateShape = x0_fn(input_x_B_StateShape, sigma_cur_0 * ones_B)
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_pred_B_StateShape, x0_preds
                )
            else:
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_fn
                )

            if callback_fns:
                for callback_fn in callback_fns:
                    callback_fn(**locals())

            return output_x_B_StateShape, x0_preds

        x_at_eps, _ = fori_loop(0, num_step, step_fn, [input_xT_B_StateShape, None])
        return x_at_eps

    return sample_fn
