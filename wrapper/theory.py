"""Closed-form coverage bounds and the two-state constructive kernels with their continuous score lift."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def dkw_slack(n: int, delta: float) -> float:
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def upper_bound(T: int, eps: float, rho_sum: float = 0.0, n: Optional[int] = None,
                delta: float = 0.05) -> float:
    slack = 0.0 if n is None else T * dkw_slack(n, delta)
    return T * eps + rho_sum + slack


def r1_lower_bound(T: int, eps: float) -> float:
    return 1.0 - (1.0 - eps) ** T


def r2_lower_bound(T: int, eps: float, rho: float) -> float:
    if not (0.0 <= rho < 1.0 - eps):
        raise ValueError("rho must lie in [0, 1-eps)")
    return 1.0 - (1.0 - eps) * (1.0 - eps - rho) ** (T - 1)


def r3_lower_bound(T: int, eps: float) -> float:
    return 1.0 - (1.0 - eps) * ((1.0 - 2.0 * eps) / (1.0 - eps)) ** (T - 1)


def wrapper_bound(T: int, eps: float, rho_sum: float = 0.0, n: Optional[int] = None,
                  delta: float = 0.05) -> float:
    slack = 0.0 if n is None else T * dkw_slack(n, delta)
    return eps + rho_sum + slack


def wrapper_dc_bound(T: int, eps: float, tau: float, n: Optional[int] = None,
                     delta: float = 0.05) -> float:
    slack = 0.0 if n is None else T * dkw_slack(n, delta)
    return eps + tau + slack


def implied_iid_rate(mis_T: float, T: int) -> float:
    if not (0.0 <= mis_T < 1.0):
        raise ValueError("mis_T must be in [0,1)")
    return 1.0 - (1.0 - mis_T) ** (1.0 / T)


def k_diamond_exit_rate(eps: float) -> float:
    """K_diamond(b|a) = eps / (1 - eps)."""
    return eps / (1.0 - eps)


def mu_cal_two_state(eps: float) -> np.ndarray:
    return np.array([1.0 - eps, eps])


def kernel_iid(eps: float) -> np.ndarray:
    mu = mu_cal_two_state(eps)
    return np.vstack([mu, mu])


def kernel_memoryless(eps: float, rho: float) -> np.ndarray:
    if not (0.0 <= rho < 1.0 - eps):
        raise ValueError("rho must lie in [0, 1-eps)")
    nu = np.array([1.0 - eps - rho, eps + rho])
    return np.vstack([nu, nu])


def kernel_two_state(eps: float, p_bb: float) -> np.ndarray:
    if not (0.0 <= p_bb <= 1.0):
        raise ValueError("p_bb must be in [0,1]")
    k_ba = eps * (1.0 - p_bb) / (1.0 - eps)
    return np.array([[1.0 - k_ba, k_ba], [1.0 - p_bb, p_bb]])


def kernel_diamond(eps: float) -> np.ndarray:
    return kernel_two_state(eps, 0.0)


def conditional_drift_two_state(eps: float, p_bb: float) -> float:
    return eps * abs(eps - p_bb) / (1.0 - eps)


def simulate_two_state(kernel: np.ndarray, mu0: np.ndarray, T: int, n_traj: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Simulate n_traj trajectories of length T; returns (n_traj, T) int array of states."""
    states = np.empty((n_traj, T), dtype=int)
    states[:, 0] = rng.random(n_traj) < mu0[1]
    for t in range(1, T):
        p_b = kernel[states[:, t - 1], 1]
        states[:, t] = rng.random(n_traj) < p_b
    return states


def continuous_lift_scores(states: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(states.shape)
    return np.where(states == 1, (1.0 - eps) + eps * u, (1.0 - eps) * u)
