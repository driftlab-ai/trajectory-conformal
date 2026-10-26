"""Split-conformal thresholds, level schedules (polynomial, LIL), calibrators (CRC, ACI), and trajectory exceedance events."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


_CEIL_EPS = 1e-9


class Uncalibratable(Exception):
    pass


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((n+1)(1-alpha)), clamped to [1, n]."""
    if n < 1:
        raise ValueError("need at least one calibration score")
    k = math.ceil((n + 1) * (1.0 - alpha) - _CEIL_EPS)
    return int(min(max(k, 1), n))


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError("scores must be 1-D")
    n = s.shape[0]
    k = conformal_rank(n, alpha)
    return float(np.partition(s, k - 1)[k - 1])


def alpha_schedule(eps: float, T: int) -> np.ndarray:
    if T < 1:
        raise ValueError("T must be >= 1")
    t = np.arange(1, T + 1, dtype=float)
    return 6.0 * eps / (math.pi ** 2 * t ** 2)


def _lil_normalizer(kappa: float, n_terms: int = 200000) -> float:
    """Infinite-sum normalizer for the LIL schedule, with integral tail correction."""
    N = n_terms + 1
    t = np.arange(2, N + 1, dtype=float)
    partial = 1.0 + float(np.sum(1.0 / (t * np.log(t) ** (1.0 + kappa))))
    tail = 1.0 / (kappa * math.log(N) ** kappa)
    return partial + tail


def lil_schedule(eps: float, T: int, kappa: float = 1.0) -> np.ndarray:
    """Law-of-the-iterated-logarithm schedule: alpha_t proportional to 1/(t log^{1+kappa}(t))."""
    if T < 1:
        raise ValueError("T must be >= 1")
    t = np.arange(1, T + 1, dtype=float)
    log_t = np.where(t >= 2, np.log(t), 1.0)
    raw = np.where(t >= 2, 1.0 / (t * log_t ** (1.0 + kappa)), 1.0)
    return eps * raw / _lil_normalizer(kappa)


def alpha_budget(eps: float, T: int) -> float:
    return float(alpha_schedule(eps, T).sum())


def realised_alpha_budget(eps: float, T: int, n: int) -> float:
    """Budget once clamped steps contribute 1/(n+1) each."""
    a = alpha_schedule(eps, T)
    return float(sum(max(float(ai), 1.0 / (n + 1)) for ai in a))


def dc_schedule(alphas: np.ndarray, s_hat_T: float, eps: float) -> np.ndarray:
    if s_hat_T >= eps:
        raise Uncalibratable(
            f"observed cumulative drift S_hat_T={s_hat_T:.4g} exceeds the available "
            f"budget eps={eps:.4g}: abstain-uncalibratable"
        )
    return np.asarray(alphas, dtype=float) * (1.0 - s_hat_T / eps)


def gate_passes(s_hat_T: float, eps: float) -> bool:
    return bool(s_hat_T < eps)


METHODS = ("perstep", "bonferroni", "tcw", "tcw_dc", "crc", "tcw_lil")


def thresholds(
    cal_scores: Sequence[float],
    eps: float,
    T: int,
    method: str,
    s_hat_T: Optional[float] = None,
) -> np.ndarray:
    """Per-step threshold vector (length T) for a calibrator."""
    if method == "perstep":
        return np.full(T, conformal_quantile(cal_scores, eps))
    if method == "bonferroni":
        return np.full(T, conformal_quantile(cal_scores, eps / T))
    if method == "crc":
        s = np.sort(np.asarray(cal_scores, dtype=float))
        n = s.shape[0]
        alpha_crc = ((n + 1) * eps - 1.0) / n
        k = int(min(max(math.ceil(n * (1.0 - alpha_crc) - _CEIL_EPS), 1), n))
        return np.full(T, float(s[k - 1]))
    if method in ("tcw", "tcw_dc"):
        alphas = alpha_schedule(eps, T)
        if method == "tcw_dc":
            if s_hat_T is None:
                raise ValueError("tcw_dc needs s_hat_T")
            alphas = dc_schedule(alphas, s_hat_T, eps)
        return np.array([conformal_quantile(cal_scores, a) for a in alphas])
    if method == "tcw_lil":
        alphas = lil_schedule(eps, T)
        return np.array([conformal_quantile(cal_scores, a) for a in alphas])
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def aci_outcomes(
    cal_scores: Sequence[float],
    scores: np.ndarray,
    eps: float,
    gamma: float = 0.05,
    failed: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> "TrajectoryOutcome":
    """ACI trajectory outcomes with online level adjustment."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    n = cal.shape[0]
    if n < 1:
        raise ValueError("need at least one calibration score")
    s = np.asarray(scores, dtype=float)
    N, T = s.shape
    f = np.zeros((N, T), dtype=bool) if failed is None else np.asarray(failed, dtype=bool)
    lab = np.full((N, T), np.nan) if labels is None else np.asarray(labels, dtype=float)
    alpha_t = np.full(N, eps)
    exceeded = np.zeros(N, dtype=bool)
    first = np.zeros(N, dtype=int)
    horizon_used = np.zeros(N, dtype=int)
    for t in range(T):
        present = ~np.isnan(s[:, t])
        horizon_used += present.astype(int)
        a = np.clip(alpha_t, 1e-9, 1.0 - 1e-9)
        k = np.clip(np.ceil((n + 1) * (1.0 - a) - _CEIL_EPS), 1, n).astype(int)
        step_exceeds = present & ((s[:, t] > cal[k - 1]) | f[:, t])
        err = np.where(np.isnan(lab[:, t]), step_exceeds, lab[:, t] < 0.5).astype(float)
        first = np.where(step_exceeds & ~exceeded, t + 1, first)
        exceeded = exceeded | step_exceeds
        alpha_t = np.where(present, alpha_t + gamma * (eps - err), alpha_t)
    return TrajectoryOutcome(exceeded=exceeded, first_exceedance=first, horizon_used=horizon_used)


@dataclass
class TrajectoryOutcome:
    exceeded: np.ndarray
    first_exceedance: np.ndarray
    horizon_used: np.ndarray


def step_exceedances(
    scores: np.ndarray,
    thresholds_per_step: np.ndarray,
    failed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-step exceedance indicators s(X_t) > q_hat_t on an (N, T) score matrix."""
    s = np.asarray(scores, dtype=float)
    q = np.asarray(thresholds_per_step, dtype=float)
    if s.ndim != 2 or q.ndim != 1 or s.shape[1] != q.shape[0]:
        raise ValueError(f"shape mismatch: scores {s.shape}, thresholds {q.shape}")
    present = ~np.isnan(s)
    exceed = present & (np.nan_to_num(s, nan=-np.inf) > q[None, :])
    if failed is not None:
        exceed = exceed | np.asarray(failed, dtype=bool)
    return exceed


def per_step_exceedance_rate(
    scores: np.ndarray,
    thresholds_per_step: np.ndarray,
    failed: Optional[np.ndarray] = None,
) -> float:
    """Mean per-step exceedance rate across all present steps."""
    ex = step_exceedances(scores, thresholds_per_step, failed=failed)
    present = ~np.isnan(np.asarray(scores, dtype=float))
    n_present = int(present.sum())
    return float(ex.sum() / n_present) if n_present else float("nan")


def trajectory_outcomes(
    scores: np.ndarray,
    thresholds_per_step: np.ndarray,
    failed: Optional[np.ndarray] = None,
) -> TrajectoryOutcome:
    """Trajectory miscoverage event {exists t <= T : s(X_t) > q_hat_t} per trajectory."""
    exceed = step_exceedances(scores, thresholds_per_step, failed=failed)
    present = ~np.isnan(np.asarray(scores, dtype=float))
    horizon_used = present.sum(axis=1).astype(int)
    exceeded = exceed.any(axis=1)
    first = np.where(exceeded, exceed.argmax(axis=1) + 1, 0).astype(int)
    return TrajectoryOutcome(exceeded=exceeded, first_exceedance=first, horizon_used=horizon_used)
