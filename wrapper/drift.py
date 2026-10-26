"""Drift diagnostics: binned total-variation estimates, conditional drift from the certified state, jackknife, split-bank tau, initial-state TV, permutation null."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def n_cells_sqrt(m: int) -> int:
    return max(1, math.isqrt(m))


def binned_tv(p: np.ndarray, q: np.ndarray, n_cells: int) -> float:
    """Plug-in total-variation distance between two empirical score laws on [0,1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p[~np.isnan(p)]
    q = q[~np.isnan(q)]
    if p.size == 0 or q.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_cells + 1)
    hp, _ = np.histogram(np.clip(p, 0.0, 1.0), bins=edges)
    hq, _ = np.histogram(np.clip(q, 0.0, 1.0), bins=edges)
    return 0.5 * float(np.abs(hp / hp.sum() - hq / hq.sum()).sum())


def marginal_drift(bank: np.ndarray, T: int, n_cells: Optional[int] = None) -> np.ndarray:
    """rho_hat_t = binned TV between the step-t and step-1 score laws for t = 1..T (rho_hat_1 = 0)."""
    bank = np.asarray(bank, dtype=float)
    m = bank.shape[0]
    K = n_cells_sqrt(m) if n_cells is None else n_cells
    ref = bank[:, 0]
    rho = np.zeros(T)
    for t in range(2, T + 1):
        if t - 1 >= bank.shape[1]:
            rho[t - 1] = float("nan")
            continue
        rho[t - 1] = binned_tv(bank[:, t - 1], ref, K)
    return rho


def marginal_drift_null(bank: np.ndarray, T: int, n_perms: int, rng: np.random.Generator,
                        n_cells: Optional[int] = None) -> float:
    """Permutation null for marginal_drift: shuffle step indices within each trajectory."""
    bank = np.asarray(bank, dtype=float)
    m = bank.shape[0]
    K = n_cells_sqrt(m) if n_cells is None else n_cells
    vals = []
    for _ in range(n_perms):
        perm_bank = bank.copy()
        for i in range(m):
            present = ~np.isnan(perm_bank[i])
            n_p = int(present.sum())
            if n_p > 1:
                valid = perm_bank[i, present].copy()
                rng.shuffle(valid)
                perm_bank[i, present] = valid
        vals.append(rho_scalar(marginal_drift(perm_bank, T, K)))
    return float(np.mean(vals)) if vals else float("nan")


def rho_scalar(rho_t: np.ndarray) -> float:
    """Mean of rho_hat_t over t >= 2."""
    r = np.asarray(rho_t, dtype=float)[1:]
    r = r[~np.isnan(r)]
    return float(r.mean()) if r.size else 0.0


def s_hat_T(rho_scalar_value: float, T: int) -> float:
    """Cumulative drift estimate T * rho_hat used by the calibratability gate and the DC schedule."""
    return float(T * rho_scalar_value)


@dataclass
class ConditionalDrift:
    eta_cert: float
    k_ba: float
    k_bb: float
    mu_b: float
    n_from_a: int
    n_from_b: int


def conditional_drift_cert(bank: np.ndarray, q: float, cal_scores: np.ndarray,
                           T: int) -> ConditionalDrift:
    """Conditional drift from the certified state on the two-state partition s <= q / s > q."""
    bank = np.asarray(bank, dtype=float)
    cur = bank[:, : T - 1]
    nxt = bank[:, 1:T]
    present = ~np.isnan(cur) & ~np.isnan(nxt)
    cur_b = cur > q
    nxt_b = nxt > q
    from_a = present & ~cur_b
    n_a = int(from_a.sum())
    k_ba = float((from_a & nxt_b).sum() / n_a) if n_a else float("nan")
    from_b = present & cur_b
    n_b = int(from_b.sum())
    k_bb = float((from_b & nxt_b).sum() / n_b) if n_b else float("nan")
    mu_b = float(np.mean(np.asarray(cal_scores, dtype=float) > q))
    eta = abs(k_ba - mu_b) if n_a else float("nan")
    return ConditionalDrift(eta_cert=eta, k_ba=k_ba, k_bb=k_bb, mu_b=mu_b,
                            n_from_a=n_a, n_from_b=n_b)


def eta_cert_iid_null(n_cal: int, k_rank: int, n_from_a: int = 0) -> float:
    """Analytic i.i.d. null for eta_cert: E[|X - mu_b|] where X ~ N(mu_beta, sd^2)."""
    a, b = n_cal + 1 - k_rank, k_rank
    mu_beta = a / (a + b)
    sd_cal = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    mu_b_fixed = (n_cal - k_rank) / n_cal
    sd_bank = math.sqrt(mu_beta * (1.0 - mu_beta) / max(n_from_a, 1)) if n_from_a > 0 else 0.0
    sd = math.sqrt(sd_cal ** 2 + sd_bank ** 2)
    d = abs(mu_beta - mu_b_fixed)
    if sd < 1e-15:
        return d
    return sd * math.sqrt(2.0 / math.pi) * math.exp(-d * d / (2.0 * sd * sd)) + d * math.erf(d / (sd * math.sqrt(2.0)))


def parse_failure_rate(failed: np.ndarray, scores: Optional[np.ndarray] = None) -> float:
    """Per-step parse-failure rate across all present steps (steps whose score is not NaN when scores are given)."""
    f = np.asarray(failed, dtype=bool)
    if scores is not None:
        f = f[~np.isnan(np.asarray(scores, dtype=float))]
    return float(f.mean()) if f.size else float("nan")


def jackknife_debias(bank: np.ndarray, T: int, n_cells: Optional[int] = None) -> np.ndarray:
    """Jackknife bias-corrected rho_hat_t: rho_hat - (m-1)(mean_i rho_hat^{(-i)} - rho_hat)."""
    bank = np.asarray(bank, dtype=float)
    m = bank.shape[0]
    K = n_cells_sqrt(m) if n_cells is None else n_cells
    full = marginal_drift(bank, T, K)
    loo = np.zeros((m, T))
    for i in range(m):
        sub = np.delete(bank, i, axis=0)
        loo[i] = marginal_drift(sub, T, K)
    bias = (m - 1) * (np.nanmean(loo, axis=0) - full)
    return np.clip(full - bias, 0.0, 1.0)


def split_bank_tau(bank: np.ndarray, T: int, n_splits: int, rng: np.random.Generator,
                   n_cells: Optional[int] = None) -> float:
    """Per-step discrepancy |rho_hat_t^A - rho_hat_t^B| between random half-banks, averaged over t >= 2 and over splits."""
    bank = np.asarray(bank, dtype=float)
    m = bank.shape[0]
    K = n_cells_sqrt(m) if n_cells is None else n_cells
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(m)
        A, B = bank[perm[: m // 2]], bank[perm[m // 2:]]
        ra = marginal_drift(A, T, K)[1:]
        rb = marginal_drift(B, T, K)[1:]
        d = np.abs(ra - rb)
        d = d[~np.isnan(d)]
        if d.size:
            vals.append(float(d.mean()))
    return float(np.mean(vals)) if vals else float("nan")


def initial_state_tv(deploy_initial_scores: np.ndarray, cal_scores: np.ndarray,
                     n_cells: Optional[int] = None) -> float:
    """Delta_0 = binned TV between the deployment and calibration initial-state score laws."""
    d = np.asarray(deploy_initial_scores, dtype=float)
    K = n_cells_sqrt(d.size) if n_cells is None else n_cells
    return binned_tv(d, np.asarray(cal_scores, dtype=float), K)


def bootstrap_se(stat: Callable[[np.ndarray], float], bank: np.ndarray, n_resamples: int,
                 rng: np.random.Generator) -> float:
    bank = np.asarray(bank, dtype=float)
    m = bank.shape[0]
    vals = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, m, size=m)
        vals[r] = stat(bank[idx])
    return float(np.nanstd(vals, ddof=1))
