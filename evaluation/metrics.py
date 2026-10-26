"""Miscoverage, certified prefix, seed statistics, paired bootstrap, Holm adjustment, Wilson, Pearson, ECE, density."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def miscoverage(exceeded: np.ndarray) -> float:
    e = np.asarray(exceeded, dtype=bool)
    return float(e.mean()) if e.size else float("nan")


def certified_prefix(first_exceedance: np.ndarray, horizon_used: np.ndarray) -> float:
    """Mean over trajectories of (first exceedance step - 1) / steps present; 1 when no step exceeds."""
    fe = np.asarray(first_exceedance, dtype=float)
    h = np.asarray(horizon_used, dtype=float)
    h = np.where(h <= 0, 1.0, h)
    frac = np.where(fe > 0, (fe - 1.0) / h, 1.0)
    return float(frac.mean()) if frac.size else float("nan")


def seed_mean_se(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and standard error (sample std / sqrt(n)) over calibration seeds."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def paired_bootstrap_pvalue(a: np.ndarray, b: np.ndarray, n_resamples: int,
                            rng: np.random.Generator, alternative: str = "less") -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have equal shape")
    n = a.size
    d = a - b
    idx = rng.integers(0, n, size=(n_resamples, n))
    dstar = d[idx].mean(axis=1)
    ge = float(((dstar >= 0).sum() + 1) / (n_resamples + 1))
    le = float(((dstar <= 0).sum() + 1) / (n_resamples + 1))
    if alternative == "less":
        return ge
    if alternative == "greater":
        return le
    if alternative == "two-sided":
        return min(1.0, 2.0 * min(ge, le))
    raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adjusted[i] = running
    reject = adjusted <= alpha
    return reject, adjusted


def wilson_halfwidth(p: float, n: int, z: float = 1.959964) -> float:
    denom = 1.0 + z * z / n
    return float(z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def density_near_quantile(scores: np.ndarray, q: float, halfwidth: float = 0.05) -> float:
    s = np.asarray(scores, dtype=float)
    lo, hi = max(0.0, q - halfwidth), min(1.0, q + halfwidth)
    if hi <= lo:
        return float("nan")
    return float(((s >= lo) & (s <= hi)).mean() / (hi - lo))
