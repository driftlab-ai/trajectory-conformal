"""Entry point: closed-form bounds, synthetic kernels, and the experiment stages."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import wrapper.theory as Th
from evaluation.experiments import (evaluate_methods, load_configs, run_aggregate, run_all, run_drift,
                                    run_evaluate, run_generate, run_horizon, run_mechanism, run_prm)


def print_bounds(horizons, eps, n, delta):
    print("T   R1_LB  R2_LB(rho=.05)  R3_LB  upper(rho=0)  wrapper(rho=0)   [%]")
    for T in horizons:
        print(f"{T:<3} {100 * Th.r1_lower_bound(T, eps):6.2f} {100 * Th.r2_lower_bound(T, eps, 0.05):14.2f} "
              f"{100 * Th.r3_lower_bound(T, eps):6.2f} {100 * Th.upper_bound(T, eps, 0.0, n, delta):12.1f} "
              f"{100 * Th.wrapper_bound(T, eps, 0.0, n, delta):14.1f}")


def run_synthetic(horizons, eps, n_cal, n_seeds, n_test, seed):
    """Evaluate per-step CP, Bonferroni-T and TC-Wrapper on trajectories drawn from the two-state kernels."""
    kernels = [("iid", Th.kernel_iid(eps)), ("memoryless rho=0.05", Th.kernel_memoryless(eps, 0.05)),
               ("k_diamond", Th.kernel_diamond(eps))]
    print("kernel               T   perstep  bonferroni  tcw   R1_LB  R3_LB   [%]")
    for name, K in kernels:
        for T in horizons:
            rng = np.random.default_rng(seed)
            cal_pool = rng.uniform(size=n_cal * n_seeds)
            states = Th.simulate_two_state(K, Th.mu_cal_two_state(eps), T, n_test, rng)
            scores = Th.continuous_lift_scores(states, eps, rng)
            res = evaluate_methods(cal_pool, scores, np.zeros_like(scores, dtype=bool), eps, T, n_cal, n_seeds, seed)
            print(f"{name:20s} {T:<3} {100 * res['perstep']['mean']:7.1f} {100 * res['bonferroni']['mean']:10.1f} "
                  f"{100 * res['tcw']['mean']:6.1f} {100 * Th.r1_lower_bound(T, eps):6.2f} "
                  f"{100 * Th.r3_lower_bound(T, eps):6.2f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="cmd")

    def add_common(sp):
        sp.add_argument("-c", "--config", action="append", required=True, help="YAML config (repeatable; later files override)")
        sp.add_argument("--model", default=None, help="restrict to one model tag from the config")
        sp.add_argument("--no-resume", action="store_true", help="regenerate trajectories already present on disk")

    sp = sub.add_parser("generate", help="generate one trajectory split")
    add_common(sp)
    sp.add_argument("--split", required=True, choices=["cal", "test", "prm_bank", "drift_bank"])
    for name, h in (("prm", "judge-label the PRM bank, train the head, score all splits"),
                    ("evaluate", "calibrate and evaluate per-step CP, Bonferroni-T, CRC, ACI, TC-Wrapper, TC-Wrapper (LIL)"),
                    ("drift", "drift diagnostics, calibratability gate, DC perturbation"),
                    ("mechanism", "context-reset and tool-randomization ablation (needs a `mechanism` config block)"),
                    ("aggregate", "summarise all results under out_dir"),
                    ("run-all", "every stage in order")):
        add_common(sub.add_parser(name, help=h))
    sp = sub.add_parser("horizon", help="horizon-controlled experiment")
    add_common(sp)
    sp.add_argument("--no-generate", action="store_true", help="only re-evaluate existing horizon files")
    for name, h in (("bounds", "closed-form bounds"), ("synthetic", "calibrators on two-state kernel trajectories")):
        sp = sub.add_parser(name, help=h)
        sp.add_argument("--T", type=int, nargs="+", default=[3, 4, 5, 8, 15])
        sp.add_argument("--eps", type=float, default=0.10)
        sp.add_argument("--n", type=int, default=1000)
        sp.add_argument("--delta", type=float, default=0.05)
        sp.add_argument("--seeds", type=int, default=5)
        sp.add_argument("--n-test", type=int, default=20000)
        sp.add_argument("--seed", type=int, default=0)

    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        return 2
    if a.cmd == "bounds":
        print_bounds(a.T, a.eps, a.n, a.delta)
        return 0
    if a.cmd == "synthetic":
        run_synthetic(a.T, a.eps, a.n, a.seeds, a.n_test, a.seed)
        return 0
    cfg = load_configs(a.config)
    if a.cmd == "generate":
        run_generate(cfg, a.split, a.model, resume=not a.no_resume)
    elif a.cmd == "prm":
        run_prm(cfg, a.model)
    elif a.cmd == "evaluate":
        run_evaluate(cfg, a.model)
    elif a.cmd == "drift":
        run_drift(cfg, a.model)
    elif a.cmd == "horizon":
        run_horizon(cfg, a.model, generate=not a.no_generate, resume=not a.no_resume)
    elif a.cmd == "mechanism":
        run_mechanism(cfg, a.model, resume=not a.no_resume)
    elif a.cmd == "aggregate":
        run_aggregate(cfg)
    elif a.cmd == "run-all":
        run_all(cfg, a.model, resume=not a.no_resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
