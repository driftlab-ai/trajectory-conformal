"""Config loading and the experiment stages: generate, prm, evaluate, drift, horizon, mechanism, aggregate, run_all."""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

import evaluation.metrics as M
import wrapper.conformal as C
import wrapper.drift as D
import wrapper.theory as Th
from agents.agent import HFLM, ReActAgent, labels_matrix, steps_matrix
from agents.data import (RETRIEVAL_BENCHMARKS, SPLIT_POOL, SPLIT_SEED_OFFSET, STEP_LABEL_BENCHMARKS, load_pool,
                         make_splits, sample_with_replacement)
from agents.envs import ChainedEnv, GAIAEnv, RandomizedToolEnv, RetrievalQAEnv, WorkArenaEnv
from agents.prm import FrozenEncoder, label_bank, prm_input_text, score_trajectories, train_prm_head
from agents.scores import NLIClusterer

METHODS = ("perstep", "bonferroni", "tcw", "crc", "aci", "tcw_lil")


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, resolving `_base_` (relative to the file) recursively."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base)
        cfg = _deep_update(load_config(base_path), cfg)
    return cfg


def load_configs(paths: Sequence[str]) -> Dict[str, Any]:
    """Load several configs (later files override earlier ones), then apply `by_benchmark` overrides."""
    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = _deep_update(cfg, load_config(p))
    per_bench = cfg.pop("by_benchmark", None) or {}
    if cfg.get("benchmark") in per_bench:
        cfg = _deep_update(cfg, per_bench[cfg["benchmark"]])
    return cfg


def resolve_device(cfg: Dict[str, Any]) -> str:
    dev = cfg.get("device", "auto")
    if dev == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _out_root(cfg: Dict[str, Any]) -> str:
    out = cfg["out_dir"]
    return out if os.path.isabs(out) else os.path.join(REPO_ROOT, out)


def run_dir(cfg: Dict[str, Any], model_tag: str) -> str:
    d = os.path.join(_out_root(cfg), cfg["benchmark"], model_tag)
    os.makedirs(d, exist_ok=True)
    return d


def model_specs(cfg: Dict[str, Any], model_tag: Optional[str]) -> List[Dict[str, Any]]:
    specs = cfg["models"]
    if model_tag is None:
        return specs
    sel = [m for m in specs if m["tag"] == model_tag]
    if not sel:
        raise ValueError(f"model tag {model_tag!r} not in config models {[m['tag'] for m in specs]}")
    return sel


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_env(cfg: Dict[str, Any]):
    b = cfg["benchmark"]
    env_cfg = cfg.get("env", {})
    if b in RETRIEVAL_BENCHMARKS:
        return RetrievalQAEnv(**env_cfg)
    if b == "gaia":
        return GAIAEnv(**env_cfg)
    if b == "workarena":
        return WorkArenaEnv(**env_cfg)
    raise ValueError(f"unknown benchmark {b!r}")


def build_lm(cfg: Dict[str, Any], spec: Dict[str, Any]):
    device = resolve_device(cfg)
    dtype = cfg.get("dtype", "float32") if device != "cpu" else "float32"
    return HFLM(spec["name"], revision=spec.get("revision"), device=device, dtype=dtype,
                max_new_tokens=cfg["agent"]["max_new_tokens"],
                max_prompt_tokens=cfg["agent"]["max_prompt_tokens"],
                device_map=cfg.get("device_map") if device != "cpu" else None)


def build_clusterer(cfg: Dict[str, Any]):
    s = cfg["semant"]
    if s.get("n_samples", 0) <= 0:
        return None
    return NLIClusterer(s["nli_model"], device=resolve_device(cfg), entail_threshold=s["entail_threshold"],
                        max_length=s.get("nli_max_length", 256))


def build_agent(cfg: Dict[str, Any], lm, env, clusterer, context_reset_every: int = 0) -> ReActAgent:
    prompt_path = cfg["prompt_file"]
    if not os.path.isabs(prompt_path):
        prompt_path = os.path.join(REPO_ROOT, prompt_path)
    with open(prompt_path) as f:
        template = f.read()
    s = cfg["semant"]
    return ReActAgent(lm, env, template, max_steps=cfg["max_steps"], n_samples=s["n_samples"],
                      sample_temperature=s["temperature"], clusterer=clusterer,
                      selfeval=bool(cfg.get("selfeval", True)), context_reset_every=context_reset_every)


def build_encoder(cfg: Dict[str, Any]):
    return FrozenEncoder(cfg["prm"]["encoder"], device=resolve_device(cfg))


def results_path(cfg: Dict[str, Any], d: str, stem: str) -> str:
    """`<stem><results_tag>.json` inside a run directory; the tag keeps ablation runs from overwriting the main results."""
    return os.path.join(d, f"{stem}{cfg.get('results_tag', '')}.json")


def load_tasks_and_splits(cfg: Dict[str, Any]):
    tasks = load_pool(cfg["benchmark"], limit=cfg.get("pool_limit", 0), cache_dir=cfg.get("data_cache_dir", ""))
    splits = make_splits(len(tasks), cfg["seed"], cfg["cal_pool_fraction"])
    return tasks, splits


SPLIT_SIZE_KEY = {"cal": "n_cal", "test": "n_test", "prm_bank": "n_prm_bank", "drift_bank": "m_drift"}


def cal_pool_size(cfg: Dict[str, Any]) -> int:
    """Calibration initial states to generate: n_cal_pool if set, else n_cal * n_seeds."""
    return int(cfg.get("n_cal_pool") or cfg["n_cal"] * cfg["n_seeds"])


def _resume_rows(path: str, prefix: str, resume: bool) -> List[Dict[str, Any]]:
    """Rows already present in a JSONL output, if they form a contiguous prefix of this split; an incomplete last line is dropped."""
    if not resume or not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise ValueError(f"{path}: line {i + 1} is not valid JSON; rerun with --no-resume")
            print(f"[{prefix}] dropped an incomplete last line of {path}", flush=True)
            write_jsonl(path, rows)
    for j, r in enumerate(rows):
        if r.get("traj_id") != f"{prefix}-{j}":
            return []
    return rows


def run_generate(cfg: Dict[str, Any], split: str, model_tag: Optional[str] = None,
                 resume: bool = True) -> List[str]:
    """Generate trajectories for one split (calibration trajectories stop after the initial state)."""
    if split not in SPLIT_SIZE_KEY:
        raise ValueError(f"split must be one of {sorted(SPLIT_SIZE_KEY)}")
    tasks, splits = load_tasks_and_splits(cfg)
    n = cal_pool_size(cfg) if split == "cal" else cfg[SPLIT_SIZE_KEY[split]]
    idx = sample_with_replacement(splits[SPLIT_POOL[split]], n, cfg["seed"] * 100 + SPLIT_SEED_OFFSET[split])
    max_steps = 1 if split == "cal" else cfg["max_steps"]
    clusterer = None
    outputs = []
    for spec in model_specs(cfg, model_tag):
        path = os.path.join(run_dir(cfg, spec["tag"]), f"{split}.jsonl")
        done = _resume_rows(path, split, resume)
        outputs.append(path)
        if len(done) >= len(idx):
            print(f"[{cfg['benchmark']}/{spec['tag']}/{split}] {len(done)} trajectories already present, skipping",
                  flush=True)
            continue
        if clusterer is None:
            clusterer = build_clusterer(cfg)
        lm = build_lm(cfg, spec)
        env = build_env(cfg)
        agent = build_agent(cfg, lm, env, clusterer)
        if done:
            print(f"[{cfg['benchmark']}/{spec['tag']}/{split}] resuming at trajectory {len(done)}", flush=True)
        new_rows = []
        with open(path, "a" if done else "w") as f:
            for j in range(len(done), len(idx)):
                ti = idx[j]
                traj_seed = cfg["seed"] * 10_000_000 + SPLIT_SEED_OFFSET[split] * 1_000_000 + j
                tr = agent.run(tasks[int(ti)], traj_id=f"{split}-{j}", seed=traj_seed, max_steps=max_steps)
                tr.update(split=split, model=spec["tag"], benchmark=cfg["benchmark"])
                f.write(json.dumps(tr) + "\n")
                f.flush()
                new_rows.append(tr)
                if (j + 1) % 10 == 0 or j + 1 == len(idx):
                    print(f"[{cfg['benchmark']}/{spec['tag']}/{split}] {j + 1}/{len(idx)} trajectories", flush=True)
        tr_rates = truncation_rates(new_rows, max_steps)
        print(f"[{cfg['benchmark']}/{spec['tag']}/{split}] truncated step inputs: LM prompt {tr_rates['prompt_truncated']:.4f}, "
              f"NLI pair {tr_rates['nli_truncated']:.4f}", flush=True)
        if hasattr(env, "close"):
            env.close()
        del lm
    return outputs


def run_prm(cfg: Dict[str, Any], model_tag: Optional[str] = None) -> List[str]:
    """Judge-label the PRM bank, train the logistic head, and score every split under the judge's score key."""
    pcfg = cfg["prm"]
    judge_spec = {"name": pcfg["judge_model"], "tag": pcfg.get("judge_tag", "judge"), "revision": pcfg.get("judge_revision")}
    score_key = pcfg.get("score_key", "prm")
    label_key = pcfg.get("label_key", "judge_label")
    outputs = []
    judge = None
    encoder = None
    for spec in model_specs(cfg, model_tag):
        d = run_dir(cfg, spec["tag"])
        bank_path = os.path.join(d, "prm_bank.jsonl")
        bank = read_jsonl(bank_path)
        if judge is None:
            judge = build_lm(cfg, judge_spec)
        n_labelled = label_bank(judge, bank, label_key=label_key)
        write_jsonl(bank_path, bank)
        steps = [st for tr in bank for st in tr["steps"]]
        labels = np.array([int(st[label_key]) for st in steps])
        if encoder is None:
            encoder = build_encoder(cfg)
        emb = encoder.embed([prm_input_text(st) for st in steps])
        head, history = train_prm_head(emb, labels, lr=pcfg["lr"], weight_decay=pcfg["weight_decay"],
                                       batch_size=pcfg["batch_size"], epochs=pcfg["epochs"],
                                       val_fraction=pcfg["val_fraction"], split_seed=pcfg["split_seed"],
                                       ece_bins=pcfg["ece_bins"], patience=pcfg.get("patience", 1),
                                       seed=cfg["seed"])
        head_path = os.path.join(d, f"prm_{judge_spec['tag']}.npz")
        head.save(head_path, meta={"encoder": pcfg["encoder"], "judge": pcfg["judge_model"], "score_key": score_key,
                                   "label_key": label_key, "n_steps": int(labels.size),
                                   "label_mean": float(labels.mean())})
        with open(os.path.join(d, f"prm_{judge_spec['tag']}_train.json"), "w") as f:
            json.dump({"history": history, "n_steps": int(labels.size), "label_mean": float(labels.mean()),
                       "n_labelled": n_labelled, "score_key": score_key, "label_key": label_key,
                       "n_encoder_inputs_truncated": int(getattr(encoder, "n_truncated", 0))}, f, indent=1)
        files = [f"{split}.jsonl" for split in SPLIT_SIZE_KEY]
        files += [p for p in sorted(os.listdir(d)) if p.endswith(".jsonl")
                  and (p.startswith("horizon_T") or p.startswith("mechanism_"))]
        for name in files:
            p = os.path.join(d, name)
            if os.path.exists(p):
                rows = read_jsonl(p)
                score_trajectories(encoder, head, rows, key=score_key)
                write_jsonl(p, rows)
        print(f"[{cfg['benchmark']}/{spec['tag']}] PRM head trained on {labels.size} judge-labelled steps; "
              f"history={history}", flush=True)
        outputs.append(head_path)
    return outputs


def first_step_scores(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    """Initial-state score of each calibration trajectory; a parse failure scores above the [0,1] range."""
    out = []
    for tr in rows:
        if not tr["steps"]:
            continue
        st = tr["steps"][0]
        if st.get("parse_failed"):
            out.append(float("inf"))
            continue
        v = st.get(key)
        if v is not None:
            out.append(float(v))
    return np.asarray(out, dtype=float)


def truncation_rates(rows: Sequence[Dict[str, Any]], T: int) -> Dict[str, float]:
    """Fraction of steps t <= T whose LM prompt or NLI pair exceeded its token limit and was truncated."""
    steps = [st for tr in rows for st in tr["steps"] if st["t"] <= T]
    return {k: (sum(bool(st.get(k)) for st in steps) / len(steps) if steps else float("nan"))
            for k in ("prompt_truncated", "nli_truncated")}


def available_score_keys(cfg: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Configured score keys plus any PRM score key present in the rows."""
    keys = list(cfg["score_keys"])
    present = {k for tr in rows for st in tr["steps"] for k in st if k.startswith("prm") and st[k] is not None}
    keys += sorted(present - set(keys))
    return [k for k in keys if any(st.get(k) is not None for tr in rows for st in tr["steps"])]


def calibration_draw(cal_pool: np.ndarray, n_cal: int, s: int, seed_base: int = 0, n_seeds: int = 1) -> np.ndarray:
    """Calibration scores for seed s: a disjoint block of the pool when it is large enough, else a resample."""
    cal_pool = np.asarray(cal_pool, dtype=float)
    if cal_pool.size >= n_cal * n_seeds:
        return cal_pool[s * n_cal:(s + 1) * n_cal]
    rng = np.random.default_rng(seed_base + s)
    return cal_pool[rng.integers(0, cal_pool.size, size=n_cal)]


def evaluate_methods(cal_pool: np.ndarray, scores: np.ndarray, failed: np.ndarray, eps: float, T: int,
                     n_cal: int, n_seeds: int, seed_base: int, methods: Sequence[str] = METHODS,
                     s_hat_T: Optional[float] = None,
                     gamma_aci: float = 0.05, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Miscoverage and certified prefix per method over calibration seeds, plus per-trajectory event rates."""
    N = scores.shape[0]
    res: Dict[str, Any] = {m: {"per_seed": [], "prefix_per_seed": [], "events": np.zeros(N),
                                "uncalibratable": False} for m in methods}
    for s in range(n_seeds):
        cal = calibration_draw(cal_pool, n_cal, s, seed_base, n_seeds)
        for m in methods:
            if m == "aci":
                out = C.aci_outcomes(cal, scores, eps, gamma=gamma_aci, failed=failed, labels=labels)
            else:
                try:
                    thr = C.thresholds(cal, eps, T, m, s_hat_T=s_hat_T)
                except C.Uncalibratable:
                    res[m]["per_seed"].append(float("nan"))
                    res[m]["prefix_per_seed"].append(float("nan"))
                    res[m]["uncalibratable"] = True
                    continue
                out = C.trajectory_outcomes(scores, thr, failed=failed)
            res[m]["per_seed"].append(M.miscoverage(out.exceeded))
            res[m]["prefix_per_seed"].append(M.certified_prefix(out.first_exceedance, out.horizon_used))
            res[m]["events"] += out.exceeded.astype(float) / n_seeds
    for m in methods:
        mean, se = M.seed_mean_se([v for v in res[m]["per_seed"] if not np.isnan(v)])
        pm, _ = M.seed_mean_se([v for v in res[m]["prefix_per_seed"] if not np.isnan(v)])
        res[m].update(mean=mean, se=se, prefix=pm)
    return res


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "events"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_evaluate(cfg: Dict[str, Any], model_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calibrate every method in METHODS on the calibration pool and evaluate on the test split."""
    eps, T, n_cal, n_seeds = cfg["eps"], cfg["T"], cfg["n_cal"], cfg["n_seeds"]
    results = []
    for spec in model_specs(cfg, model_tag):
        d = run_dir(cfg, spec["tag"])
        cal_rows = read_jsonl(os.path.join(d, "cal.jsonl"))
        test_rows = read_jsonl(os.path.join(d, "test.jsonl"))
        keys = available_score_keys(cfg, test_rows)
        rng = np.random.default_rng(cfg["seed"] + 777)
        out: Dict[str, Any] = {
            "benchmark": cfg["benchmark"], "model": spec["tag"], "T": T, "eps": eps, "n_cal": n_cal,
            "n_test": len(test_rows), "n_seeds": n_seeds,
            "bounds": {"r1_lb": Th.r1_lower_bound(T, eps), "r3_lb": Th.r3_lower_bound(T, eps),
                       "upper_bound": Th.upper_bound(T, eps, 0.0, n_cal, cfg["delta"]),
                       "wrapper_bound": Th.wrapper_bound(T, eps, 0.0, n_cal, cfg["delta"]),
                       "alpha_budget": C.alpha_budget(eps, T),
                       "realised_alpha_budget": C.realised_alpha_budget(eps, T, n_cal),
                       "dkw_T_slack": T * Th.dkw_slack(n_cal, cfg["delta"]),
                       "k_diamond_exit_rate": Th.k_diamond_exit_rate(eps)},
            "per_score": {}, "pvalues": {}, "truncation_rate": truncation_rates(test_rows, T),
        }
        gold_lab = labels_matrix(test_rows, T)
        for key in keys:
            cal_pool = first_step_scores(cal_rows, key)
            scores, failed = steps_matrix(test_rows, key, T)
            res = evaluate_methods(cal_pool, scores, failed, eps, T, n_cal, n_seeds, cfg["seed"],
                                   gamma_aci=cfg.get("gamma_aci", 0.05),
                                   labels=gold_lab)
            per = {m: {"mean": res[m]["mean"], "se": res[m]["se"], "per_seed": res[m]["per_seed"],
                       "certified_prefix": res[m]["prefix"],
                       "uncalibratable": res[m].get("uncalibratable", False)} for m in METHODS}
            per["implied_iid_rate_perstep"] = (Th.implied_iid_rate(res["perstep"]["mean"], T)
                                               if res["perstep"]["mean"] < 1.0 else None)
            per["wilson_halfwidth"] = {m: M.wilson_halfwidth(res[m]["mean"], len(test_rows)) for m in METHODS}
            per["n_cal_pool"] = int(cal_pool.size)
            per["n_distinct_cal"] = int(np.unique(cal_pool).size)
            q_ps = C.conformal_quantile(cal_pool, eps)
            per["per_step_exceedance_rate"] = C.per_step_exceedance_rate(scores, np.full(T, q_ps), failed=failed)
            per["parse_failure_rate"] = D.parse_failure_rate(failed, scores)
            per["horizon_used"] = {"mean": float(np.nanmean((~np.isnan(scores)).sum(axis=1))),
                                   "min": int((~np.isnan(scores)).sum(axis=1).min()),
                                   "frac_reaching_T": float(((~np.isnan(scores)).sum(axis=1) >= T).mean())}
            out["per_score"][key] = per
            out["pvalues"][key] = {
                "tcw_lt_perstep": M.paired_bootstrap_pvalue(res["tcw"]["events"], res["perstep"]["events"],
                                                            cfg["bootstrap_resamples"], rng, "less"),
                "tcw_vs_bonferroni": M.paired_bootstrap_pvalue(res["tcw"]["events"], res["bonferroni"]["events"],
                                                               cfg["bootstrap_resamples"], rng, "two-sided"),
                "tcw_lt_crc": M.paired_bootstrap_pvalue(res["tcw"]["events"], res["crc"]["events"],
                                                        cfg["bootstrap_resamples"], rng, "less"),
            }
            q_dens = C.conformal_quantile(cal_pool, eps)
            hw = cfg.get("prm", {}).get("density_halfwidth", 0.05)
            bd = cfg.get("prm", {}).get("density_bound", 8.5)
            dens = M.density_near_quantile(cal_pool, q_dens, hw)
            out.setdefault("score_density", {})[key] = {
                "q_hat": q_dens, "density": dens, "bound": bd,
                "within_bound": bool(dens <= bd),
                "n_distinct": int(np.unique(cal_pool).size)}
        with open(results_path(cfg, d, "results"), "w") as f:
            json.dump(_jsonable(out), f, indent=1)
        results.append(out)
        _print_results(out)
    return results


def _print_results(out: Dict[str, Any]) -> None:
    print(f"== {out['benchmark']} / {out['model']} (T={out['T']}, eps={out['eps']}, n_test={out['n_test']}, "
          f"seeds={out['n_seeds']}) ==")
    print(f"  R1 LB {100 * out['bounds']['r1_lb']:.2f}%  R3 LB {100 * out['bounds']['r3_lb']:.2f}%  "
          f"truncated step inputs: LM prompt {out['truncation_rate']['prompt_truncated']:.4f}, "
          f"NLI pair {out['truncation_rate']['nli_truncated']:.4f}")
    show = [m for m in METHODS if m in (out["per_score"].get(next(iter(out["per_score"]), ""), {}))]
    if not show:
        show = list(METHODS)
    for key, per in out["per_score"].items():
        parts = []
        for m in show:
            if m in per and isinstance(per[m], dict) and "mean" in per[m]:
                parts.append(f"{m}={100 * per[m]['mean']:.1f}±{100 * (per[m]['se'] or 0):.1f}")
        pv = out.get("pvalues", {}).get(key, {})
        pv_txt = ""
        if "tcw_lt_perstep" in pv:
            pv_txt = f" | p(tcw<ps)={pv['tcw_lt_perstep']:.3g}"
        print(f"  [{key}] {'  '.join(parts)}{pv_txt}")


def score_correctness(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Pearson correlation between a non-conformity score and step error (1 - gold step label) over all gold-labelled steps."""
    xs, ys = [], []
    for tr in rows:
        for st in tr["steps"]:
            if st.get("gold_label") is not None and st.get(key) is not None:
                xs.append(float(st[key]))
                ys.append(1.0 - float(st["gold_label"]))
    return {"pearson_r": M.pearson(xs, ys), "n_steps": len(xs),
            "n_traj": sum(1 for tr in rows if any(st.get("gold_label") is not None for st in tr["steps"]))}


def run_drift(cfg: Dict[str, Any], model_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Drift diagnostics on the held-out bank: marginal and conditional drift, tau, Delta_0, gate, DC perturbation."""
    eps, T, n_cal, n_seeds = cfg["eps"], cfg["T"], cfg["n_cal"], cfg["n_seeds"]
    results = []
    for spec in model_specs(cfg, model_tag):
        d = run_dir(cfg, spec["tag"])
        cal_rows = read_jsonl(os.path.join(d, "cal.jsonl"))
        bank_rows = read_jsonl(os.path.join(d, "drift_bank.jsonl"))
        test_rows = read_jsonl(os.path.join(d, "test.jsonl")) if os.path.exists(os.path.join(d, "test.jsonl")) else []
        keys = available_score_keys(cfg, bank_rows)
        out: Dict[str, Any] = {"benchmark": cfg["benchmark"], "model": spec["tag"], "T": T, "eps": eps,
                               "m": len(bank_rows), "n_cells": D.n_cells_sqrt(len(bank_rows)),
                               "k_diamond_reference": Th.conditional_drift_two_state(eps, 0.0), "per_score": {}}
        rng = np.random.default_rng(cfg["seed"] + 4242)
        for key in keys:
            cal_pool = first_step_scores(cal_rows, key)
            bank, bank_failed = steps_matrix(bank_rows, key, T)
            rho_t = D.marginal_drift(bank, T)
            rho = D.rho_scalar(rho_t)
            jk = D.jackknife_debias(bank, T)
            tau = D.split_bank_tau(bank, T, cfg["tau_splits"], rng)
            initial = steps_matrix(test_rows, key, T)[0][:, 0] if test_rows else bank[:, 0]
            delta0 = D.initial_state_tv(initial, cal_pool)
            s_T = D.s_hat_T(rho, T)
            etas = []
            cds = []
            for s in range(n_seeds):
                cal = calibration_draw(cal_pool, n_cal, s, cfg["seed"], n_seeds)
                q = C.conformal_quantile(cal, eps)
                cd = D.conditional_drift_cert(bank, q, cal, T)
                etas.append(cd.eta_cert)
                cds.append(cd)
            q_full = C.conformal_quantile(cal_pool, eps)
            n_cells = D.n_cells_sqrt(len(bank_rows))
            rho_se = D.bootstrap_se(lambda b: D.rho_scalar(D.marginal_drift(b, T, n_cells)),
                                    bank, cfg["drift_bootstrap_resamples"], rng)
            eta_mean = float(np.nanmean(etas))
            eta_seed_se = float(np.std(etas, ddof=1) / np.sqrt(len(etas))) if len(etas) > 1 else float("nan")
            eta_bank_se = D.bootstrap_se(lambda b: D.conditional_drift_cert(b, q_full, cal_pool, T).eta_cert,
                                         bank, cfg["drift_bootstrap_resamples"], rng)
            rho_null = D.marginal_drift_null(bank, T, cfg.get("drift_null_perms", 50), rng, n_cells)
            k_rank = C.conformal_rank(n_cal, eps)
            n_from_a_mean = int(np.mean([cd.n_from_a for cd in cds]))
            eta_null = D.eta_cert_iid_null(n_cal, k_rank, n_from_a_mean)
            cd_means = {f: _safe_mean([getattr(cd, f) for cd in cds]) for f in ("k_ba", "k_bb", "mu_b")}
            cd_counts = {"n_from_a": int(np.mean([cd.n_from_a for cd in cds])),
                         "n_from_b": int(np.mean([cd.n_from_b for cd in cds]))}
            per = {"rho_t": rho_t, "rho": rho, "rho_se": rho_se, "rho_null": rho_null,
                   "rho_jackknife_t": jk,
                   "rho_jackknife": D.rho_scalar(jk), "tau": tau, "delta0": delta0, "s_hat_T": s_T,
                   "dc_bound": Th.wrapper_dc_bound(T, eps, tau, n_cal, cfg["delta"]),
                   "eta_cert": eta_mean, "eta_cert_se": eta_bank_se,
                   "eta_cert_seed_se": eta_seed_se,
                   "eta_cert_iid_null": eta_null,
                   "k_ba": cd_means["k_ba"], "k_bb": cd_means["k_bb"], "mu_b": cd_means["mu_b"],
                   "n_from_a": cd_counts["n_from_a"], "n_from_b": cd_counts["n_from_b"],
                   "parse_failure_rate": D.parse_failure_rate(bank_failed, bank),
                   "ratio_eta_over_rho": (eta_mean / rho if rho > 0 else None),
                   "fraction_of_k_diamond": eta_mean / out["k_diamond_reference"],
                   "gate": {str(e): C.gate_passes(s_T, e) for e in cfg["gate_eps_list"]},
                   "score_correctness": score_correctness(bank_rows, key)}
            if test_rows:
                scores, failed = steps_matrix(test_rows, key, T)
                pert = {}
                for f in cfg["perturbation_factors"]:
                    r = evaluate_methods(cal_pool, scores, failed, eps, T, n_cal, n_seeds, cfg["seed"],
                                         methods=("tcw_dc",), s_hat_T=f * s_T)
                    pert[str(f)] = r["tcw_dc"]["mean"]
                base = pert.get("1.0")
                deltas = [abs(100 * (v - base)) for v in pert.values()
                          if base is not None and v is not None and not np.isnan(v) and not np.isnan(base)]
                per["dc_perturbation"] = {"miscoverage": pert,
                                          "max_abs_delta_pp": (max(deltas) if deltas else None)}
            out["per_score"][key] = per
        with open(results_path(cfg, d, "drift"), "w") as f:
            json.dump(_jsonable(out), f, indent=1)
        results.append(out)
        for key, per in out["per_score"].items():
            print(f"[{cfg['benchmark']}/{spec['tag']}/{key}] rho={per['rho']:.4f}±{per['rho_se']:.4f} "
                  f"eta_cert={per['eta_cert']:.4f}±{per['eta_cert_se']:.4f} tau={per['tau']:.4f} "
                  f"D0={per['delta0']:.3f} S_T={per['s_hat_T']:.4f} gate={per['gate']}", flush=True)
    return results


def run_horizon(cfg: Dict[str, Any], model_tag: Optional[str] = None, generate: bool = True,
                resume: bool = True) -> List[Dict[str, Any]]:
    """Horizon-controlled experiment: chained 2-hop questions evaluated at each horizon in cfg['horizons']."""
    if cfg["benchmark"] != "hotpotqa":
        raise ValueError("the horizon experiment is defined for the hotpotqa benchmark")
    horizons = cfg["horizons"]
    block = cfg.get("block_len", 3)
    eps, n_cal, n_seeds = cfg["eps"], cfg["n_cal"], cfg["n_seeds"]
    tasks, splits = load_tasks_and_splits(cfg)
    clusterer = None
    results = []
    for spec in model_specs(cfg, model_tag):
        d = run_dir(cfg, spec["tag"])
        lm = None
        cal_rows = read_jsonl(os.path.join(d, "cal.jsonl"))
        out: Dict[str, Any] = {"benchmark": cfg["benchmark"], "model": spec["tag"], "eps": eps,
                               "n_seeds": n_seeds, "per_T": {}}
        for T in horizons:
            path = os.path.join(d, f"horizon_T{T}.jsonl")
            if generate:
                n_q = T
                idx = sample_with_replacement(splits["eval_pool"], cfg["n_test"] * n_q,
                                              cfg["seed"] * 100 + 50 + T).reshape(cfg["n_test"], n_q)
                inner = build_env(cfg)
                done = _resume_rows(path, f"horizon{T}", resume)
                if len(done) < cfg["n_test"]:
                    if lm is None:
                        lm = build_lm(cfg, spec)
                    if clusterer is None:
                        clusterer = build_clusterer(cfg)
                    if done:
                        print(f"[horizon T={T}] resuming at trajectory {len(done)}", flush=True)
                    with open(path, "a" if done else "w") as f:
                        for j in range(len(done), cfg["n_test"]):
                            chain = ChainedEnv(inner, [tasks[int(i)] for i in idx[j]], block_len=block)
                            agent = build_agent(cfg, lm, chain, clusterer)
                            tr = agent.run(chain.tasks[0], traj_id=f"horizon{T}-{j}",
                                           seed=cfg["seed"] * 10_000_000 + 5_000_000 + T * 10_000 + j, max_steps=T)
                            tr.update(split=f"horizon_T{T}", model=spec["tag"], benchmark=cfg["benchmark"],
                                      chained_task_ids=[tasks[int(i)].task_id for i in idx[j]])
                            f.write(json.dumps(tr) + "\n")
                            f.flush()
                            if (j + 1) % 10 == 0 or j + 1 == cfg["n_test"]:
                                print(f"[horizon T={T}] {j + 1}/{cfg['n_test']}", flush=True)
            rows = read_jsonl(path)
            keys = available_score_keys(cfg, rows)
            per_T: Dict[str, Any] = {"r1_lb": Th.r1_lower_bound(T, eps), "r3_lb": Th.r3_lower_bound(T, eps),
                                     "n_traj": len(rows), "per_score": {}}
            for key in keys:
                cal_pool = first_step_scores(cal_rows, key)
                scores, failed = steps_matrix(rows, key, T)
                res = evaluate_methods(cal_pool, scores, failed, eps, T, n_cal, n_seeds, cfg["seed"])
                per_T["per_score"][key] = {m: {"mean": res[m]["mean"], "se": res[m]["se"]} for m in METHODS}
            out["per_T"][str(T)] = per_T
            print(f"[horizon T={T}] " + "; ".join(
                f"{k}: " + " ".join(f"{m}={100 * v[m]['mean']:.1f}" for m in METHODS)
                for k, v in per_T["per_score"].items()), flush=True)
        with open(results_path(cfg, d, "horizon"), "w") as f:
            json.dump(_jsonable(out), f, indent=1)
        results.append(out)
    return results


def _safe_mean(vals) -> float:
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def _eta_cert_over_seeds(cal_pool: np.ndarray, bank: np.ndarray, eps: float, T: int, n_cal: int, n_seeds: int,
                         seed_base: int) -> Dict[str, Any]:
    """Conditional drift from the certified state, averaged over the calibration seeds' thresholds."""
    cds = [D.conditional_drift_cert(bank, C.conformal_quantile(calibration_draw(cal_pool, n_cal, s, seed_base, n_seeds), eps),
                                    calibration_draw(cal_pool, n_cal, s, seed_base, n_seeds), T) for s in range(n_seeds)]
    etas = [cd.eta_cert for cd in cds]
    return {"eta_cert": _safe_mean(etas),
            "eta_cert_seed_se": float(np.std(etas, ddof=1) / np.sqrt(len(etas))) if len(etas) > 1 else float("nan"),
            "k_ba": _safe_mean([cd.k_ba for cd in cds]), "k_bb": _safe_mean([cd.k_bb for cd in cds]),
            "mu_b": _safe_mean([cd.mu_b for cd in cds]),
            "n_from_a": int(np.mean([cd.n_from_a for cd in cds])), "n_from_b": int(np.mean([cd.n_from_b for cd in cds]))}


def run_mechanism(cfg: Dict[str, Any], model_tag: Optional[str] = None, resume: bool = True) -> List[Dict[str, Any]]:
    """Mechanism ablation: eta_cert on n_test trajectories regenerated under each intervention setting of cfg['mechanism']."""
    mcfg = cfg["mechanism"]
    settings = mcfg["settings"]
    eps, T, n_cal, n_seeds = cfg["eps"], cfg["T"], cfg["n_cal"], cfg["n_seeds"]
    tasks, splits = load_tasks_and_splits(cfg)
    idx = sample_with_replacement(splits[SPLIT_POOL["test"]], cfg["n_test"], cfg["seed"] * 100 + SPLIT_SEED_OFFSET["test"])
    clusterer = None
    results = []
    for spec in model_specs(cfg, model_tag):
        d = run_dir(cfg, spec["tag"])
        cal_rows = read_jsonl(os.path.join(d, "cal.jsonl"))
        lm = None
        out: Dict[str, Any] = {"benchmark": cfg["benchmark"], "model": spec["tag"], "T": T, "eps": eps,
                               "n_traj": len(idx), "n_seeds": n_seeds,
                               "k_diamond_reference": Th.conditional_drift_two_state(eps, 0.0), "settings": {}}
        for si, st in enumerate(settings):
            name, reset_every = st["name"], int(st.get("context_reset_every", 0))
            p_rand = float(st.get("tool_output_randomization", 0.0))
            intervened = bool(reset_every or p_rand > 0)
            path = os.path.join(d, f"mechanism_{name}.jsonl")
            done = _resume_rows(path, f"mechanism_{name}", resume)
            if p_rand > 0 and 0 < len(done) < len(idx):
                print(f"[mechanism_{name}] partial output regenerated from the start (the decoy draws cannot resume)", flush=True)
                done = []
            if len(done) < len(idx):
                if lm is None:
                    lm = build_lm(cfg, spec)
                if clusterer is None:
                    clusterer = build_clusterer(cfg)
                env = build_env(cfg)
                if p_rand > 0:
                    env = RandomizedToolEnv(env, [t.question for t in tasks], p_rand, cfg["seed"] * 100 + 60 + si)
                agent = build_agent(cfg, lm, env, clusterer, context_reset_every=reset_every)
                offset = (7 + si) if intervened else SPLIT_SEED_OFFSET["test"]
                with open(path, "a" if done else "w") as f:
                    for j in range(len(done), len(idx)):
                        tr = agent.run(tasks[int(idx[j])], traj_id=f"mechanism_{name}-{j}",
                                       seed=cfg["seed"] * 10_000_000 + offset * 1_000_000 + j)
                        tr.update(split=f"mechanism_{name}", model=spec["tag"], benchmark=cfg["benchmark"])
                        f.write(json.dumps(tr) + "\n")
                        f.flush()
                        if (j + 1) % 10 == 0 or j + 1 == len(idx):
                            print(f"[{cfg['benchmark']}/{spec['tag']}/mechanism_{name}] {j + 1}/{len(idx)} trajectories",
                                  flush=True)
                if hasattr(env, "close"):
                    env.close()
            rows = read_jsonl(path)
            per: Dict[str, Any] = {"context_reset_every": reset_every, "tool_output_randomization": p_rand,
                                   "n_traj": len(rows), "per_score": {}}
            for key in available_score_keys(cfg, rows):
                cal_pool = first_step_scores(cal_rows, key)
                bank, failed = steps_matrix(rows, key, T)
                r = _eta_cert_over_seeds(cal_pool, bank, eps, T, n_cal, n_seeds, cfg["seed"])
                r["fraction_of_k_diamond"] = r["eta_cert"] / out["k_diamond_reference"]
                r["parse_failure_rate"] = D.parse_failure_rate(failed, bank)
                per["per_score"][key] = r
            out["settings"][name] = per
        floor = mcfg.get("floor_setting")
        if floor in out["settings"]:
            for name, per in out["settings"].items():
                for key, r in per["per_score"].items():
                    f0 = out["settings"][floor]["per_score"].get(key, {}).get("eta_cert")
                    ref = out["k_diamond_reference"]
                    r["floor_adjusted_fraction"] = ((r["eta_cert"] - f0) / (ref - f0)
                                                    if f0 is not None and ref > f0 else None)
        with open(results_path(cfg, d, "mechanism"), "w") as f:
            json.dump(_jsonable(out), f, indent=1)
        results.append(out)
        for name, per in out["settings"].items():
            print(f"[{cfg['benchmark']}/{spec['tag']}/mechanism_{name}] " + "; ".join(
                f"{k}: eta_cert={r['eta_cert']:.4f} frac={r['fraction_of_k_diamond']:.2f}"
                for k, r in per["per_score"].items()), flush=True)
    return results


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


METHOD_LABELS = (("perstep", "Per-step CP"), ("bonferroni", "Bonferroni-T"), ("crc", "CRC"), ("aci", "ACI"),
                 ("tcw", "TC-Wrapper"), ("tcw_lil", "TC-Wrapper (LIL)"))


def model_average(entries: Sequence[Dict[str, Any]], convention: str) -> Dict[str, float]:
    """Mean over models of per-model means; SE = mean of per-model seed SEs (`mean_of_model_ses`) or SE of the pooled per-seed values (`pooled`)."""
    means = [float(e["mean"]) for e in entries]
    if convention == "pooled":
        vals = [v for e in entries for v in e.get("per_seed", []) if v is not None]
        se = M.seed_mean_se(vals)[1] if vals else float("nan")
    elif convention == "mean_of_model_ses":
        ses = [float(e["se"]) for e in entries if e.get("se") is not None]
        se = float(np.mean(ses)) if ses else float("nan")
    else:
        raise ValueError("model_average_se must be 'mean_of_model_ses' or 'pooled'")
    return {"mean": float(np.mean(means)), "se": se}


def _tagged_files(d: str, stem: str) -> List[str]:
    return sorted(p for p in os.listdir(d) if p.startswith(stem) and p.endswith(".json"))


def _fmt(v, spec: str) -> str:
    return "nan" if v is None else format(v, spec)


def run_aggregate(cfg: Dict[str, Any], alpha: float = 0.05) -> str:
    """Summarise every results/drift/mechanism JSON under out_dir; Holm-adjust the p-values jointly."""
    root = _out_root(cfg)
    convention = cfg.get("model_average_se", "mean_of_model_ses")
    entries = []
    for bench in sorted(os.listdir(root)):
        bdir = os.path.join(root, bench)
        if not os.path.isdir(bdir):
            continue
        for model in sorted(os.listdir(bdir)):
            d = os.path.join(bdir, model)
            if not os.path.isdir(d):
                continue
            for name in _tagged_files(d, "results"):
                tag = name[len("results"):-len(".json")]
                drift_p = os.path.join(d, f"drift{tag}.json")
                mech_p = os.path.join(d, f"mechanism{tag}.json")
                entries.append({"bench": bench, "model": model, "tag": tag, "res": _load_json(os.path.join(d, name)),
                                "drift": _load_json(drift_p) if os.path.exists(drift_p) else None,
                                "mech": _load_json(mech_p) if os.path.exists(mech_p) else None})
    if not entries:
        raise FileNotFoundError(f"no results.json under {root}; run `evaluate` first")
    pv_names = ("tcw_lt_perstep", "tcw_vs_bonferroni", "tcw_lt_crc")
    tests = [(e["bench"], e["model"] + e["tag"], k, name, e["res"]["pvalues"][k][name]) for e in entries
             for k in e["res"]["pvalues"] for name in pv_names if name in e["res"]["pvalues"][k]]
    reject, adjusted = M.holm_bonferroni([t[4] for t in tests], alpha)
    lines = []
    groups = sorted({(e["bench"], e["tag"]) for e in entries})
    for bench, tag in groups:
        es = [e for e in entries if e["bench"] == bench and e["tag"] == tag]
        rs = [e["res"] for e in es]
        T, eps = rs[0]["T"], rs[0]["eps"]
        lines += [f"# {bench}{tag} (T={T}, eps={eps}); models {[r['model'] for r in rs]}; "
                  f"model-average SE convention: {convention}", "",
                  "| Method | Score | Miscoverage % (mean over models +- SE) | Per-model miscoverage % | Certified prefix % |",
                  "|---|---|---|---|---|"]
        for key in sorted({k for r in rs for k in r["per_score"]}):
            for m, label in METHOD_LABELS:
                sel = [r["per_score"][key][m] for r in rs if key in r["per_score"] and m in r["per_score"][key]]
                if not sel or any(v["mean"] is None for v in sel):
                    continue
                avg = model_average(sel, convention)
                per_model = "/".join(f"{100 * v['mean']:.1f}" for v in sel)
                lines.append(f"| {label} | {key} | {100 * avg['mean']:.1f} (+-{100 * avg['se']:.1f}) | {per_model} | "
                             f"{100 * np.mean([v['certified_prefix'] for v in sel]):.1f} |")
        b = rs[0]["bounds"]
        lines += [f"| R1 lower bound | - | {100 * b['r1_lb']:.2f} | - | - |",
                  f"| R3 lower bound | - | {100 * b['r3_lb']:.2f} | - | - |", ""]
        for r in rs:
            tr_rate = r.get("truncation_rate") or {}
            lines.append(f"- {r['model']}: n_test {r['n_test']}, n_seeds {r['n_seeds']}, truncated step inputs: "
                         f"LM prompt {tr_rate.get('prompt_truncated')}, NLI pair {tr_rate.get('nli_truncated')}")
            for key, per in r["per_score"].items():
                lines.append(f"- {r['model']} [{key}]: per-step exceedance rate {_fmt(per['per_step_exceedance_rate'], '.4f')}, "
                             f"parse-failure rate {_fmt(per['parse_failure_rate'], '.4f')}, implied i.i.d. rate "
                             f"{per['implied_iid_rate_perstep']}, distinct calibration scores {per['n_distinct_cal']}")
            for key, sd in (r.get("score_density") or {}).items():
                lines.append(f"- {r['model']} [{key}]: score density at q_hat +/- {cfg['prm']['density_halfwidth']} = "
                             f"{_fmt(sd['density'], '.2f')} (bound {sd['bound']})")
        for e in es:
            dr = e["drift"]
            if dr is None:
                continue
            for key, per in dr["per_score"].items():
                sc = per["score_correctness"]
                lines.append(f"- {e['model']} drift[{key}]: rho={_fmt(per['rho'], '.4f')}+-{_fmt(per['rho_se'], '.4f')} "
                             f"(jackknife {_fmt(per['rho_jackknife'], '.4f')}, permutation null {_fmt(per['rho_null'], '.4f')}), "
                             f"eta_cert={_fmt(per['eta_cert'], '.4f')}+-{_fmt(per['eta_cert_se'], '.4f')} "
                             f"(i.i.d. null {_fmt(per['eta_cert_iid_null'], '.4f')}), K(b|b)={per['k_bb']}, "
                             f"tau={_fmt(per['tau'], '.4f')}, Delta0={_fmt(per['delta0'], '.3f')}, S_T={_fmt(per['s_hat_T'], '.4f')}, "
                             f"gate={per['gate']}, DC perturbation max |delta|="
                             f"{per.get('dc_perturbation', {}).get('max_abs_delta_pp')} pp")
                if sc["n_steps"]:
                    lines.append(f"- {e['model']}: Pearson r({key}, step error) = "
                                 f"{'nan' if sc['pearson_r'] is None else round(sc['pearson_r'], 3)} "
                                 f"(n_traj={sc['n_traj']}, n_steps={sc['n_steps']})")
        for e in es:
            mech = e["mech"]
            if mech is None:
                continue
            for name, per in mech["settings"].items():
                for key, r in per["per_score"].items():
                    lines.append(f"- {e['model']} mechanism[{name}][{key}]: eta_cert={_fmt(r['eta_cert'], '.4f')}, "
                                 f"fraction of K_diamond reference={_fmt(r['fraction_of_k_diamond'], '.2f')}, "
                                 f"floor-adjusted={r.get('floor_adjusted_fraction')}")
        lines.append("")
    lines += ["# Paired bootstrap p-values (Holm-adjusted over all tests listed)", "",
              "| Benchmark | Model | Score | Test | p | p_adj | reject |", "|---|---|---|---|---|---|---|"]
    for (b, m, k, name, pv), rj, adj in zip(tests, reject, adjusted):
        lines.append(f"| {b} | {m} | {k} | {name} | {pv:.3g} | {adj:.3g} | {bool(rj)} |")
    text = "\n".join(lines) + "\n"
    out_path = os.path.join(root, "tables.md")
    with open(out_path, "w") as f:
        f.write(text)
    print(text)
    return out_path


def run_all(cfg: Dict[str, Any], model_tag: Optional[str] = None, resume: bool = True) -> None:
    for split in ("cal", "test", "prm_bank", "drift_bank"):
        if split != "prm_bank" or cfg["benchmark"] in STEP_LABEL_BENCHMARKS:
            run_generate(cfg, split, model_tag, resume=resume)
    if cfg["benchmark"] in STEP_LABEL_BENCHMARKS:
        run_prm(cfg, model_tag)
    run_evaluate(cfg, model_tag)
    run_drift(cfg, model_tag)
    if cfg.get("horizons") and cfg["benchmark"] == "hotpotqa":
        run_horizon(cfg, model_tag, resume=resume)
    if cfg.get("mechanism"):
        run_mechanism(cfg, model_tag, resume=resume)
    run_aggregate(cfg)
