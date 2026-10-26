"""Benchmark loaders and deterministic pool splits / draws."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Sequence

import numpy as np

from agents.envs import Task


def _split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


def load_hotpotqa(limit: int = 0, cache_dir: str = "") -> List[Task]:
    """HotpotQA distractor validation split."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                      cache_dir=cache_dir or None)
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        sup = set(row["supporting_facts"]["title"])
        paras = [{"title": t, "sentences": list(s), "supporting": t in sup}
                 for t, s in zip(row["context"]["title"], row["context"]["sentences"])]
        tasks.append(Task(task_id=row["id"], question=row["question"], answer=row["answer"],
                          paragraphs=paras, supporting_titles=sorted(sup), has_step_labels=True))
    return tasks


def load_musique(limit: int = 0, cache_dir: str = "") -> List[Task]:
    """MuSiQue-Ans dev split with paragraphs and gold hop decompositions."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl", repo_type="dataset",
                           cache_dir=cache_dir or None)
    tasks = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            row = json.loads(line)
            paras = [{"title": p["title"], "sentences": _split_sentences(p["paragraph_text"]),
                      "supporting": bool(p["is_supporting"])} for p in row["paragraphs"]]
            sup = [p["title"] for p in paras if p["supporting"]]
            dec = [{"question": h["question"], "answer": h["answer"]} for h in row["question_decomposition"]]
            tasks.append(Task(task_id=row["id"], question=row["question"], answer=row["answer"],
                              aliases=list(row.get("answer_aliases", [])), paragraphs=paras,
                              supporting_titles=sup, decomposition=dec, has_step_labels=True))
    return tasks


def load_financebench(limit: int = 0, cache_dir: str = "") -> List[Task]:
    """FinanceBench released questions; evidence passages form the retrieval corpus."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("PatronusAI/financebench", "financebench_merged.jsonl", repo_type="dataset",
                           cache_dir=cache_dir or None)
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    corpus = []
    for r in rows:
        for j, ev in enumerate(r["evidence"]):
            corpus.append({"title": f"{ev['doc_name']} p.{ev.get('evidence_page_num', '?')}",
                           "sentences": _split_sentences(ev.get("evidence_text_full_page") or ev["evidence_text"]),
                           "owner": r["financebench_id"]})
    tasks = []
    for i, r in enumerate(rows):
        if limit and i >= limit:
            break
        paras = [{"title": c["title"], "sentences": c["sentences"], "supporting": c["owner"] == r["financebench_id"]}
                 for c in corpus]
        tasks.append(Task(task_id=r["financebench_id"], question=r["question"], answer=str(r["answer"]),
                          paragraphs=paras, supporting_titles=[p["title"] for p in paras if p["supporting"]],
                          has_step_labels=False))
    return tasks


def load_gaia(limit: int = 0, cache_dir: str = "") -> List[Task]:
    """GAIA level-1 validation split (gated dataset; set HF_TOKEN)."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    ds = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation",
                      cache_dir=cache_dir or None, token=os.environ.get("HF_TOKEN"))
    root = snapshot_download("gaia-benchmark/GAIA", repo_type="dataset", allow_patterns=["2023/validation/*"],
                             cache_dir=cache_dir or None, token=os.environ.get("HF_TOKEN"))
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        rel = row.get("file_path") or ""
        tasks.append(Task(task_id=row["task_id"], question=row["Question"], answer=str(row["Final answer"]),
                          has_step_labels=True,
                          meta={"file_path": os.path.join(root, rel) if rel else ""}))
    return tasks


def load_workarena(limit: int = 0, cache_dir: str = "") -> List[Task]:
    """WorkArena-L2 tasks enumerated from the browsergym-workarena package."""
    import browsergym.workarena as wa
    task_classes = None
    for attr in ("ALL_COMPOSITIONAL_TASKS_L2", "ALL_COMPOSITIONAL_TASKS"):
        if hasattr(wa, attr):
            val = getattr(wa, attr)
            if isinstance(val, (list, tuple)) and len(val) > 0 and hasattr(val[0], "get_task_id"):
                task_classes = val
                break
    if task_classes is None:
        raise RuntimeError("could not find the WorkArena-L2 task list in browsergym.workarena")
    tasks = []
    for i, cls in enumerate(task_classes):
        if limit and i >= limit:
            break
        task_id = cls.get_task_id() if hasattr(cls, "get_task_id") else str(cls)
        tasks.append(Task(task_id=task_id, question=f"Complete the WorkArena task {task_id}.", answer="",
                          has_step_labels=False))
    return tasks


LOADERS = {"hotpotqa": load_hotpotqa, "musique": load_musique, "financebench": load_financebench,
           "gaia": load_gaia, "workarena": load_workarena}
RETRIEVAL_BENCHMARKS = ("hotpotqa", "musique", "financebench")
STEP_LABEL_BENCHMARKS = ("hotpotqa", "musique", "gaia")


def load_pool(benchmark: str, limit: int = 0, cache_dir: str = "") -> List[Task]:
    if benchmark not in LOADERS:
        raise ValueError(f"unknown benchmark {benchmark!r}; known: {sorted(LOADERS)}")
    return LOADERS[benchmark](limit=limit, cache_dir=cache_dir)


def make_splits(n_pool: int, seed: int, cal_pool_fraction: float = 0.5) -> Dict[str, np.ndarray]:
    """Disjoint random split of the pool into a calibration half and an evaluation half."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)
    n_cal = max(1, int(round(cal_pool_fraction * n_pool)))
    if n_pool >= 2:
        n_cal = min(n_cal, n_pool - 1)
    return {"cal_pool": np.sort(perm[:n_cal]), "eval_pool": np.sort(perm[n_cal:])}


def sample_with_replacement(pool_idx: Sequence[int], size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool_idx)
    return pool[rng.integers(0, pool.size, size=size)]


SPLIT_POOL = {"cal": "cal_pool", "test": "eval_pool", "prm_bank": "eval_pool", "drift_bank": "eval_pool"}
SPLIT_SEED_OFFSET = {"cal": 1, "test": 2, "prm_bank": 3, "drift_bank": 4}
