"""Semantic-entropy clustering with an NLI cross-encoder and the self-evaluation score."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


class NLIClusterer:
    """Bidirectional-entailment clustering with a sequence-classification NLI model."""

    def __init__(self, model_name: str, device: str = "cpu", entail_threshold: float = 0.5,
                 max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.threshold = entail_threshold
        self.max_length = max_length
        self.n_truncated = 0
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in id2label.items() if l.startswith("entail"))

    def entail_prob(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """Entailment probabilities; pairs longer than max_length are truncated and counted in n_truncated."""
        enc = self.tokenizer(list(premises), list(hypotheses), return_tensors="pt", padding=True)
        over = int((enc["attention_mask"].sum(dim=1) > self.max_length).sum())
        if over:
            self.n_truncated += over
            enc = self.tokenizer(list(premises), list(hypotheses), return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length)
        enc = enc.to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)
        return probs[:, self.entail_idx].cpu().numpy()

    def equivalent(self, a: str, b: str, context: str = "") -> bool:
        """True iff a entails b and b entails a above the threshold."""
        pa, pb = f"{context} {a}".strip(), f"{context} {b}".strip()
        p = self.entail_prob([pa, pb], [pb, pa])
        return bool(p[0] > self.threshold and p[1] > self.threshold)

    def cluster(self, answers: Sequence[Optional[str]], context: str = "") -> List[int]:
        """Greedy clustering: each answer joins the first cluster whose representative it is equivalent to."""
        labels: List[int] = []
        reps: List[str] = []
        for ans in answers:
            if ans is None or not str(ans).strip():
                labels.append(len(reps))
                reps.append("")
                continue
            assigned = None
            for ci, rep in enumerate(reps):
                if rep and self.equivalent(str(ans), rep, context):
                    assigned = ci
                    break
            if assigned is None:
                assigned = len(reps)
                reps.append(str(ans))
            labels.append(assigned)
        return labels


def semantic_entropy(cluster_labels: Sequence[int]) -> float:
    """H = - sum_c p_c log p_c with p_c the empirical cluster frequency over the K samples."""
    labels = np.asarray(cluster_labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return max(0.0, float(-(p * np.log(p)).sum()))


def semant_score(cluster_labels: Sequence[int], n_samples: int) -> float:
    """SemEnt score in [0,1]: H / log K (H is maximised at log K when all K samples differ)."""
    if n_samples <= 1:
        return 0.0
    return min(1.0, semantic_entropy(cluster_labels) / math.log(n_samples))


def selfeval_score(p_yes: float) -> float:
    """SelfEval score = 1 - P(Yes): larger when the agent doubts its own step."""
    return float(min(1.0, max(0.0, 1.0 - p_yes)))
