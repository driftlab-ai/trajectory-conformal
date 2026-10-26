"""Process-reward model: frozen-encoder embeddings, logistic head with ECE early stopping, judge labelling, scoring."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.agent import JUDGE_TEMPLATE, judge_step
from evaluation.metrics import ece


def prm_input_text(step: Dict[str, Any], max_obs_chars: int = 300) -> str:
    return (f"Question: {step.get('question', '')}\n{step['step_text']}\n"
            f"Observation: {step.get('observation', '')[:max_obs_chars]}")


class FrozenEncoder:
    """Frozen transformer encoder producing one embedding per text (first token)."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 384, batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.hidden_size = int(self.model.config.hidden_size)
        self.n_truncated = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embeddings; texts longer than max_length are truncated and counted in n_truncated."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            self.n_truncated += sum(len(ids) > self.max_length for ids in self.tokenizer(batch)["input_ids"])
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length).to(self.device)
            with self.torch.no_grad():
                h = self.model(**enc).last_hidden_state[:, 0]
            out.append(h.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.hidden_size))


@dataclass
class PRMHead:
    weight: np.ndarray
    bias: float

    def p_correct(self, emb: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(emb @ self.weight + self.bias)))

    def score(self, emb: np.ndarray) -> np.ndarray:
        """s_PRM = 1 - P(correct), in [0, 1]."""
        return 1.0 - self.p_correct(emb)

    def save(self, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
        np.savez(path, weight=self.weight, bias=np.array(self.bias), meta=json.dumps(meta or {}))


def fixed_split(n: int, val_fraction: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n))) if n > 1 else 0
    return perm[n_val:], perm[:n_val]


def train_prm_head(emb: np.ndarray, labels: np.ndarray, lr: float, weight_decay: float, batch_size: int,
                   epochs: int, val_fraction: float, split_seed: int, ece_bins: int, patience: int,
                   seed: int) -> Tuple[PRMHead, List[Dict[str, float]]]:
    """Logistic head trained with Adam and weight decay, early-stopped on validation ECE."""
    import torch
    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(emb, dtype=np.float32))
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    tr_idx, va_idx = fixed_split(X.shape[0], val_fraction, split_seed)
    tr_idx_t = torch.tensor(tr_idx)
    head = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_state, best_ece, best_epoch, history = None, float("inf"), -1, []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        head.train()
        perm = tr_idx_t[torch.randperm(tr_idx_t.numel(), generator=gen)]
        total = 0.0
        for i in range(0, perm.numel(), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(head(X[b]).squeeze(-1), y[b])
            loss.backward()
            opt.step()
            total += float(loss) * b.numel()
        head.eval()
        with torch.no_grad():
            if va_idx.size:
                p = torch.sigmoid(head(X[va_idx]).squeeze(-1)).numpy()
                val_ece = ece(p, y[va_idx].numpy(), n_bins=ece_bins)
                val_acc = float(((p >= 0.5) == (y[va_idx].numpy() >= 0.5)).mean())
            else:
                val_ece, val_acc = float("nan"), float("nan")
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, tr_idx.size),
                        "val_ece": val_ece, "val_acc": val_acc})
        if not np.isnan(val_ece) and val_ece < best_ece - 1e-12:
            best_ece, best_epoch = val_ece, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        elif best_epoch >= 0 and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    w = head.weight.detach().numpy().reshape(-1).astype(np.float64)
    b = float(head.bias.detach().numpy().reshape(-1)[0])
    return PRMHead(weight=w, bias=b), history


def label_bank(lm, trajectories: Sequence[Dict[str, Any]], log_every: int = 50, label_key: str = "judge_label") -> int:
    """Assign judge labels in place (under `label_key`) to every step of the PRM bank; returns the number labelled."""
    n = 0
    for tr in trajectories:
        for st in tr["steps"]:
            st[label_key] = judge_step(lm, st.get("question", tr["question"]), st["step_text"],
                                       st.get("observation", ""), st.get("reference", ""), JUDGE_TEMPLATE)
            n += 1
            if log_every and n % log_every == 0:
                print(f"  judged {n} steps", flush=True)
    return n


def score_trajectories(encoder: FrozenEncoder, head: PRMHead, trajectories: Sequence[Dict[str, Any]],
                       key: str = "prm") -> None:
    """Add the PRM score under `key` to every step in place."""
    steps = [st for tr in trajectories for st in tr["steps"]]
    if not steps:
        return
    emb = encoder.embed([prm_input_text(st) for st in steps])
    s = head.score(emb)
    for st, v in zip(steps, s):
        st[key] = float(v)
