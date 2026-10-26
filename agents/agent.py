"""Hugging Face causal-LM wrapper, ReAct agent loop with per-step scores, LLM judge, score-matrix construction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.scores import selfeval_score, semant_score

ACTION_RE = re.compile(r"Action\s*\d*\s*:\s*([A-Za-z]+)\s*\[(.*?)\]", re.DOTALL)
STOP_STRINGS = ("\nObservation",)

SELFEVAL_SUFFIX = ("\nIs the step above (its thought and action) correct and useful for answering "
                   "the question? Answer Yes or No.\nAnswer:")

JUDGE_TEMPLATE = (
    "You are grading one step taken by an agent solving a task.\n"
    "Task: {question}\n"
    "Agent step:\n{step_text}\n"
    "Observation returned by the environment: {observation}\n"
    "Reference information:\n{reference}\n"
    "Is the agent step correct, i.e. consistent with the reference and moving toward the gold "
    "answer? Answer Yes or No.\nAnswer:"
)


class HFLM:
    def __init__(self, model_name: str, revision: Optional[str] = None, device: str = "cpu",
                 dtype: str = "float32", max_new_tokens: int = 128, max_prompt_tokens: int = 4096,
                 device_map: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        torch_dtype = getattr(torch, dtype)
        if device_map:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch_dtype,
                                                              device_map=device_map).eval()
            self.device = self.model.device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                              torch_dtype=torch_dtype).to(device).eval()
            self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.n_truncated = 0
        self._yes_ids = self._token_ids(["Yes", " Yes", "yes", " yes", "YES"])
        self._no_ids = self._token_ids(["No", " No", "no", " no", "NO"])

    def _token_ids(self, variants: Sequence[str]) -> List[int]:
        ids = []
        for v in variants:
            enc = self.tokenizer.encode(v, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return ids

    def _encode(self, prompt: str):
        """Tokenize; a prompt longer than max_prompt_tokens is left-truncated and counted in n_truncated."""
        enc = self.tokenizer(prompt, return_tensors="pt")
        if enc["input_ids"].shape[1] > self.max_prompt_tokens:
            self.n_truncated += 1
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_prompt_tokens)
        return {k: v.to(self.device) for k, v in enc.items()}

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0, seed: Optional[int] = None,
                 max_new_tokens: Optional[int] = None, stop: Sequence[str] = STOP_STRINGS) -> List[str]:
        """n completions; temperature 0 -> greedy (n must be 1), else sampling."""
        torch = self.torch
        enc = self._encode(prompt)
        kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id, repetition_penalty=1.0)
        if temperature <= 0.0:
            kwargs.update(do_sample=False, num_return_sequences=1)
        else:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0,
                          num_return_sequences=n)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        plen = enc["input_ids"].shape[1]
        texts = [self.tokenizer.decode(o[plen:], skip_special_tokens=True) for o in out]
        return [_truncate_at(t, stop) for t in texts]

    def yes_no_prob(self, prompt: str) -> float:
        """P(Yes) among {Yes, No} next tokens (first-token variants summed)."""
        torch = self.torch
        enc = self._encode(prompt)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = float(probs[self._yes_ids].sum())
        p_no = float(probs[self._no_ids].sum())
        if p_yes + p_no <= 0:
            return 0.5
        return p_yes / (p_yes + p_no)


def _truncate_at(text: str, stop: Sequence[str]) -> str:
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


def parse_step(text: str) -> Tuple[str, Optional[str], str, str, bool]:
    """Split a generated step into (thought, action_type, action_arg, action_raw, parse_failed)."""
    m = ACTION_RE.search(text)
    if m is None:
        thought = text.strip().split("\n")[0] if text.strip() else ""
        return thought, None, "", text.strip(), True
    thought = text[: m.start()].strip()
    thought = re.sub(r"^Thought\s*\d*\s*:\s*", "", thought).strip()
    return thought, m.group(1), m.group(2).strip(), f"{m.group(1)}[{m.group(2).strip()}]", False


@dataclass
class StepRecord:
    t: int
    thought: str
    action_type: Optional[str]
    action_arg: str
    action_raw: str
    parse_failed: bool
    observation: str
    step_text: str
    reference: str
    gold_label: Optional[int]
    semant: Optional[float] = None
    selfeval: Optional[float] = None
    judge_label: Optional[int] = None
    samples: List[Optional[str]] = field(default_factory=list)
    question: str = ""
    prompt_truncated: bool = False
    nli_truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class ReActAgent:
    """ReAct loop; `context_reset_every` = k clears the scratchpad after every k steps (0 = never)."""

    def __init__(self, lm, env, prompt_template: str, max_steps: int,
                 n_samples: int = 5, sample_temperature: float = 0.7,
                 clusterer=None, selfeval: bool = True, context_reset_every: int = 0):
        self.lm = lm
        self.env = env
        self.template = prompt_template
        self.max_steps = max_steps
        self.n_samples = n_samples
        self.sample_temperature = sample_temperature
        self.clusterer = clusterer
        self.selfeval = selfeval
        self.context_reset_every = context_reset_every

    def _prompt(self, question: str, scratchpad: str, t: int) -> str:
        return self.template.format(question=question, scratchpad=scratchpad) + f"Thought {t}:"

    def run(self, task, traj_id: str, seed: int, max_steps: Optional[int] = None) -> Dict[str, Any]:
        env = self.env
        initial = env.reset(task)
        question = task.question
        scratchpad = f"Observation 0: {initial}\n" if initial else ""
        steps: List[StepRecord] = []
        final_answer = None
        limit = max_steps or self.max_steps
        t_local = 0
        for t in range(1, limit + 1):
            t_local += 1
            n_lm0 = getattr(self.lm, "n_truncated", 0)
            n_nli0 = getattr(self.clusterer, "n_truncated", 0)
            prompt = self._prompt(question, scratchpad, t_local)
            greedy = self.lm.generate(prompt, n=1, temperature=0.0, seed=seed * 1000 + 500 + t)[0]
            thought, a_type, a_arg, a_raw, failed = parse_step(greedy)
            step_text = f"Thought {t_local}: {thought}\nAction {t_local}: {a_raw if not failed else greedy.strip()}"
            semant = None
            samples: List[Optional[str]] = []
            if self.n_samples > 0 and self.clusterer is not None:
                outs = self.lm.generate(prompt, n=self.n_samples, temperature=self.sample_temperature,
                                        seed=seed * 1000 + t)
                for o in outs:
                    _, st, sa, _, sf = parse_step(o)
                    samples.append(None if sf else f"{st}[{sa}]")
                labels = self.clusterer.cluster(samples, context=question)
                semant = semant_score(labels, self.n_samples)
            selfeval = None
            if self.selfeval:
                p_yes = self.lm.yes_no_prob(prompt + " " + greedy.strip() + SELFEVAL_SUFFIX)
                selfeval = selfeval_score(p_yes)
            observation, done, info = env.step(a_type, a_arg)
            gold = env.gold_label(a_type, a_arg, observation)
            reference = env.reference_text(a_type)
            steps.append(StepRecord(t=t, thought=thought, action_type=a_type, action_arg=a_arg,
                                    action_raw=a_raw, parse_failed=failed, observation=observation,
                                    step_text=step_text, reference=reference, gold_label=gold,
                                    semant=semant, selfeval=selfeval, samples=samples, question=question,
                                    prompt_truncated=getattr(self.lm, "n_truncated", 0) > n_lm0,
                                    nli_truncated=getattr(self.clusterer, "n_truncated", 0) > n_nli0))
            scratchpad += f"Thought {t_local}: {thought}\nAction {t_local}: {a_raw if not failed else 'None'}\nObservation {t_local}: {observation}\n"
            if a_type == "Finish":
                final_answer = a_arg
            if info.get("new_question"):
                question = info["new_question"]
                scratchpad = ""
                t_local = 0
            elif self.context_reset_every and t_local >= self.context_reset_every:
                scratchpad = ""
                t_local = 0
            if done:
                break
        return {
            "traj_id": traj_id, "task_id": task.task_id, "question": task.question,
            "gold_answer": task.answer, "seed": seed, "n_steps": len(steps),
            "final_answer": final_answer,
            "steps": [s.to_dict() for s in steps],
        }


def judge_step(lm: HFLM, question: str, step_text: str, observation: str, reference: str,
               template: str = JUDGE_TEMPLATE) -> int:
    """Binary judge label for one step (1 = correct)."""
    prompt = template.format(question=question, step_text=step_text,
                             observation=observation[:600], reference=reference)
    return int(lm.yes_no_prob(prompt) >= 0.5)


def steps_matrix(trajectories: Sequence[Dict[str, Any]], key: str, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build (N, T) score and parse-failure matrices from trajectory dicts (NaN where absent)."""
    N = len(trajectories)
    scores = np.full((N, T), np.nan)
    failed = np.zeros((N, T), dtype=bool)
    for i, tr in enumerate(trajectories):
        for st in tr["steps"][:T]:
            j = st["t"] - 1
            if j >= T:
                continue
            v = st.get(key)
            failed[i, j] = bool(st.get("parse_failed", False))
            scores[i, j] = np.nan if v is None else float(v)
            if failed[i, j]:
                scores[i, j] = float("inf")
    return scores, failed


def labels_matrix(trajectories: Sequence[Dict[str, Any]], T: int) -> np.ndarray:
    """Build (N, T) gold-label matrix from trajectory dicts (NaN where absent)."""
    N = len(trajectories)
    lab = np.full((N, T), np.nan)
    for i, tr in enumerate(trajectories):
        for st in tr["steps"][:T]:
            j = st["t"] - 1
            if j >= T:
                continue
            g = st.get("gold_label")
            if g is not None:
                lab[i, j] = float(g)
    return lab
