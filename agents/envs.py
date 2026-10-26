"""Task container and environments: BM25 retrieval QA, chained horizon construction, randomized-tool wrapper, GAIA, WorkArena."""
from __future__ import annotations

import collections
import re
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class Task:
    task_id: str
    question: str
    answer: str
    aliases: List[str] = field(default_factory=list)
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    supporting_titles: List[str] = field(default_factory=list)
    decomposition: List[Dict[str, str]] = field(default_factory=list)
    has_step_labels: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA answer normalisation (lowercase, strip punctuation/articles/whitespace)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_correct(pred: str, task: Task, f1_threshold: float = 0.5) -> bool:
    golds = [task.answer] + list(task.aliases)
    return any(normalize_answer(pred) == normalize_answer(g) or f1_score(pred, g) >= f1_threshold
               for g in golds if g)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RetrievalQAEnv:
    """Search[entity] / Lookup[keyword] / Finish[answer] over a task's paragraphs with BM25 retrieval."""

    def __init__(self, max_obs_sentences: int = 5, max_obs_chars: int = 600, f1_threshold: float = 0.5):
        self.max_obs_sentences = max_obs_sentences
        self.max_obs_chars = max_obs_chars
        self.f1_threshold = f1_threshold
        self.task: Optional[Task] = None
        self._bm25: Optional[BM25Okapi] = None
        self._current: Optional[Dict[str, Any]] = None
        self._lookup_cursor = 0
        self._lookup_keyword = ""

    def reset(self, task: Task) -> str:
        self.task = task
        docs = [_tokenize(p["title"] + " " + " ".join(p["sentences"])) for p in task.paragraphs]
        self._bm25 = BM25Okapi(docs) if docs else None
        self._current = None
        self._lookup_cursor = 0
        self._lookup_keyword = ""
        return ""

    def step(self, action_type: Optional[str], argument: str) -> Tuple[str, bool, Dict[str, Any]]:
        if action_type == "Finish":
            return f"Answer recorded: {argument}", True, {}
        if action_type == "Search":
            return self._search(argument), False, {}
        if action_type == "Lookup":
            return self._lookup(argument), False, {}
        return ("Invalid action. Valid actions are Search[entity], Lookup[keyword] and "
                "Finish[answer]."), False, {}

    def _search(self, query: str) -> str:
        if self._bm25 is None or not self.task.paragraphs:
            return "No documents available."
        scores = self._bm25.get_scores(_tokenize(query))
        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        self._current = self.task.paragraphs[best]
        self._lookup_cursor = 0
        text = " ".join(self._current["sentences"][: self.max_obs_sentences])
        return f"{self._current['title']}: {text}"[: self.max_obs_chars]

    def _lookup(self, keyword: str) -> str:
        if self._current is None:
            return "No passage is open. Use Search first."
        if keyword != self._lookup_keyword:
            self._lookup_keyword = keyword
            self._lookup_cursor = 0
        sents = self._current["sentences"]
        kw = keyword.lower()
        for i in range(self._lookup_cursor, len(sents)):
            if kw in sents[i].lower():
                self._lookup_cursor = i + 1
                return f"(Result {i + 1}) {sents[i]}"[: self.max_obs_chars]
        self._lookup_cursor = len(sents)
        return "No more results."

    def gold_label(self, action_type: Optional[str], argument: str, observation: str) -> Optional[int]:
        """1 if a retrieval step opened a gold supporting paragraph or a Finish step matched the gold answer."""
        if not self.task.has_step_labels:
            return None
        if action_type in ("Search", "Lookup"):
            if self._current is None:
                return 0
            return int(bool(self._current.get("supporting")))
        if action_type == "Finish":
            return int(answer_correct(argument, self.task, self.f1_threshold))
        return 0

    def reference_text(self, action_type: Optional[str]) -> str:
        t = self.task
        parts = []
        if t.supporting_titles:
            parts.append("Gold supporting paragraphs: " + "; ".join(t.supporting_titles))
        if t.decomposition:
            hops = " | ".join(f"{h.get('question', '')} -> {h.get('answer', '')}" for h in t.decomposition)
            parts.append("Gold hop decomposition: " + hops)
        parts.append(f"Gold final answer: {t.answer}")
        return "\n".join(parts)


class ChainedEnv:
    """Runs several tasks back to back on one inner environment; a new task starts when the current one is answered or after block_len steps."""

    def __init__(self, inner: RetrievalQAEnv, tasks: Sequence[Task], block_len: int = 3):
        self.inner = inner
        self.tasks = list(tasks)
        self.block_len = block_len
        self.k = 0
        self.steps_in_task = 0
        self._pending: Optional[Task] = None

    def reset(self, task: Optional[Task] = None) -> str:
        self.k = 0
        self.steps_in_task = 0
        self._pending = None
        return self.inner.reset(self.tasks[0])

    def step(self, action_type, argument):
        """One step; the next task is loaded at the start of the following step, so a block's last step is labelled against its own task."""
        if self._pending is not None:
            self.inner.reset(self._pending)
            self._pending = None
        obs, done, info = self.inner.step(action_type, argument)
        self.steps_in_task += 1
        if done or self.steps_in_task >= self.block_len:
            self.k += 1
            self.steps_in_task = 0
            if self.k < len(self.tasks):
                nxt = self.tasks[self.k]
                self._pending = nxt
                info = dict(info, new_question=nxt.question)
                return obs, False, info
            return obs, True, info
        return obs, done, info

    def gold_label(self, *a, **k):
        return self.inner.gold_label(*a, **k)

    def reference_text(self, *a, **k):
        return self.inner.reference_text(*a, **k)


class RandomizedToolEnv:
    """With probability p at each non-Finish step, returns the tool's response to a uniformly random decoy query instead of to the agent's argument."""

    def __init__(self, inner, decoy_queries: Sequence[str], p: float, seed: int):
        self.inner = inner
        self.decoys = list(decoy_queries)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def reset(self, task: Task) -> str:
        return self.inner.reset(task)

    def step(self, action_type, argument):
        if action_type not in (None, "Finish") and self.decoys and self.rng.random() < self.p:
            obs, _, _ = self.inner.step(action_type, self.decoys[int(self.rng.integers(len(self.decoys)))])
            return obs, False, {}
        return self.inner.step(action_type, argument)

    def gold_label(self, *a, **k):
        return self.inner.gold_label(*a, **k)

    def reference_text(self, *a, **k):
        return self.inner.reference_text(*a, **k)

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()


class GAIAEnv:
    """GAIA level-1 tools: Search[query] (web search), ReadFile[] (attached file), Finish[answer]."""

    def __init__(self, search_fn: Optional[Callable[[str], str]] = None, max_obs_chars: int = 1200):
        self.search_fn = search_fn or _default_web_search
        self.max_obs_chars = max_obs_chars
        self.task: Optional[Task] = None

    def reset(self, task: Task) -> str:
        self.task = task
        return ""

    def step(self, action_type, argument):
        if action_type == "Finish":
            return f"Answer recorded: {argument}", True, {}
        if action_type == "Search":
            try:
                return self.search_fn(argument)[: self.max_obs_chars], False, {}
            except Exception as e:
                return f"Search failed: {e}", False, {}
        if action_type == "ReadFile":
            return _read_attachment(self.task.meta.get("file_path", ""))[: self.max_obs_chars], False, {}
        return "Invalid action. Valid actions are Search[query], ReadFile[] and Finish[answer].", False, {}

    def gold_label(self, action_type, argument, observation):
        if action_type == "Finish":
            return int(gaia_match(argument, self.task.answer))
        return None

    def reference_text(self, action_type):
        return f"Gold final answer: {self.task.answer}"


def gaia_match(pred: str, gold: str) -> bool:
    """GAIA answer match: numeric, comma-list, or normalised string comparison."""
    def _num(x):
        try:
            return float(str(x).replace(",", "").replace("$", "").replace("%", "").strip())
        except ValueError:
            return None
    g = str(gold).strip()
    p = str(pred).strip()
    gn = _num(g)
    if gn is not None:
        pn = _num(p)
        return pn is not None and abs(pn - gn) < 1e-9
    if "," in g:
        gl = [normalize_answer(x) for x in g.split(",")]
        pl = [normalize_answer(x) for x in p.split(",")]
        return gl == pl
    return normalize_answer(p) == normalize_answer(g)


def _default_web_search(query: str) -> str:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=3))
    return "\n".join(f"{h.get('title', '')}: {h.get('body', '')}" for h in hits) or "No results."


def _read_attachment(path: str) -> str:
    if not path:
        return "No file is attached to this task."
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages[:5])
    with open(path, "r", errors="replace") as f:
        return f.read()


class WorkArenaEnv:
    """Wraps a BrowserGym WorkArena-L2 task: Browser[<browsergym action>] / Finish[answer]."""

    def __init__(self, headless: bool = True, max_obs_chars: int = 4000):
        self.headless = headless
        self.max_obs_chars = max_obs_chars
        self.env = None
        self.task: Optional[Task] = None

    def reset(self, task: Task) -> str:
        import gymnasium as gym
        import browsergym.workarena
        from browsergym.core.action.highlevel import HighLevelActionSet
        self.task = task
        if self.env is not None:
            self.env.close()
        self.env = gym.make(
            f"browsergym/{task.task_id}",
            headless=self.headless,
            action_mapping=HighLevelActionSet(subsets=["chat", "bid", "nav"], strict=False, multiaction=False).to_python_code,
        )
        obs, _ = self.env.reset()
        return self._render(obs)

    def step(self, action_type, argument):
        if action_type == "Finish":
            code = f"send_msg_to_user({argument!r})"
        elif action_type == "Browser":
            code = argument
        else:
            return "Invalid action. Use Browser[<browsergym action>] or Finish[answer].", False, {}
        obs, reward, terminated, truncated, info = self.env.step(code)
        done = bool(terminated or truncated or action_type == "Finish")
        return self._render(obs), done, {"reward": float(reward)}

    def _render(self, obs) -> str:
        from browsergym.utils.obs import flatten_axtree_to_str
        goal = obs.get("goal", "") if isinstance(obs, dict) else ""
        tree = flatten_axtree_to_str(obs["axtree_object"]) if isinstance(obs, dict) else str(obs)
        return f"Goal: {goal}\n{tree}"[: self.max_obs_chars]

    def gold_label(self, action_type, argument, observation):
        return None

    def reference_text(self, action_type):
        return ""

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
