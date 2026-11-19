"""Answer cache: reuse an answer when the same question finds the same properties.

Why: every AI answer costs a few seconds and one request from the free daily
quota. Many visitors ask the same things ("Which DarGlobal projects are in
Dubai?"), so we keep recent answers and send the saved one instantly.

When is an answer reused? Both must be true:
  1. The search found exactly the same thing: the same filters and the same
     set of properties (the "fingerprint"). So "...under 60k" and "...under 40k"
     never share an answer, because they find different listings.
  2. The question means the same: the same words after tidying (case, spaces,
     punctuation), or a very similar meaning according to the embedding model
     (cosine similarity >= SIMILARITY), e.g. "Which DarGlobal projects are in
     Dubai?" and "DarGlobal projects in Dubai".

Only answers to searches are cached. Small talk and follow-ups answered from the
conversation depend on what was said before, so they are never cached.

The cache lives in the server's memory: it is empty after a restart, entries
expire after TTL, and at most MAX_ENTRIES are kept.

Provider-side "prompt caching" is different: some AI providers reuse the
unchanged beginning of a prompt to answer faster and cheaper. Our prompts put
the fixed instructions first so that works where offered; the free models we
use don't offer it, and the audit trail records cached tokens when they do.
"""

import hashlib
import json
import re
import time
from collections import OrderedDict
from threading import Lock

import numpy as np

from app.database import embedding_function

TTL = 6 * 60 * 60     # seconds an answer stays valid
MAX_ENTRIES = 500
SIMILARITY = 0.90     # how close two questions' meanings must be (1.0 = identical)


def tidy(question: str) -> str:
    """'  Which DarGlobal projects are in Dubai?? ' -> 'which darglobal projects are in dubai'."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", question.lower())).strip()


def fingerprint(found: dict) -> str:
    """A short id for "what the search found": filters + the set of result ids.

    The order is ignored: slightly different wording can rank the same
    properties differently, but the same properties mean the same data.
    """
    data = {"filters": found["filters"], "relaxed": found["relaxed"],
            "ids": sorted(p["id"] for p in found["results"])}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def embed(text: str) -> np.ndarray:
    """Turn text into a meaning vector (length 1) with the same model the search uses."""
    return np.array(embedding_function()([text])[0])


class AnswerCache:
    def __init__(self) -> None:
        # fingerprint -> list of {"question", "vector", "answer", "model", "saved_at"}
        self._entries: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, question: str, found: dict) -> dict | None:
        """Return a saved {"answer", "model", "question", "similarity"} or None."""
        key, words = fingerprint(found), tidy(question)
        with self._lock:
            candidates = [e for e in self._entries.get(key, []) if time.time() - e["saved_at"] < TTL]
        if not candidates:
            self.misses += 1
            return None

        exact = next((e for e in candidates if e["words"] == words), None)
        if exact:
            self.hits += 1
            return {**self._public(exact), "similarity": 1.0}

        vector = embed(question)
        best = max(candidates, key=lambda e: float(vector @ e["vector"]))
        similarity = float(vector @ best["vector"])  # vectors are normalised: dot = cosine
        if similarity >= SIMILARITY:
            self.hits += 1
            return {**self._public(best), "similarity": round(similarity, 3)}
        self.misses += 1
        return None

    def put(self, question: str, found: dict, answer: str, model: str | None) -> None:
        key = fingerprint(found)
        entry = {"question": question, "words": tidy(question), "vector": embed(question),
                 "answer": answer, "model": model, "saved_at": time.time()}
        with self._lock:
            self._entries.setdefault(key, []).append(entry)
            self._entries.move_to_end(key)
            while sum(len(v) for v in self._entries.values()) > MAX_ENTRIES:
                oldest = next(iter(self._entries))
                self._entries[oldest].pop(0)
                if not self._entries[oldest]:
                    del self._entries[oldest]

    def stats(self) -> dict:
        size = sum(len(v) for v in self._entries.values())
        total = self.hits + self.misses
        return {"entries": size, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else None}

    @staticmethod
    def _public(entry: dict) -> dict:
        return {"answer": entry["answer"], "model": entry["model"], "question": entry["question"]}
