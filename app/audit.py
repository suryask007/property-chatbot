"""Traceability: one audit record per chat message.

Every message gets a request id. While it is handled we record what happened
at each step, and when the answer is finished we write one JSON record:

  {
    "request_id": "c9d7e12bc9fe4c1e9f0a7b3d2e6f1a08",   shown under the answer; same id in Langfuse
    "session_id": "...",            which conversation it belongs to
    "timestamp":  "2026-11-19T05:01:02Z",
    "message":    "what about Jeddah?",
    "plan":   {"by": "ai" | "rules", "search": true, "query": "3 bedroom apartments for rent in Jeddah..."},
    "search": {"filters": {...}, "keywords": [...], "total_matches": 10, "relaxed": false,
               "result_ids": ["wasalt-5938381", ...]},
    "answer": {"model": "nvidia/...:free", "fallback": false, "text": "..."},
    "timings_ms": {"plan": 812, "search": 64, "first_word": 1430, "total": 4210},
    "errors": []
  }

Records go to three places:
  * the server log (one line, prefixed "AUDIT") - visible in Hugging Face's Logs tab
  * a JSON Lines file, data/audit/audit.jsonl (AUDIT_FILE to change it)
  * the last MAX_RECENT records in memory, served by GET /api/traces (needs ADMIN_TOKEN)

The AI calls themselves (prompts, tokens, latency) can also be sent to
Langfuse; see app/llm.py. They share the same request id, so a record here and
a trace in Langfuse can be matched.
"""

import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from pathlib import Path
from threading import Lock

AUDIT_FILE = Path(os.getenv("AUDIT_FILE", Path(__file__).resolve().parent.parent / "data" / "audit" / "audit.jsonl"))
MAX_RECENT = 500

log = logging.getLogger("uvicorn.error")
recent: deque = deque(maxlen=MAX_RECENT)
_file_lock = Lock()


class Trace:
    """Collects what happens while one chat message is handled."""

    def __init__(self, session_id: str, message: str) -> None:
        # 32 hex characters: the trace id format Langfuse uses, so the ids are identical there.
        self.request_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.record = {
            "request_id": self.request_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "message": message,
            "plan": None,
            "search": None,
            "answer": None,
            "timings_ms": {},
            "errors": [],
        }

    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self.started) * 1000)

    def timing(self, step: str, since_ms: int) -> None:
        """Record how long a step took, given the elapsed time when it started."""
        self.record["timings_ms"][step] = self.elapsed_ms() - since_ms

    def mark(self, moment: str) -> None:
        """Record when something happened, counted from the start of the request."""
        self.record["timings_ms"][moment] = self.elapsed_ms()

    def error(self, where: str, message: str) -> None:
        self.record["errors"].append({"step": where, "error": message[:500]})

    def finish(self) -> None:
        """Write the finished record to the log, the audit file and the recent list."""
        self.mark("total")
        line = json.dumps(self.record, ensure_ascii=False)
        log.info(f"AUDIT {line}")
        recent.append(self.record)
        try:
            with _file_lock:
                AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
                with AUDIT_FILE.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as error:  # a read-only disk must not break the chat
            log.warning(f"could not write audit file: {error}")


def find(request_id: str) -> dict | None:
    return next((r for r in reversed(recent) if r["request_id"] == request_id), None)
