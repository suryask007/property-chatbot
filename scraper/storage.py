"""Helpers for saving raw scraped records as JSON Lines (one JSON object per line).

JSON Lines lets us append records one at a time, so if a scrape is interrupted
we keep everything collected so far and can resume where we stopped.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def raw_file(source: str) -> Path:
    """Return the output path for a source, e.g. data/raw/wasalt.jsonl."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{source}.jsonl"


def already_scraped(path: Path) -> set[str]:
    """Return the URLs that are already saved in a .jsonl file."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["url"] for line in f if line.strip()}


def append_record(path: Path, record: dict) -> None:
    """Add one record to the end of a .jsonl file, stamped with the scrape time."""
    record = {**record, "scraped_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
