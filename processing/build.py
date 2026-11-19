"""Build the clean dataset from the raw scraped files.

Reads   data/raw/wasalt.jsonl, data/raw/darglobal.jsonl, data/raw/darglobal_projects.json
Writes  data/clean/properties.json   (a list of records, see processing/record.py)

Run:
    uv run python -m processing.build
"""

import json
from collections import Counter
from pathlib import Path

from processing.darglobal import clean_project
from processing.wasalt import clean_listing

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT = DATA_DIR / "clean" / "properties.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def remove_duplicates(records: list[dict]) -> list[dict]:
    """Keep the first record for each id (the same page may be scraped twice)."""
    unique = {}
    for record in records:
        unique.setdefault(record["id"], record)
    return list(unique.values())


def build() -> list[dict]:
    cards = {card["url"]: card for card in json.loads((RAW_DIR / "darglobal_projects.json").read_text())}
    projects = [
        clean_project(raw, cards[raw["url"]])
        for raw in read_jsonl(RAW_DIR / "darglobal.jsonl")
        if raw["url"] in cards  # only pages that are listed as projects
    ]
    listings = [clean_listing(raw) for raw in read_jsonl(RAW_DIR / "wasalt.jsonl")]
    return remove_duplicates(projects + listings)


def print_summary(records: list[dict]) -> None:
    print(f"{len(records)} records written to {OUTPUT}")
    print("  by source:", dict(Counter(r["source"] for r in records)))
    print("  Wasalt by purpose:", dict(Counter(r["purpose"] for r in records if r["source"] == "Wasalt")))
    print("  top cities:", Counter(r["city"] for r in records).most_common(8))
    missing_price = sum(1 for r in records if r["source"] == "Wasalt" and r["price"] is None)
    print(f"  Wasalt listings without a price: {missing_price}")


def main() -> None:
    records = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(records)


if __name__ == "__main__":
    main()
