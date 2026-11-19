"""Module 3 - Storage.

Loads data/clean/properties.json into two stores that work together:

1. SQLite (data/properties.db) - exact data and keyword search
     properties      one row per property, one column per field (see processing/record.py).
                     List/dict fields (amenities, facts, faqs) are stored as JSON text.
                     Used for exact filters: city, price, bedrooms, sale/rent...
     properties_fts  a full-text keyword index (SQLite FTS5). Matches words like
                     "beach", "pool", "Trump"; understands plurals and accents.

2. Chroma vector database (data/chroma/) - search by meaning
     Each property's text is turned into an "embedding" (a list of 384 numbers
     that captures its meaning) by a small local model, all-MiniLM-L6-v2.
     Questions are embedded the same way, and Chroma returns the properties
     whose meaning is closest. So "seaside homes" finds "beachfront villas",
     and small typos ("ocenfront") still work. The model runs on the CPU, for
     free, with no API key.

Both are built once, before deployment, and shipped inside the Docker image.

Run:
    uv run python -m app.database
"""

import json
import shutil
import sqlite3
from pathlib import Path

import chromadb
from chromadb.config import Settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLEAN_FILE = DATA_DIR / "clean" / "properties.json"
DB_FILE = DATA_DIR / "properties.db"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION = "properties"

JSON_FIELDS = ("amenities", "facts", "faqs")

SCHEMA = """
CREATE TABLE properties (
    id TEXT PRIMARY KEY,
    source TEXT, kind TEXT, title TEXT,
    purpose TEXT, property_type TEXT, usage TEXT,
    country TEXT, city TEXT, district TEXT, location_text TEXT,
    price REAL, currency TEXT, price_period TEXT,
    bedrooms INTEGER, bathrooms INTEGER, area_sqm REAL,
    unit_types TEXT, area_text TEXT, status TEXT, furnishing TEXT,
    amenities TEXT, facts TEXT, description TEXT, faqs TEXT,
    url TEXT, image_url TEXT, latitude REAL, longitude REAL,
    scraped_at TEXT
);

CREATE INDEX idx_filters ON properties (source, purpose, city, bedrooms, price);

CREATE VIRTUAL TABLE properties_fts USING fts5(
    id UNINDEXED,
    title,
    location,
    details,
    description,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
"""


def search_text(record: dict) -> dict:
    """Build the text that the full-text index searches for one property."""
    details = [
        record["property_type"], record["usage"], record["unit_types"], record["status"],
        record["furnishing"], f"for {record['purpose']}" if record["purpose"] else None,
        f"{record['bedrooms']} bedrooms" if record["bedrooms"] else None,
        record["kind"], record["source"],
        " ".join(record["amenities"]),
        " ".join(f"{k} {v}" for k, v in record["facts"].items()),
    ]
    faqs = " ".join(f"{f['question']} {f['answer']}" for f in record["faqs"])
    return {
        "id": record["id"],
        "title": record["title"] or "",
        "location": " ".join(filter(None, [
            record["district"], record["city"], record["country"], record["location_text"],
        ])),
        "details": " ".join(filter(None, details)),
        "description": f"{record['description']} {faqs}".strip(),
    }


def embedding_text(record: dict) -> str:
    """The text whose meaning is stored in Chroma for one property.

    The embedding model reads only about the first 250 words, so the most
    descriptive parts come first.
    """
    text = search_text(record)
    return f"{text['title']}. {text['details']}. {text['location']}. {text['description']}"[:2000]


def chroma_collection():
    """Open the Chroma collection (cosine distance: 0 = same meaning, 2 = opposite)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def build_vector_index(records: list[dict]) -> None:
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)  # always rebuild from scratch
    collection = chroma_collection()
    batch = 100
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        collection.add(
            ids=[r["id"] for r in chunk],
            documents=[embedding_text(r) for r in chunk],
            metadatas=[{"source": r["source"], "title": r["title"] or ""} for r in chunk],
        )
        print(f"  embedded {min(start + batch, len(records))}/{len(records)}")
    print(f"Saved {collection.count()} embeddings to {CHROMA_DIR}")


def build_database() -> None:
    records = json.loads(CLEAN_FILE.read_text(encoding="utf-8"))
    DB_FILE.unlink(missing_ok=True)  # always rebuild from scratch

    with sqlite3.connect(DB_FILE) as db:
        db.executescript(SCHEMA)
        for record in records:
            row = {k: json.dumps(v, ensure_ascii=False) if k in JSON_FIELDS else v
                   for k, v in record.items()}
            columns = ", ".join(row)
            placeholders = ", ".join(f":{k}" for k in row)
            db.execute(f"INSERT INTO properties ({columns}) VALUES ({placeholders})", row)
            db.execute(
                "INSERT INTO properties_fts (id, title, location, details, description) "
                "VALUES (:id, :title, :location, :details, :description)",
                search_text(record),
            )
    print(f"Saved {len(records)} properties to {DB_FILE}")


def connect() -> sqlite3.Connection:
    """Open the database read-only. Rows behave like dicts: row["title"]."""
    db = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def row_to_dict(row: sqlite3.Row) -> dict:
    """Turn a database row back into a property record (JSON fields decoded)."""
    record = dict(row)
    for field in JSON_FIELDS:
        if field in record and isinstance(record[field], str):
            record[field] = json.loads(record[field])
    return record


if __name__ == "__main__":
    build_database()
    build_vector_index(json.loads(CLEAN_FILE.read_text(encoding="utf-8")))
