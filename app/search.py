"""Module 4 - Hybrid search (the "retrieval" in RAG).

Given a search query, find the few properties the AI should look at.

Step 1 - Filters. Read exact conditions from the query with simple rules, e.g.
    "3 bedroom apartment for rent in Riyadh under 60k"
      -> purpose=rent, city=Riyadh, property_type=apartment, bedrooms=3, price_max=60000
    Only properties that satisfy them are considered (SQLite, app/database.py).
Step 2 - Two rankings of those properties:
    * keyword ranking  - SQLite full-text search on the remaining words
                         ("beach", "golden visa", "Trump")
    * meaning ranking  - Chroma vector search on the whole query, which also
                         catches synonyms and typos ("seaside", "ocenfront")
Step 3 - Merge the two rankings with Reciprocal Rank Fusion: each property gets
    1/(60 + its position) from every ranking it appears in, and we sort by the
    total. Properties ranked well by either method rise to the top.
Step 4 - Return the top matches, how many matched in total, and price
    statistics, so the AI can also answer "how many" and "average price".
If nothing matches, we drop the price and bedroom filters and return close
alternatives, clearly marked as such.
"""

import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from app.database import chroma_collection, connect, row_to_dict

RRF_K = 60             # standard constant for Reciprocal Rank Fusion
CANDIDATES = 40        # how many results each ranking contributes
MAX_DISTANCE = 0.55    # with no filters/keywords, only keep close meaning matches

# ---------------------------------------------------------------- vocabulary

CITY_ALIASES = {
    "mecca": "Makkah", "makkah": "Makkah", "medina": "Madinah", "madina": "Madinah",
    "madinah": "Madinah", "jeddah": "Jeddah", "jedda": "Jeddah", "jiddah": "Jeddah",
    "riyadh": "Riyadh", "dammam": "Dammam", "khobar": "Khobar", "taif": "Taif",
    "rak": "Ras Al Khaimah", "ras al khaimah": "Ras Al Khaimah", "dubai": "Dubai",
    "muscat": "Muscat", "doha": "Doha", "london": "London",
}
COUNTRY_ALIASES = {
    "uae": "United Arab Emirates", "emirates": "United Arab Emirates",
    "saudi": "Saudi Arabia", "ksa": "Saudi Arabia", "oman": "Oman", "qatar": "Qatar",
    "spain": "Spain", "uk": "United Kingdom", "united kingdom": "United Kingdom",
    "england": "United Kingdom", "britain": "United Kingdom",
}
TYPE_ALIASES = {
    "apartment": "apartment", "flat": "apartment", "villa": "villa", "house": "villa",
    "land": "land", "plot": "land", "floor": "floor", "building": "building",
    "penthouse": "penthouse", "duplex": "duplex", "office": "office", "shop": "shop",
    "warehouse": "warehouse", "townhouse": "townhouse", "residence": "residence",
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

STOP_WORDS = set("""
a an the and or of to in on at for from with without by near is are was be there any some
me my i we you your show find give list tell want looking look need get see search
what which who where how many much do does can could would should please about all
properties property listing listings home homes options available have has price prices
cost costs under below above over less more than between budget max maximum min minimum
sar aed riyal riyals usd dollar dollars million thousand k m bed beds bedroom bedrooms br bhk
sale buy buying purchase rent rental renting lease cheap cheapest cheaper expensive
lowest highest affordable al el new best good nice it its this that these those also
offer offers hello hi hey thanks thank know tell
first second third fourth fifth last one ones other another they them their those
previous above mentioned detail details info information
darglobal dar global wasalt project projects development developments
""".split())


@dataclass
class Filters:
    source: str | None = None         # "DarGlobal" or "Wasalt"
    kind: str | None = None           # "project" or "listing"
    purpose: str | None = None        # "sale" or "rent"
    city: str | None = None
    country: str | None = None
    property_type: str | None = None  # matched loosely: "villa" also finds "Luxury Villas"
    bedrooms: int | None = None
    price_min: float | None = None
    price_max: float | None = None
    sort: str | None = None           # "price_asc" or "price_desc"

    def active(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------- reading filters

def parse_amount(number: str, unit: str | None) -> float:
    """'1.5' + 'm' -> 1_500_000, '800' + 'k' -> 800_000, '60,000' -> 60_000."""
    value = float(number.replace(",", ""))
    unit = (unit or "").lower()
    if unit in ("k", "thousand"):
        value *= 1_000
    elif unit in ("m", "mn", "million"):
        value *= 1_000_000
    return value


AMOUNT = r"(?:sar|aed|usd|\$)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m|mn|million|thousand)?\b(?!\s*(?:bed|br|bhk|sq|m2))"


def last_match(pattern: str, text: str) -> re.Match | None:
    """The last place a regex matches in the text (None if it doesn't match)."""
    matches = list(re.finditer(pattern, text))
    return matches[-1] if matches else None


def last_mentioned(text: str, aliases: dict, plural: bool = False) -> str | None:
    """Return the value of the alias that appears LAST in the text (None if none appears)."""
    best_position, best_value = -1, None
    for alias, value in aliases.items():
        pattern = rf"\b{re.escape(alias)}{'s?' if plural else ''}\b"
        for match in re.finditer(pattern, text):
            if match.start() > best_position:
                best_position, best_value = match.start(), value
    return best_value


@lru_cache(maxsize=1)  # the database never changes while the app runs
def known_places() -> tuple[dict, dict]:
    """City and country names: the aliases above plus every name found in the database."""
    cities, countries = dict(CITY_ALIASES), dict(COUNTRY_ALIASES)
    for (city,) in get_db().execute("SELECT DISTINCT city FROM properties WHERE city IS NOT NULL"):
        cities.setdefault(city.lower(), city)
    for (country,) in get_db().execute("SELECT DISTINCT country FROM properties WHERE country IS NOT NULL"):
        countries.setdefault(country.lower(), country)
    return cities, countries


def parse_filters(question: str) -> Filters:
    """Read structured filters from a free-text question."""
    q = question.lower()
    f = Filters()
    cities, countries = known_places()

    if re.search(r"\bdar\s*-?\s*global\b", q):
        f.source = "DarGlobal"
    elif "wasalt" in q:
        f.source = "Wasalt"
    if re.search(r"\b(projects?|developments?|developer)\b", q) and f.source != "Wasalt":
        f.kind = "project"

    wants_rent = re.search(r"\b(rent|rental|renting|lease|leasing)\b", q)
    wants_sale = re.search(r"\b(buy|buying|purchase|sale|selling)\b", q)
    if wants_rent and not wants_sale:
        f.purpose = "rent"
    elif wants_sale and not wants_rent:
        f.purpose = "sale"

    # If a place or type is mentioned twice, the last mention wins, so a combined
    # follow-up like "...in Riyadh ... what about Jeddah?" means Jeddah.
    f.city = last_mentioned(q, cities)
    if not f.city:
        f.country = last_mentioned(q, countries)
    f.property_type = last_mentioned(q, TYPE_ALIASES, plural=True)

    beds = re.search(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s*-?\s*(?:bed(?:room)?s?|br|bhk)\b", q)
    if beds:
        f.bedrooms = int(NUMBER_WORDS.get(beds.group(1), beds.group(1)))

    between = re.search(rf"\bbetween\s+{AMOUNT}\s*(?:and|to|-)\s*{AMOUNT}", q)
    if between:
        f.price_min = parse_amount(between.group(1), between.group(2))
        f.price_max = parse_amount(between.group(3), between.group(4))
    else:
        # Last mention wins here too ("under 60k ... what about under 40k?").
        below = last_match(rf"\b(?:under|below|less than|up to|upto|max(?:imum)?|within|budget(?: of)?|cheaper than)\s+{AMOUNT}", q)
        above = last_match(rf"\b(?:above|over|more than|at least|min(?:imum)?|starting(?: from)?)\s+{AMOUNT}", q)
        if below:
            f.price_max = parse_amount(below.group(1), below.group(2))
        if above:
            f.price_min = parse_amount(above.group(1), above.group(2))

    if re.search(r"\b(cheapest|lowest|affordable|cheap|budget)\b", q):
        f.sort = "price_asc"
    elif re.search(r"\b(most expensive|highest|priciest|luxurious)\b", q):
        f.sort = "price_desc"
    return f


def keywords(question: str, f: Filters) -> list[str]:
    """Words left over after removing filler words and words used as filters."""
    cities, countries = known_places()
    used = {str(v).lower() for v in f.active().values()}
    used |= set(cities) | set(countries) | set(TYPE_ALIASES) | set(NUMBER_WORDS)
    words = re.findall(r"[a-z\u0600-\u06ff]+", question.lower())
    result = []
    for word in words:
        singular = word[:-1] if word.endswith("s") else word
        if len(word) < 3 or word in STOP_WORDS or word in used or singular in used:
            continue
        if word not in result:
            result.append(word)
    return result


# ------------------------------------------------------------ the database

def where_clause(f: Filters) -> tuple[str, list]:
    """Turn filters into a SQL WHERE clause and its parameters."""
    conditions, params = [], []
    for column in ("source", "kind", "purpose", "city", "country"):
        value = getattr(f, column)
        if value:
            conditions.append(f"p.{column} = ?")
            params.append(value)
    if f.property_type:
        conditions.append("(p.property_type LIKE ? OR p.title LIKE ? OR p.unit_types LIKE ?)")
        params += [f"%{f.property_type}%"] * 3
    if f.bedrooms is not None:
        conditions.append("p.bedrooms = ?")
        params.append(f.bedrooms)
    if f.price_min is not None:
        conditions.append("p.price >= ?")
        params.append(f.price_min)
    if f.price_max is not None:
        conditions.append("p.price <= ?")
        params.append(f.price_max)
    return (" AND ".join(conditions) or "1 = 1"), params


def order_clause(f: Filters) -> str:
    if f.sort == "price_asc":
        return "p.price IS NULL, p.price ASC"
    if f.sort == "price_desc":
        return "p.price IS NULL, p.price DESC"
    return "p.kind = 'project' DESC, p.price ASC"


def keyword_query(db, f: Filters, words: list[str], limit: int) -> dict:
    """Keyword ranking: find matches for filters + keywords with SQLite.

    Keywords are tried first. If they match nothing but there are filters,
    we fall back to the filters alone (e.g. a misspelt word shouldn't hide
    every villa in Jeddah). Returns total, price stats and the ranked results.
    """
    where, params = where_clause(f)
    searches = []
    if words:
        match = " OR ".join(f'"{w}"' for w in words)
        searches.append((
            "properties_fts JOIN properties p ON p.id = properties_fts.id",
            f"properties_fts MATCH ? AND {where}",
            [match, *params],
            # Weights per column: id, title, location, details, description.
            order_clause(f) if f.sort else "bm25(properties_fts, 0, 8.0, 4.0, 2.0, 1.0)",
        ))
    if f.active():
        searches.append(("properties p", where, params, order_clause(f)))

    for sql_from, sql_where, sql_params, order in searches:
        total = db.execute(f"SELECT COUNT(*) FROM {sql_from} WHERE {sql_where}", sql_params).fetchone()[0]
        if total == 0:
            continue
        rows = db.execute(
            f"SELECT p.* FROM {sql_from} WHERE {sql_where} ORDER BY {order} LIMIT ?",
            [*sql_params, limit],
        ).fetchall()
        # Count and min/average/max price of all matches, separately for sale and rent.
        stats = db.execute(
            f"""SELECT p.purpose, p.currency, COUNT(*) AS listings, MIN(p.price) AS min_price,
                       ROUND(AVG(p.price)) AS avg_price, MAX(p.price) AS max_price
                FROM {sql_from} WHERE {sql_where} AND p.price IS NOT NULL
                GROUP BY p.purpose, p.currency""",
            sql_params,
        ).fetchall()
        return {"total_matches": total, "stats": [dict(s) for s in stats],
                "results": [row_to_dict(r) for r in rows]}

    return {"total_matches": 0, "stats": [], "results": []}


def meaning_query(query: str, candidate_ids: list[str] | None, limit: int) -> list[tuple[str, float]]:
    """Meaning ranking: ask Chroma for the properties closest in meaning to the query.

    candidate_ids limits the search to properties that passed the filters
    (None = search everything). Returns [(id, distance)], closest first.
    """
    if candidate_ids is not None and not candidate_ids:
        return []
    result = get_vectors().query(
        query_texts=[query],
        ids=candidate_ids,
        n_results=min(limit, len(candidate_ids) if candidate_ids else limit),
        include=["distances"],
    )
    return list(zip(result["ids"][0], result["distances"][0]))


def fuse(*rankings: list[str]) -> list[str]:
    """Reciprocal Rank Fusion: merge several ranked id lists into one."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0) + 1 / (RRF_K + position + 1)
    return sorted(scores, key=scores.get, reverse=True)


def fetch(db, ids: list[str]) -> list[dict]:
    """Load full records for ids, keeping their order."""
    if not ids:
        return []
    rows = db.execute(f"SELECT * FROM properties WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
    by_id = {row["id"]: row_to_dict(row) for row in rows}
    return [by_id[pid] for pid in ids if pid in by_id]


def hybrid_query(db, query: str, f: Filters, words: list[str], limit: int) -> dict:
    """Filters + keyword ranking + meaning ranking, merged."""
    found = keyword_query(db, f, words, CANDIDATES)
    keyword_ids = [p["id"] for p in found["results"]]

    if f.sort:  # "cheapest"/"most expensive": price order matters more than meaning
        found["results"] = found["results"][:limit]
        return found

    where, params = where_clause(f)
    candidate_ids = None
    if f.active():
        candidate_ids = [row[0] for row in db.execute(f"SELECT p.id FROM properties p WHERE {where}", params)]
    meaning = meaning_query(query, candidate_ids, CANDIDATES)

    if found["total_matches"] == 0:
        # Nothing matched the keywords or filters: keep only close meaning matches.
        close = [pid for pid, distance in meaning if f.active() or distance <= MAX_DISTANCE]
        return {"total_matches": len(close), "stats": [], "results": fetch(db, close[:limit]),
                "meaning_only": bool(close)}

    meaning_ids = [pid for pid, _ in meaning]
    found["results"] = fetch(db, fuse(keyword_ids, meaning_ids)[:limit])
    return found


_db = None
_vectors = None


def get_db():
    global _db
    if _db is None:
        _db = connect()
    return _db


def get_vectors():
    global _vectors
    if _vectors is None:
        _vectors = chroma_collection()
    return _vectors


def search(query: str, limit: int = 8) -> dict:
    """Find the properties that best answer a search query.

    Returns a dict with:
      filters        filters read from the query
      keywords       words used for keyword search
      total_matches  how many properties match
      stats          price statistics over all matches
      relaxed        True if there was no exact match and these are close alternatives
      results        the top `limit` properties
    """
    db = get_db()
    f = parse_filters(query)
    words = keywords(query, f)

    found = hybrid_query(db, query, f, words, limit)
    relaxed = found.pop("meaning_only", False)
    if found["total_matches"] == 0 and (f.bedrooms is not None or f.price_min or f.price_max):
        f.bedrooms = f.price_min = f.price_max = None
        found = hybrid_query(db, query, f, words, limit)
        found.pop("meaning_only", None)
        relaxed = found["total_matches"] > 0

    return {"filters": f.active(), "keywords": words, "relaxed": relaxed, **found}


def dataset_overview() -> str:
    """A short description of everything in the database, for the AI's instructions."""
    db = get_db()
    projects = db.execute(
        "SELECT country, COUNT(*) FROM properties WHERE source = 'DarGlobal' GROUP BY country ORDER BY 2 DESC"
    ).fetchall()
    listings = db.execute(
        "SELECT purpose, COUNT(*) FROM properties WHERE source = 'Wasalt' GROUP BY purpose"
    ).fetchall()
    cities = db.execute(
        "SELECT city, COUNT(*) FROM properties WHERE source = 'Wasalt' GROUP BY city ORDER BY 2 DESC LIMIT 10"
    ).fetchall()
    return (
        f"DarGlobal: {sum(n for _, n in projects)} luxury development projects in "
        + ", ".join(f"{c} ({n})" for c, n in projects)
        + ". DarGlobal does not publish unit prices on its website.\n"
        + f"Wasalt: {sum(n for _, n in listings)} property listings in Saudi Arabia ("
        + ", ".join(f"{n} for {p}" for p, n in listings)
        + "), mainly in " + ", ".join(f"{c} ({n})" for c, n in cities) + "."
    )
