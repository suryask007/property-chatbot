"""Clean one raw DarGlobal project into the common record shape.

Most project pages (26 of 38) share one layout whose JSON holds well-labelled
sections (about, amenities, why invest, location, FAQs). We read those
directly. The rest (landing pages, "one-of-one" pages) use other layouts, so
for them we fall back to the readable page text from Firecrawl.

Name and location always come from the project's card on the /projects page,
because every project has one and they are consistent.
"""

from processing.record import new_record
from processing.text import clean_markdown, tidy, to_number

# DarGlobal writes the same fact with different labels ("Status", "STATUS",
# "Stage", "Unit Type", "Units"...). Map each label to one record field.
FACT_FIELDS = {
    "property type": "property_type",
    "status": "status",
    "stage": "status",
    "unit type": "unit_types",
    "units": "unit_types",
    "area (sqm)": "area_text",
    "area(sqm)": "area_text",
    "area (sqft)": "area_text",
    "location": None,  # already known from the project card
}

# Words in a card location -> country.
COUNTRY_WORDS = {
    "United Arab Emirates": ["uae", "united arab emirates", "dubai", "rak"],
    "Saudi Arabia": ["ksa", "saudi", "riyadh", "jeddah"],
    "Oman": ["oman", "muscat", "aida"],
    "Qatar": ["qatar", "doha"],
    "Spain": ["spain"],
    "United Kingdom": ["united kingdom", "england", "london"],
}

# Words in a card location -> city.
CITY_WORDS = {
    "Dubai": "Dubai", "RAK": "Ras Al Khaimah", "Muscat": "Muscat", "Aida": "Muscat",
    "Doha": "Doha", "Simaisma": "Simaisma", "London": "London", "Riyadh": "Riyadh",
    "Jeddah": "Jeddah", "Benahav": "Benahavís", "Cortesin": "Casares",
}


def guess_country(location: str) -> str | None:
    lowered = location.lower()
    for country, words in COUNTRY_WORDS.items():
        if any(word in lowered for word in words):
            return country
    return None


def guess_city(location: str) -> str | None:
    lowered = location.lower()
    for word, city in CITY_WORDS.items():
        if word.lower() in lowered:
            return city
    return None


def rich_text_to_plain(blocks) -> str:
    """FAQ answers are stored as nested rich-text blocks; collect all their text."""
    if isinstance(blocks, list):
        return " ".join(rich_text_to_plain(block) for block in blocks).strip()
    if isinstance(blocks, dict):
        return blocks.get("text") or rich_text_to_plain(blocks.get("children", []))
    return ""


def name_value_list(items) -> list[str]:
    """Turn [{"name": "Golden Visa:", "value": "Available"}] into ["Golden Visa: Available"]."""
    lines = []
    for item in items or []:
        name, value = tidy(item.get("name") or ""), tidy(item.get("value") or "")
        lines.append(f"{name} {value}".strip())
    return [line for line in lines if line]


def read_standard_layout(attributes: dict) -> dict:
    """Extract fields from the standard project-page JSON layout."""
    details = attributes["ProjectDetails"]
    about = details.get("AboutProject") or {}
    location = details.get("loaction") or {}  # (sic) the website's own spelling
    coordinates = location.get("locationDetailsWithImages") or {}
    why_invest = details.get("whyInvest") or {}

    fields, facts = {}, {}
    for fact in about.get("AboutDetailsType") or []:
        label, value = tidy(fact["name"]), tidy(fact.get("value") or "")
        if label.lower() in FACT_FIELDS and FACT_FIELDS[label.lower()] is None:
            continue
        field = FACT_FIELDS.get(label.lower())
        if field == "area_text" and "sqft" in label.lower():
            value += " sq ft"
        elif field == "area_text":
            value += " sqm"
        if field:
            fields[field] = value
        elif value:
            facts[label.title()] = value

    invest_points = name_value_list(why_invest.get("investlist"))
    if invest_points:
        facts["Why Invest"] = "; ".join(invest_points)

    description_parts = [about.get("title"), about.get("description"), location.get("description")]
    faqs = [
        {"question": tidy(faq["question"]), "answer": rich_text_to_plain(faq.get("answer"))}
        for block in attributes.get("faqContent") or []
        for faq in block.get("faqs") or []
    ]
    return {
        **fields,
        "facts": facts,
        "amenities": [item["name"] for item in (details.get("Amenities") or {}).get("amenitiesList") or []],
        "description": tidy("\n".join(part for part in description_parts if part)),
        "faqs": faqs,
        "latitude": to_number(coordinates.get("lat")),
        "longitude": to_number(coordinates.get("long")),
    }


def clean_project(raw: dict, card: dict) -> dict:
    """Convert one line of data/raw/darglobal.jsonl into a clean record."""
    attributes = raw["page_data"].get("projectDetailsData", {}).get("attributes") or {}
    if "ProjectDetails" in attributes:
        extracted = read_standard_layout(attributes)
    else:
        extracted = {"description": clean_markdown(raw["markdown"])}

    slug = raw["url"].removeprefix("https://darglobal.co.uk/").strip("/").replace("/", "-").lower()
    location = card["location"]
    return new_record(
        id=f"darglobal-{slug}",
        source="DarGlobal",
        kind="project",
        title=card["name"],
        country=guess_country(location),
        city=guess_city(location),
        location_text=location,
        url=raw["url"],
        image_url=card["image"],
        scraped_at=raw["scraped_at"],
        **extracted,
    )
