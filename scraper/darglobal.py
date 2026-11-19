"""DarGlobal scraper - collects luxury development projects from darglobal.co.uk.

How it works
------------
DarGlobal is protected by an anti-bot service (Imperva) that blocks normal
HTTP requests and even automated browsers. So we fetch pages through the
Firecrawl API (https://firecrawl.dev), which handles that for us.

1. Fetch the /projects page and read every project card on it (~38): name,
   location, image and link. The cards are saved to
   data/raw/darglobal_projects.json, because some project pages don't state
   their location in a standard place but the card always does.
2. Fetch each project page and keep:
     * page_data - the JSON the website embeds in the page (<script id="__NEXT_DATA__">),
                   which holds the title, location, unit types, amenities, FAQs, etc.
     * markdown  - the readable page text, as a fallback for pages whose JSON
                   uses a different layout (landing pages, "one-of-one" pages).
3. Save one line per project to data/raw/darglobal.jsonl.

Needs FIRECRAWL_API_KEY in .env. Each page costs 1 Firecrawl credit.

Run:
    uv run python -m scraper.darglobal
"""

import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

from scraper.storage import already_scraped, append_record, raw_file

BASE_URL = "https://darglobal.co.uk"
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/scrape"
OUTPUT = raw_file("darglobal")
CARDS_OUTPUT = OUTPUT.with_name("darglobal_projects.json")

# One project card on the /projects page, as Firecrawl renders it in markdown:
#   [![<name>](<image>)\
#   \
#   <location>\
#   \
#   **<name>**](<url> "<name>")
PROJECT_CARD = re.compile(
    r"\[!\[(?P<name>[^\]]*)\]\((?P<image>[^)]*)\)[\\\s]*"
    r"(?P<location>(?:\\\||[^\\\n])+?)[\\\s]*\*\*.*?\]\((?P<url>https://darglobal\.co\.uk/[^\s)]+)",
    re.S,
)  # a location may contain an escaped pipe, e.g. "AIDA \| Oman"

# Parts of the embedded page JSON that are the same on every page (menus,
# footer, forms) or only link to other projects. We drop them to keep files small.
SHARED_PAGE_PARTS = {"globalData", "thankyouPageData", "thankyouData", "otherProjectData", "isMobile"}


def firecrawl_scrape(url: str, formats: list[str], retries: int = 3) -> dict:
    """Fetch one page through Firecrawl and return its result ('data' object)."""
    headers = {"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}"}
    body = {"url": url, "formats": formats, "waitFor": 2000, "onlyMainContent": True}

    for attempt in range(1, retries + 1):
        try:
            response = httpx.post(FIRECRAWL_URL, json=body, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()["data"]
        except httpx.HTTPError as error:
            print(f"  attempt {attempt} failed for {url}: {error}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {url}")


def get_project_cards() -> list[dict]:
    """Return name, location, image and url of every project card on /projects."""
    data = firecrawl_scrape(f"{BASE_URL}/projects", formats=["markdown"])
    cards = {}
    for match in PROJECT_CARD.finditer(data["markdown"]):
        card = {key: value.replace("\\|", "|").strip() for key, value in match.groupdict().items()}
        cards.setdefault(card["url"], card)  # keep the first card if a link repeats
    return list(cards.values())


def extract_page_data(raw_html: str) -> dict:
    """Return the page's embedded JSON (Next.js pageProps) without shared parts."""
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', raw_html, re.S)
    if not match:
        return {}
    page_props = json.loads(match.group(1))["props"]["pageProps"]
    page_data = {k: v for k, v in page_props.items() if k not in SHARED_PAGE_PARTS}

    # Translations and edit history of the project are not needed.
    attributes = page_data.get("projectDetailsData", {}).get("attributes", {})
    attributes.pop("localizations", None)
    attributes.pop("versions", None)
    return page_data


def main() -> None:
    load_dotenv()
    cards = get_project_cards()
    CARDS_OUTPUT.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(cards)} project cards to {CARDS_OUTPUT}")

    project_urls = [card["url"] for card in cards]
    done = already_scraped(OUTPUT)
    todo = [u for u in project_urls if u not in done]
    print(f"Found {len(project_urls)} projects, {len(done)} already saved, {len(todo)} to scrape")

    for i, url in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {url}")
        try:
            data = firecrawl_scrape(url, formats=["rawHtml", "markdown"])
        except RuntimeError as error:
            print(f"  skipped: {error}")
            continue
        append_record(OUTPUT, {
            "url": url,
            "page_title": data.get("metadata", {}).get("title"),
            "page_data": extract_page_data(data.get("rawHtml", "")),
            "markdown": data.get("markdown", ""),
        })

    print(f"Done. {len(already_scraped(OUTPUT))} projects saved in {OUTPUT}")


if __name__ == "__main__":
    main()
