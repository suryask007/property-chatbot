"""Wasalt scraper - collects property listings (sale and rent) from wasalt.sa.

How it works
------------
1. Download Wasalt's public sitemap. It lists every property page URL
   (about 60,000), so we don't need to touch the search pages that
   robots.txt asks bots to avoid.
2. Pick a random sample of sale and rent listings.
3. Open each page in your installed Google Chrome and read the JSON that the
   website itself embeds in the page (<script id="__NEXT_DATA__">). That JSON
   already holds the title, price, city, bedrooms, etc., so no fragile HTML
   parsing is needed.
4. Save one line per listing to data/raw/wasalt.jsonl.

Why a visible Chrome window? Wasalt is behind Cloudflare, which blocks plain
HTTP requests and every headless (invisible) browser we tried. A normal,
visible Chrome window passes. Scraping is a one-time job run on a laptop, so a
window opening for a few minutes is fine. Leave it alone; it closes at the end.

Run:
    uv run python -m scraper.wasalt --sale 600 --rent 400
"""

import argparse
import asyncio
import gzip
import json
import random
import re

import httpx
from playwright.async_api import Page, async_playwright

from scraper.storage import already_scraped, append_record, raw_file

SITEMAP_URL = "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz"
USER_AGENT = "Mozilla/5.0 (compatible; property-research-bot)"
OUTPUT = raw_file("wasalt")


def load_sitemap_urls() -> list[str]:
    """Download the sitemap and return all property page URLs in it."""
    response = httpx.get(SITEMAP_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    content = response.content
    if content[:2] == b"\x1f\x8b":  # the file is named .gz but is sometimes plain XML
        content = gzip.decompress(content)
    return re.findall(r"<loc>(.*?)</loc>", content.decode("utf-8"))


def pick_sample(urls: list[str], n_sale: int, n_rent: int, seed: int = 42) -> list[str]:
    """Randomly choose n_sale sale listings and n_rent rent listings.

    A fixed seed makes the sample reproducible between runs.
    """
    sale = [u for u in urls if "/property/sale/" in u]
    rent = [u for u in urls if "/property/rent/" in u]
    print(f"Sitemap: {len(sale)} sale + {len(rent)} rent listings")
    rng = random.Random(seed)
    return rng.sample(sale, min(n_sale, len(sale))) + rng.sample(rent, min(n_rent, len(rent)))


async def scrape_listing(page: Page, url: str) -> dict | None:
    """Open one listing page and return its embedded property JSON (or None)."""
    response = await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    if response is None or response.status != 200:
        print(f"  skip ({response.status if response else 'no response'}): {url}")
        return None

    next_data = await page.evaluate(
        "() => document.getElementById('__NEXT_DATA__')?.textContent"
    )
    if not next_data:
        print(f"  skip (no page data, maybe a bot check): {url}")
        return None

    listing = json.loads(next_data)["props"]["pageProps"].get("propertyDetailsV3")
    if not listing or not listing.get("propertyInfo"):
        print(f"  skip (listing removed): {url}")
        return None
    return {"url": url, "listing": listing}


async def worker(queue: asyncio.Queue, context, counter: dict, delay: float):
    """Use one browser tab to take URLs from the queue, scrape them, and save the results."""
    page = await context.new_page()
    # Don't download images, fonts or videos: faster for us, lighter for the site.
    await page.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type in {"image", "media", "font"}
        else route.continue_(),
    )

    while not queue.empty():
        url = await queue.get()
        try:
            record = await scrape_listing(page, url)
            if record:
                append_record(OUTPUT, record)
                counter["saved"] += 1
                if counter["saved"] % 25 == 0:
                    print(f"  saved {counter['saved']} listings")
        except Exception as error:  # one bad page should not stop the whole run
            print(f"  error on {url}: {error}")
        await asyncio.sleep(delay)  # be polite: pause between requests

    await page.close()


async def run(n_sale: int, n_rent: int, concurrency: int, delay: float) -> None:
    urls = pick_sample(load_sitemap_urls(), n_sale, n_rent)
    done = already_scraped(OUTPUT)
    todo = [u for u in urls if u not in done]
    print(f"{len(done)} already saved, {len(todo)} to scrape -> {OUTPUT}")

    queue: asyncio.Queue = asyncio.Queue()
    for url in todo:
        queue.put_nowait(url)

    counter = {"saved": 0}
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome",  # the Google Chrome installed on this computer
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(locale="en-US")
        await asyncio.gather(
            *(worker(queue, context, counter, delay) for _ in range(concurrency))
        )
        await browser.close()
    print(f"Done. Saved {counter['saved']} new listings.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Wasalt property listings.")
    parser.add_argument("--sale", type=int, default=600, help="number of sale listings")
    parser.add_argument("--rent", type=int, default=400, help="number of rent listings")
    parser.add_argument("--concurrency", type=int, default=3, help="browser tabs in parallel")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests per tab")
    args = parser.parse_args()
    asyncio.run(run(args.sale, args.rent, args.concurrency, args.delay))


if __name__ == "__main__":
    main()
