"""Small text helpers shared by the cleaners."""

import html
import re


def html_to_text(value: str | None) -> str:
    """Turn an HTML snippet like '<p>Nice <b>villa</b></p>' into plain text."""
    if not value:
        return ""
    value = re.sub(r"<(br|/p|/li|/h\d)\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    return tidy(html.unescape(value))


def tidy(value: str) -> str:
    """Collapse repeated spaces and blank lines."""
    value = re.sub(r"[ \t\xa0]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return value.strip()


def to_number(value) -> float | None:
    """Read a number from values like 850000, '850000', '1,200' or ''. Returns None if absent."""
    if value is None or value == "":
        return None
    try:
        number = float(str(value).replace(",", ""))
    except ValueError:
        return None
    return number if number > 0 else None


def to_int(value) -> int | None:
    number = to_number(value)
    return int(number) if number is not None else None


def clean_markdown(markdown: str, max_chars: int = 6000) -> str:
    """Make Firecrawl page markdown readable plain text.

    Removes images, keeps link text without the URL, and cuts off the
    contact form and cookie banner that appear at the bottom of every page.
    """
    for footer in ("## Register Your Interest", "## We value our privacy"):
        markdown = markdown.split(footer)[0]
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", markdown)          # images
    markdown = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown)       # links -> text
    markdown = markdown.replace("\\", " ")
    markdown = re.sub(r"[#*_>]+", " ", markdown)                        # markdown symbols
    return tidy(markdown)[:max_chars]
