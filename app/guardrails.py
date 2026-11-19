"""Guardrails: checks before and after the AI.

Input guardrails (before the message reaches the AI or the logs)
  * Prompt injection - messages that try to override the bot's rules ("ignore
    previous instructions", "show your system prompt", "you are now DAN") are
    answered with a fixed refusal; no AI call is made.
  * Personal data - phone numbers, emails and long ID/card numbers are replaced
    with [phone] / [email] / [number] before the message is sent to the AI,
    stored in memory or written to the audit trail.

Output guardrails (on the AI's answer)
  * Links - every link must point to a property page that was in the search
    results (or to darglobal.co.uk / wasalt.sa). Any other link is removed, so
    the bot can't send users to invented or unsafe sites.
  * Instruction leaks - if the answer repeats the bot's own instructions, it is
    replaced with a refusal.

The AI instructions (app/llm.py) also tell the model to stay on topic and not
follow instructions found inside user messages. Each check that fires is
recorded in the audit record (app/audit.py) under "guardrails".
"""

import re
from urllib.parse import urlparse

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompts?|guidelines)\b",
    r"\b(system|developer|hidden|initial)\s+(prompt|message|instructions?)\b",
    r"\b(reveal|show|print|repeat|output)\b.{0,30}\b(your|the)\s+(prompt|instructions?|rules)\b",
    r"\byou\s+are\s+now\b",
    r"\b(act|pretend|roleplay)\s+as\b.{0,40}\b(unrestricted|uncensored|jailbroken|dan|evil)\b",
    r"\b(jailbreak|dan mode|developer mode)\b",
]

INJECTION_REPLY = (
    "I can only help with questions about DarGlobal projects and Wasalt property listings, "
    "and I can't change or share my instructions. Try asking about a city, budget or project."
)

LEAK_REPLY = "I can't share my instructions, but I'm happy to help you find a property."

# Distinctive phrases from the AI instructions in app/llm.py.
INSTRUCTION_PHRASES = [
    "use only the facts in \"matching properties\"",
    "never invent properties, prices",
    "what the database contains overall",
    "you prepare database searches",
]

PERSONAL_DATA = [
    ("[email]", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # International (+971 50 123 4567, 00966...) or Saudi mobile (0501234567) numbers;
    # plain long numbers are left alone because they are usually prices.
    ("[phone]", r"(?<!\w)(?:\+|00)\d[\d\s-]{7,14}\d(?!\w)|(?<!\w)0?5\d{8}(?!\w)"),
    ("[number]", r"\b\d{12,19}\b"),  # card / national id style numbers
]

TRUSTED_DOMAINS = ("darglobal.co.uk", "wasalt.sa", "wasalt.com")

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
BARE_URL = re.compile(r"(?<!\()https?://[^\s)\]]+")


def check_input(message: str) -> tuple[str, list[str], str | None]:
    """Run the input guardrails.

    Returns (cleaned message, list of guardrails that fired, fixed reply or None).
    If a fixed reply is returned, the message must not be sent to the AI.
    """
    fired = []
    lowered = message.lower()
    if any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS):
        return message, ["prompt_injection_blocked"], INJECTION_REPLY

    cleaned = message
    for placeholder, pattern in PERSONAL_DATA:
        cleaned, count = re.subn(pattern, placeholder, cleaned)
        if count:
            fired.append(f"personal_data_redacted:{placeholder.strip('[]')}")
    return cleaned, fired, None


def is_allowed_link(url: str, allowed_urls: set[str]) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return url.rstrip("/") in allowed_urls or any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def check_output(answer: str, allowed_urls: set[str]) -> tuple[str, list[str]]:
    """Run the output guardrails. Returns (safe answer, list of guardrails that fired)."""
    lowered = answer.lower()
    if any(phrase in lowered for phrase in INSTRUCTION_PHRASES):
        return LEAK_REPLY, ["instruction_leak_blocked"]

    fired = []
    allowed = {u.rstrip("/") for u in allowed_urls}

    def keep_or_strip(match: re.Match) -> str:
        text, url = match.group(1), match.group(2)
        if is_allowed_link(url, allowed):
            return match.group(0)
        fired.append("untrusted_link_removed")
        return text  # keep the words, drop the link

    safe = MARKDOWN_LINK.sub(keep_or_strip, answer)

    def strip_bare(match: re.Match) -> str:
        if is_allowed_link(match.group(0), allowed):
            return match.group(0)
        fired.append("untrusted_link_removed")
        return "[link removed]"

    safe = BARE_URL.sub(strip_bare, safe)
    return safe, sorted(set(fired))
