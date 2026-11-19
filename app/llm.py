"""Module 5 - The AI steps, using free OpenRouter models through LiteLLM.

Each chat message uses up to two AI calls:

  1. plan_search()      Reads the conversation + new message and decides:
                          - does this need a property search? ("hi" doesn't)
                          - if so, a complete search query, with references
                            resolved: "tell me more about the second one" ->
                            "D-Villas At Jumeirah Golf Estates"
  2. generate_answer_stream()
                        Writes the reply from:
                          - instructions (use ONLY the data given, link sources)
                          - the properties found by app/search.py
                          - the recent conversation (app/memory.py)
                        and streams it piece by piece, so the first words
                        appear on screen after about a second.

How models are called
---------------------
LiteLLM gives one interface (`completion(...)`, OpenAI style) for 100+ AI
providers. We use its Router, which holds a list of models and *falls back*
automatically: if the first free model is busy or rate-limited, the next one
answers. Switching to another provider later (OpenAI, Gemini, Groq...) only
means changing the model names below.

Settings (environment variables / .env):
  OPENROUTER_API_KEY   required
  OPENROUTER_MODELS    optional, comma-separated list overriding DEFAULT_MODELS
  OPENROUTER_API_BASE  optional, another address for the API (used for tests)
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
                       optional, send every AI call to Langfuse for tracing
  LANGFUSE_BASE_URL    optional (or LANGFUSE_HOST), defaults to https://cloud.langfuse.com
"""

import json
import logging
import os
import re
from collections.abc import Iterator
from functools import lru_cache

import litellm
from litellm import Router

# LiteLLM logs a long error for every model that fails; app/main.py logs one
# short warning instead when all of them fail.
litellm.suppress_debug_info = True
for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

# Observability: if Langfuse keys are set, LiteLLM sends every AI call (prompt,
# answer, model, tokens, latency, errors) to Langfuse, grouped per chat message
# (trace) and per conversation (session). See https://langfuse.com
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
if LANGFUSE_ENABLED:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

# Free OpenRouter models, in order of preference.
DEFAULT_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free",  # OpenRouter picks any free model that is available
]

# This one refuses requests with reasoning switched off (see deployment()).
MODELS_THAT_MUST_REASON = {"openrouter/free"}

INSTRUCTIONS = """You are a helpful real-estate assistant. You answer questions using ONLY
data scraped from two public websites:
- DarGlobal (darglobal.co.uk): an international luxury property developer. Its records are
  development PROJECTS. DarGlobal does not publish unit prices, so never state a price for them.
- Wasalt (wasalt.sa): a Saudi property portal. Its records are individual LISTINGS for sale or
  rent, with prices in SAR (rent prices are per year unless another period is shown).

What the database contains overall:
{overview}

Rules:
1. Use only the facts in "Matching properties" and the overall summary above. Never invent
   properties, prices, sizes, amenities or contact details. If the data does not answer the
   question, say so plainly and suggest what the user could ask instead.
2. Mention properties by their exact title and include their link as a markdown link,
   e.g. [The Astera](https://darglobal.co.uk/the-astera).
3. If "Exact match: no" is shown, say there were no exact matches and present the results as
   the closest alternatives.
4. "Total matches" and "Price statistics" cover ALL matching properties in the database, not
   only the ones listed. Use them for "how many" and "average/cheapest/most expensive" questions.
   When "Search words" are shown, a match only needs ONE of the words, so never claim that all
   matches meet every part of the question; check the listed properties' details instead.
5. When there is no property data for this message, answer from the conversation so far. If the
   user refers to something earlier, use what was said and shown earlier in the conversation.
6. Be concise: a one-sentence answer, then a short bullet list (at most 5 properties unless asked
   for more) with title, location, price (or "price not published"), key facts and link.
7. Politely decline questions that are not about these properties or real estate.
8. Format prices with thousands separators, e.g. 1,250,000 SAR."""

PLANNER_INSTRUCTIONS = """You prepare database searches for a real-estate chatbot. The database has
DarGlobal luxury development projects (UAE, Oman, Saudi Arabia, Qatar, Spain, UK) and Wasalt
property listings in Saudi Arabia (for sale and rent).

Read the conversation and the user's latest message. Reply with JSON only, no other text:
{"search": true or false, "query": "..."}

- "search": false for greetings, thanks, small talk, questions about you, and general questions
  about DarGlobal, Wasalt or the dataset as a whole ("what is Wasalt?", "how many listings do you
  have?"). Then "query" is "".
- Otherwise "search": true and "query" is ONE complete, standalone search request in English that
  includes every detail it needs from earlier messages: project names, city, budget, bedrooms,
  sale or rent, property type. Resolve references such as "the second one", "it", "those",
  "cheaper ones", "the project I asked about before" into the actual names or criteria, using the
  property cards listed in the conversation. Correct spelling mistakes.
- If the user wants more about something discussed earlier ("tell me more about what I asked
  before"), search for that earlier property or topic: the most recent specific property question.
- If the question can be answered from the cards already shown (e.g. "which of those is the
  biggest?"), use "search": false.

Examples:
Conversation: user asked "tell me about azure ocenfront villas", then "tell me about wasalt"
Latest: "i want know more about before i asked question" -> {"search": true, "query": "Azure Oceanfront Villas Interiors by Fendi Casa"}
Conversation: user asked "DarGlobal projects in Dubai"; cards shown: 1. Trump International Hotel & Tower Dubai; 2. D-Villas At Jumeirah Golf Estates
Latest: "tell me more about the second one" -> {"search": true, "query": "D-Villas At Jumeirah Golf Estates"}
Conversation: user asked "3 bedroom apartments for rent in Riyadh"
Latest: "what about Jeddah under 50k?" -> {"search": true, "query": "3 bedroom apartments for rent in Jeddah under 50k"}
Latest: "hi" -> {"search": false, "query": ""}
Latest: "tell me about wasalt" -> {"search": false, "query": ""}
Latest: "azure ocenfront villas" -> {"search": true, "query": "Azure Oceanfront Villas"}"""


# ------------------------------------------------- writing data for the model

def format_property(number: int, p: dict) -> str:
    """Write one property as compact plain text for the model to read."""
    if p["price"]:
        price = f"{p['price']:,.0f} {p['currency'] or ''}" + (f" per {p['price_period'].removesuffix('ly')}" if p["price_period"] else "")
    else:
        price = "not published"
    fields = {
        "Source": f"{p['source']} ({p['kind']})",
        "For": p["purpose"],
        "Type": p["property_type"],
        "Location": p["location_text"] or ", ".join(filter(None, [p["district"], p["city"], p["country"]])),
        "Price": price,
        "Bedrooms": p["bedrooms"],
        "Bathrooms": p["bathrooms"],
        "Area": f"{p['area_sqm']:,.0f} sqm" if p["area_sqm"] else p["area_text"],
        "Unit types": p["unit_types"],
        "Status": p["status"],
        "Furnishing": p["furnishing"],
        "Amenities": ", ".join(p["amenities"][:12]) or None,
        "Facts": "; ".join(f"{k}: {v}" for k, v in p["facts"].items())[:700] or None,
        "Description": (p["description"] or "")[:600] or None,
        "FAQs": " | ".join(f"Q: {f['question']} A: {f['answer'][:250]}" for f in p["faqs"][:4]) or None,
        "Link": p["url"],
    }
    lines = [f"[{number}] {p['title']}"] + [f"  {k}: {v}" for k, v in fields.items() if v not in (None, "")]
    return "\n".join(lines)


def format_search(found: dict | None) -> str:
    """Write the search result (from app/search.py) as the data block for the model."""
    if found is None:
        return ("No property search was needed for this message. Reply conversationally, using "
                "the overall summary and the conversation so far.")
    if not found["results"]:
        return "Matching properties: none found for this question."
    stats = "; ".join(
        f"{s['purpose']}: {s['listings']} with price, min {s['min_price']:,.0f}, "
        f"average {s['avg_price']:,.0f}, max {s['max_price']:,.0f} {s['currency']}"
        for s in found["stats"]
    ) or "no prices available"
    header = [f"Filters understood: {json.dumps(found['filters'])}"]
    if found["keywords"]:
        header.append(f"Search words: {', '.join(found['keywords'])}")
    header += [
        f"Exact match: {'no' if found['relaxed'] else 'yes'}",
        f"Total matches: {found['total_matches']} (showing {len(found['results'])})",
        f"Price statistics: {stats}",
    ]
    properties = "\n\n".join(format_property(i, p) for i, p in enumerate(found["results"], 1))
    return "\n".join(header) + "\n\nMatching properties:\n" + properties


# ------------------------------------------------------ calling the models

def models() -> list[str]:
    custom = os.getenv("OPENROUTER_MODELS")
    return [m.strip() for m in custom.split(",") if m.strip()] if custom else DEFAULT_MODELS


def deployment(model: str) -> dict:
    """How LiteLLM should call one model ("openrouter/" tells it which provider)."""
    params = {
        "model": f"openrouter/{model}",
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "extra_headers": {"X-Title": "DarGlobal & Wasalt Property Chatbot"},  # name in OpenRouter's dashboard
    }
    if model not in MODELS_THAT_MUST_REASON:
        # Reasoning models otherwise "think" first, which is slow and can use up
        # max_tokens before any answer is written.
        params["extra_body"] = {"reasoning": {"enabled": False}}
    if os.getenv("OPENROUTER_API_BASE"):
        params["api_base"] = os.environ["OPENROUTER_API_BASE"]
    return {"model_name": model, "litellm_params": params}


@lru_cache(maxsize=1)
def router() -> Router:
    """A LiteLLM Router over the free models: try the first, fall back to the others in order."""
    names = models()
    return Router(
        model_list=[deployment(m) for m in names],
        fallbacks=[{names[0]: names[1:]}],
        num_retries=0,   # don't retry a busy model; move straight to the next one
        timeout=60,
    )


def clean(text: str) -> str:
    """Some reasoning models put their thinking in <think>...</think>; drop it."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def trace_metadata(step: str, trace: dict | None) -> dict:
    """Labels for Langfuse, so its traces line up with our audit records (app/audit.py)."""
    if not trace:
        return {}
    return {
        "trace_id": trace["request_id"],    # one trace per chat message
        "session_id": trace["session_id"],  # traces grouped per conversation
        "trace_name": "chat-message",
        "generation_name": step,            # "plan_search" or "answer"
    }


def plan_search(message: str, history: list[dict], trace: dict | None = None) -> dict:
    """Step 1: decide whether to search, and with what standalone query.

    Returns {"search": bool, "query": str}. Raises RuntimeError if no model answers
    or the reply isn't valid JSON (the caller then uses simple rules instead).
    `trace` ({"request_id", "session_id"}) labels the call for Langfuse.
    """
    transcript = "\n".join(f"{m['role']}: {m['content'][:900]}" for m in history) or "(no earlier messages)"
    try:
        response = router().completion(
            model=models()[0],
            messages=[
                {"role": "system", "content": PLANNER_INSTRUCTIONS},
                {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {message}"},
            ],
            temperature=0,
            max_tokens=2000,
            metadata=trace_metadata("plan_search", trace),
        )
    except Exception as error:  # LiteLLM raises different error types per failure
        raise RuntimeError(f"All models failed: {error}") from error

    reply = clean(response.choices[0].message.content or "")
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        plan = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        plan = None
    if not isinstance(plan, dict) or "search" not in plan:
        raise RuntimeError(f"Planner reply was not valid JSON: {reply[:200]}")
    return {"search": bool(plan["search"]), "query": str(plan.get("query") or message).strip()}


def generate_answer_stream(question: str, found: dict | None, overview: str,
                           history: list[dict], trace: dict | None = None) -> Iterator[tuple[str, str]]:
    """Step 2: write the reply, streamed.

    Yields ("model", name) once, then ("text", piece) for each piece of the answer.
    Raises RuntimeError if no model could start answering.
    """
    messages = [
        {"role": "system", "content": INSTRUCTIONS.format(overview=overview)},
        *history,  # the recent conversation, for follow-up questions
        {"role": "user", "content": f"{format_search(found)}\n\nQuestion: {question}"},
    ]
    started = False
    try:
        stream = router().completion(
            model=models()[0], messages=messages, temperature=0.2, max_tokens=2000, stream=True,
            metadata=trace_metadata("answer", trace),
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content if chunk.choices else None
            if not text:
                continue
            if not started:
                yield "model", chunk.model
                started = True
            yield "text", text
    except Exception as error:  # LiteLLM raises different error types per failure
        if started:  # broke off halfway: keep what was written
            yield "text", "\n\n*(The answer was cut off. Please ask again.)*"
            return
        raise RuntimeError(f"All models failed: {error}") from error
    if not started:
        raise RuntimeError("The model returned an empty answer")
