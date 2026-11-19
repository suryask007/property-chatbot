"""Module 6 - Web API and chat page.

Endpoints
  GET    /             the chat page (app/static/index.html)
  POST   /api/chat     {"session_id", "message"} -> streamed answer + property cards
  GET    /api/history  ?session_id=... -> the remembered conversation (to restore the page)
  POST   /api/reset    {"session_id"} -> forget the conversation ("New chat")
  GET    /api/traces   recent audit records (app/audit.py); needs "Authorization: Bearer <ADMIN_TOKEN>"
  GET    /api/traces/{request_id}  one audit record
  GET    /health       used by the hosting platform to check the app is running

Flow of one chat message:
  1. load the conversation so far             (app/memory.py)
  2. plan: search needed? standalone query?   (app/llm.py, plan_search)
  3. hybrid search: filters + keywords + Chroma (app/search.py)
  4. stream the answer as it is written       (app/llm.py, generate_answer_stream)
  5. remember the message, answer and cards   (app/memory.py)
  6. write an audit record of every step      (app/audit.py; AI calls also go to Langfuse)

Run locally:
    uv run uvicorn app.main:app --reload
"""

import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()  # reads OPENROUTER_API_KEY from .env when running locally

from app import audit  # noqa: E402
from app.llm import generate_answer_stream, plan_search  # noqa: E402  (needs the key loaded first)
from app.memory import ConversationStore, for_model  # noqa: E402
from app.search import dataset_overview, get_db, get_vectors, keywords, parse_filters, search  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"
MAX_CARDS = 6
SESSION_ID = r"^[A-Za-z0-9-]{8,64}$"
log = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Open the database and the Chroma index at startup, so the first question is fast."""
    get_db()
    get_vectors().query(query_texts=["warm up"], n_results=1)
    yield


app = FastAPI(title="DarGlobal & Wasalt Property Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
memory = ConversationStore()


# ------------------------------------------------------------ request shapes

class ChatRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID)
    message: str = Field(min_length=1, max_length=500)


class ResetRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID)


# ------------------------------------------------- simple per-visitor limit
# The free OpenRouter tier has a daily request quota, so one visitor
# shouldn't be able to use it all up.

RATE_LIMIT = 15          # messages...
RATE_WINDOW = 10 * 60    # ...per 10 minutes, per IP address
recent_requests: dict[str, deque] = defaultdict(deque)


def check_rate_limit(ip: str) -> None:
    now = time.time()
    times = recent_requests[ip]
    while times and now - times[0] > RATE_WINDOW:
        times.popleft()
    if len(times) >= RATE_LIMIT:
        raise HTTPException(429, "Too many messages. Please wait a few minutes and try again.")
    times.append(now)


def client_ip(request: Request) -> str:
    # Behind the hosting platform's proxy the real address is in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")


# ------------------------------------------------------------------ helpers

def to_card(p: dict) -> dict:
    """The few fields the chat page needs to draw a property card."""
    return {
        "title": p["title"],
        "source": p["source"],
        "kind": p["kind"],
        "purpose": p["purpose"],
        "property_type": p["property_type"],
        "location": p["location_text"] or ", ".join(filter(None, [p["district"], p["city"]])),
        "price": p["price"],
        "currency": p["currency"],
        "price_period": p["price_period"],
        "bedrooms": p["bedrooms"],
        "area": f"{p['area_sqm']:,.0f} sqm" if p["area_sqm"] else p["area_text"],
        "status": p["status"],
        "url": p["url"],
        "image_url": p["image_url"],
    }


SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|salam|salaam|marhaba|thanks|thank you|ok|okay|bye|good (morning|evening))\b[\s!.?]*$", re.I)


def plan_with_rules(message: str, turns: list[dict]) -> dict:
    """Backup for plan_search() when no AI model is available: simple rules.

    No search for small talk; a message that seems to refer back ("the second
    one", "those") is searched together with the previous question.
    """
    if SMALL_TALK.match(message):
        return {"search": False, "query": ""}
    previous = [t["content"] for t in turns if t["role"] == "user"]
    refers_back = re.search(r"\b(first|second|third|last|it|this|that|these|those|them|they|one|ones|before|previous)\b", message, re.I)
    adjusts = re.match(r"\s*(what about|how about|and|in|only|but|cheaper|bigger|under|over)\b", message, re.I)
    # Combine with the previous question; search.py lets later words override earlier
    # ones, so "what about Jeddah?" after a Riyadh question searches Jeddah.
    query = f"{previous[-1]} {message}" if previous and (refers_back or adjusts) else message

    # A general question like "tell me about Wasalt" names only a source: nothing to look up.
    filters = parse_filters(query)
    if not keywords(query, filters) and set(filters.active()) <= {"source", "kind"}:
        return {"search": False, "query": ""}
    return {"search": True, "query": query}


def fallback_answer(found: dict | None) -> str:
    """Used when no AI model is available: still show what was found."""
    if not found:
        return ("The AI assistant has reached its free daily limit, so here is a summary of what I "
                f"know:\n\n{dataset_overview()}\n\nAsk about a city, budget or project to see matching properties.")
    if not found["results"]:
        return "The AI assistant is busy right now and I couldn't find matching properties. Try different words."
    return (f"The AI assistant is busy right now, but I found {found['total_matches']} matching "
            "properties. Here are the closest ones; open a card for full details.")


# ---------------------------------------------------------------- endpoints

@app.get("/")
def chat_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    count = get_db().execute("SELECT COUNT(*) FROM properties").fetchone()[0]
    return {"status": "ok", "properties": count, "embeddings": get_vectors().count()}


@app.get("/api/history")
def history(session_id: str):
    if not re.match(SESSION_ID, session_id):
        raise HTTPException(400, "Invalid session id")
    return {"turns": memory.turns(session_id)}


@app.post("/api/reset")
def reset(body: ResetRequest):
    memory.clear(body.session_id)
    return {"ok": True}


@app.get("/api/traces")
def traces(request: Request, limit: int = 50, session_id: str | None = None):
    """Recent audit records, newest first. Requires the ADMIN_TOKEN."""
    check_admin(request)
    records = [r for r in reversed(audit.recent) if not session_id or r["session_id"] == session_id]
    return {"traces": records[:limit]}


@app.get("/api/traces/{request_id}")
def trace_detail(request_id: str, request: Request):
    check_admin(request)
    record = audit.find(request_id)
    if not record:
        raise HTTPException(404, "No trace with that id (only recent traces are kept in memory)")
    return record


def check_admin(request: Request) -> None:
    """Allow only requests with header "Authorization: Bearer <ADMIN_TOKEN>"."""
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        raise HTTPException(404, "Traces are disabled: set ADMIN_TOKEN to enable them")
    if not secrets.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
        raise HTTPException(401, "Missing or wrong admin token")


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request):
    check_rate_limit(client_ip(request))
    trace = audit.Trace(body.session_id, body.message)
    labels = {"request_id": trace.request_id, "session_id": body.session_id}  # for Langfuse
    turns = memory.turns(body.session_id)
    history = for_model(turns)

    # Step 1: does this message need a search, and for what?
    # The first message has no history to resolve, so simple rules are enough and
    # we save one AI call (the free tier has a daily request limit).
    started = trace.elapsed_ms()
    planned_by = "rules"
    if not turns:
        plan = plan_with_rules(body.message, turns)
    else:
        try:
            plan = plan_search(body.message, history, labels)
            planned_by = "ai"
        except RuntimeError as error:
            log.warning(f"planner unavailable, using rules: {error}")
            trace.error("plan", str(error))
            plan = plan_with_rules(body.message, turns)
    trace.timing("plan", started)
    trace.record["plan"] = {"by": planned_by, **plan}

    # Step 2: hybrid search (only when needed).
    started = trace.elapsed_ms()
    found = search(plan["query"]) if plan["search"] else None
    trace.timing("search", started)
    cards = [to_card(p) for p in found["results"][:MAX_CARDS]] if found else []
    if found:
        trace.record["search"] = {
            "filters": found["filters"],
            "keywords": found["keywords"],
            "total_matches": found["total_matches"],
            "relaxed": found["relaxed"],
            "result_ids": [p["id"] for p in found["results"]],
        }

    # Step 3: stream the answer. The response is one JSON object per line:
    #   {"type": "meta", ...}    search query, number of matches, property cards, request id
    #   {"type": "text", ...}    a piece of the answer (many of these)
    #   {"type": "done", ...}    which model answered
    def events():
        yield as_line({
            "type": "meta",
            "request_id": trace.request_id,
            "search_query": plan["query"] if plan["search"] else None,
            "total_matches": found["total_matches"] if found else 0,
            "properties": cards,
        })
        pieces, model, fallback = [], None, False
        try:
            resolved = plan["query"] if plan["search"] else None
            for kind, value in generate_answer_stream(body.message, found, dataset_overview(), history, labels, resolved):
                if kind == "model":
                    model = value
                    trace.mark("first_word")
                else:
                    pieces.append(value)
                    yield as_line({"type": "text", "text": value})
        except RuntimeError as error:
            log.warning(str(error))
            trace.error("answer", str(error))
            pieces, fallback = [fallback_answer(found)], True
            yield as_line({"type": "text", "text": pieces[0]})

        # Step 4: remember the exchange and write the audit record.
        answer = re.sub(r"<think>.*?</think>", "", "".join(pieces), flags=re.S).strip()
        memory.add(body.session_id, "user", body.message)
        memory.add(body.session_id, "assistant", answer, cards)
        trace.record["answer"] = {"model": model, "fallback": fallback, "text": answer}
        trace.finish()
        yield as_line({"type": "done", "model": model})

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "X-Request-ID": trace.request_id,
            # Ask proxies (e.g. Hugging Face's) to pass each piece on immediately.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def as_line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"
