"""Conversational memory.

The server remembers each conversation, identified by a session id that the
chat page creates and keeps in the browser (localStorage). For every turn we
store the text and, for answers, the property cards that were shown. That lets
the chatbot:
  * understand follow-ups ("tell me more about the second one", "which of
    those is cheapest?") - see plan_search() in app/llm.py
  * restore the conversation when the page is reloaded

Memory is kept in the server's RAM: simple and enough for a demo. It is lost
when the app restarts. Old conversations are forgotten after SESSION_TTL.
"""

import time
from dataclasses import dataclass, field
from threading import Lock

MAX_TURNS = 30              # turns kept per conversation
SESSION_TTL = 6 * 60 * 60   # forget a conversation after 6 hours without messages
MAX_SESSIONS = 2000         # safety cap on memory use


@dataclass
class Conversation:
    turns: list[dict] = field(default_factory=list)  # {"role", "content", "cards"}
    last_active: float = field(default_factory=time.time)


class ConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = Lock()  # requests can arrive at the same time

    def turns(self, session_id: str) -> list[dict]:
        with self._lock:
            conversation = self._conversations.get(session_id)
            return list(conversation.turns) if conversation else []

    def add(self, session_id: str, role: str, content: str, cards: list[dict] | None = None) -> None:
        with self._lock:
            self._forget_old()
            conversation = self._conversations.setdefault(session_id, Conversation())
            conversation.turns.append({"role": role, "content": content, "cards": cards or []})
            conversation.turns = conversation.turns[-MAX_TURNS:]
            conversation.last_active = time.time()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._conversations.pop(session_id, None)

    def _forget_old(self) -> None:
        now = time.time()
        for sid in [s for s, c in self._conversations.items() if now - c.last_active > SESSION_TTL]:
            del self._conversations[sid]
        while len(self._conversations) >= MAX_SESSIONS:  # drop the least recently used
            oldest = min(self._conversations, key=lambda s: self._conversations[s].last_active)
            del self._conversations[oldest]


def for_model(turns: list[dict], last: int = 8) -> list[dict]:
    """Turn stored turns into chat messages for the AI model.

    Answers get a note listing the cards that were shown, in order, so the model
    knows what "the second one" refers to.
    """
    messages = []
    for turn in turns[-last:]:
        content = turn["content"]
        if turn["cards"]:  # put first, so it survives if a long answer gets shortened
            shown = "; ".join(f"{i}. {card['title']}" for i, card in enumerate(turn["cards"], 1))
            content = f"[Property cards shown to the user: {shown}]\n{content}"
        messages.append({"role": turn["role"], "content": content})
    return messages
