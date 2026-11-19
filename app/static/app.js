// Chat page behaviour: send questions to /api/chat and show answers + property cards.
// The server remembers the conversation (app/memory.py); the page only keeps a
// session id in localStorage so the server knows which conversation is ours.

const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("message");
const sendButton = document.getElementById("send");
const welcome = document.getElementById("welcome");
const cardTemplate = document.getElementById("card-template");

function newSessionId() {
  const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  localStorage.setItem("sessionId", id);
  return id;
}
let sessionId = localStorage.getItem("sessionId") || newSessionId();

// ------------------------------------------------------------ markdown
// A tiny markdown converter for the AI's answers. Everything is HTML-escaped
// first, so the answer can never inject its own HTML or scripts.

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}

function markdownToHtml(markdown) {
  const html = [];
  let list = null; // "ul" or "ol" while inside a list
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${paragraph.map(inlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (list) html.push(`</${list}>`);
    list = null;
  };

  for (const rawLine of markdown.split("\n")) {
    const line = rawLine.trim();
    const bullet = line.match(/^[-*•]\s+(.*)/);
    const numbered = line.match(/^\d+[.)]\s+(.*)/);
    const heading = line.match(/^#{1,6}\s+(.*)/);

    if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (list !== type) { closeList(); html.push(`<${type}>`); list = type; }
      html.push(`<li>${inlineMarkdown((bullet || numbered)[1])}</li>`);
    } else if (heading) {
      flushParagraph(); closeList();
      html.push(`<h4>${inlineMarkdown(heading[1])}</h4>`);
    } else if (line === "") {
      flushParagraph(); closeList();
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  flushParagraph(); closeList();
  return html.join("");
}

// --------------------------------------------------------------- cards

function formatPrice(p) {
  if (!p.price) return null;
  const amount = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(p.price);
  const period = p.price_period ? ` / ${p.price_period.replace(/ly$/, "")}` : "";
  return `${amount} ${p.currency || ""}${period}`;
}

function buildCard(p) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);
  card.href = p.url;

  const imageBox = card.querySelector(".card-image");
  const img = card.querySelector("img");
  if (p.image_url) {
    img.src = p.image_url;
    img.onerror = () => { img.remove(); imageBox.classList.add("no-photo"); };
  } else {
    img.remove();
    imageBox.classList.add("no-photo");
  }

  const source = card.querySelector(".card-source");
  source.textContent = p.source === "DarGlobal" ? "DarGlobal project" : `Wasalt · for ${p.purpose}`;
  if (p.source === "DarGlobal") source.classList.add("darglobal");

  card.querySelector(".card-title").textContent = p.title;
  card.querySelector(".card-location").textContent = p.location || "";

  const price = card.querySelector(".card-price");
  const priceText = formatPrice(p);
  price.textContent = priceText || "Price not published";
  if (!priceText) price.classList.add("unpublished");

  const facts = [
    p.bedrooms ? `${p.bedrooms} bed` : null,
    p.area,
    p.kind === "project" ? p.status : p.property_type,
  ].filter(Boolean);
  card.querySelector(".card-facts").textContent = facts.join(" · ");
  return card;
}

// ------------------------------------------------------------ messages

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight });
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "message user";
  el.textContent = text;
  chat.appendChild(el);
}

function addAssistantMessage() {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = '<div class="bubble"><span class="typing" aria-label="Thinking"><span></span><span></span><span></span></span></div>';
  chat.appendChild(el);
  return el;
}

function showText(el, markdown) {
  el.querySelector(".bubble").innerHTML = markdownToHtml(markdown);
}

function showCards(el, properties) {
  if (!properties || !properties.length) return;
  const cards = document.createElement("div");
  cards.className = "cards";
  properties.forEach((p) => cards.appendChild(buildCard(p)));
  el.appendChild(cards);
}

function showDetails(el, { search_query, total_matches, model, request_id }) {
  const details = [];
  if (search_query) details.push(`Searched for “${search_query}”: ${total_matches} matches`);
  if (model) details.push(`answered by ${model}`);
  if (request_id) details.push(`ref ${request_id}`); // matches the audit record on the server
  if (!details.length) return;
  const meta = document.createElement("p");
  meta.className = "meta";
  meta.textContent = details.join(" · ");
  el.appendChild(meta);
}

// Redraw a remembered conversation after the page is reloaded.
async function restoreConversation() {
  try {
    const response = await fetch(`/api/history?session_id=${encodeURIComponent(sessionId)}`);
    const { turns } = await response.json();
    if (!turns || !turns.length) return;
    welcome.remove();
    for (const turn of turns) {
      if (turn.role === "user") {
        addUserMessage(turn.content);
      } else {
        const el = addAssistantMessage();
        showText(el, turn.content);
        showCards(el, turn.cards);
      }
    }
    scrollToBottom();
  } catch {
    // If the history can't be loaded, just start with an empty chat.
  }
}

function showError(el, message) {
  el.classList.add("error");
  el.querySelector(".bubble").textContent = message;
}

// The server streams one JSON object per line (see app/main.py):
//   {"type": "meta"} cards + search details, {"type": "text"} answer pieces, {"type": "done"}
async function readAnswerStream(response, el) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let meta = {};

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // the last line may be incomplete; keep it for the next read

    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "meta") {
        meta = event;
        showCards(el, event.properties); // cards appear before the text is written
      } else if (event.type === "text") {
        answer += event.text;
        showText(el, answer);
        scrollToBottom();
      } else if (event.type === "done") {
        showDetails(el, { ...meta, model: event.model });
      }
    }
  }
}

async function ask(question) {
  welcome?.remove();
  addUserMessage(question);
  const reply = addAssistantMessage();
  scrollToBottom();
  sendButton.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: question }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong. Please try again.");
    }
    await readAnswerStream(response, reply);
  } catch (error) {
    showError(reply, typeof error.message === "string" ? error.message : "Could not reach the server. Check your connection and try again.");
  } finally {
    sendButton.disabled = false;
    input.focus();
    scrollToBottom();
  }
}

// -------------------------------------------------------------- events

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendButton.disabled) return;
  input.value = "";
  input.style.height = "";
  ask(question);
});

// Enter sends, Shift+Enter adds a new line.
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

// Grow the text box as the user types.
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
});

document.querySelectorAll("#starters button").forEach((button) => {
  button.addEventListener("click", () => ask(button.textContent));
});

// "New chat": the server forgets this conversation and the page starts fresh.
document.getElementById("new-chat").addEventListener("click", () => {
  const oldSessionId = sessionId;
  // Clear the page first, so a question typed right away isn't wiped afterwards.
  sessionId = newSessionId();
  chat.replaceChildren(welcome);
  input.focus();
  fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: oldSessionId }),
  }).catch(() => {});
});

restoreConversation();
