# Module 7 - Container for the chatbot.
#
# The image contains only what the live app needs: the app/ code, the
# ready-made SQLite database, the Chroma vector index and the small embedding
# model Chroma uses to understand questions. Scraping and cleaning
# (Modules 1-2) run on a laptop before deployment, so Playwright/Chrome are
# not installed here.
#
# Build and run locally:
#   docker build -t property-chatbot .
#   docker run -p 7860:7860 --env-file .env property-chatbot
#   open http://localhost:7860

FROM python:3.13-slim

RUN pip install --no-cache-dir uv==0.11.13

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=7860

# Hugging Face Spaces runs containers as user id 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# Install dependencies first, so this layer is cached when only code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY data/properties.db ./data/properties.db
# Chroma writes small lock/log files next to its index, so the app user must own it.
COPY --chown=user data/chroma ./data/chroma
# Audit records (app/audit.py) are appended here.
RUN mkdir -p data/audit && chown user data/audit

USER user
ENV PATH="/home/user/app/.venv/bin:$PATH"

# Download the embedding model (all-MiniLM-L6-v2, ~80 MB) now, not on the first question.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])"

EXPOSE 7860

# PORT is 7860 by default; hosts such as Render set their own.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
