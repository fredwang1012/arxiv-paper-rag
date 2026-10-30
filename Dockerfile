# The web app with the reading list fetched and indexed at build time, so the running server never calls arXiv.
#   docker build -t arxiv-paper-rag .                                  # BM25 search only (small image)
#   docker build -t arxiv-paper-rag --build-arg EXTRAS="--extra embed" .   # hybrid search; adds PyTorch (~1 GB)
#   docker run -p 8000:8000 --env-file .env arxiv-paper-rag
# The server refuses to start without API_TOKEN or both ASK_LIMIT_PER_HOUR and ASK_LIMIT_PER_DAY, and a public
# ALLOWED_HOSTS; see "Deploying" in the README.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/.cache/huggingface PATH="/app/.venv/bin:$PATH" PORT=8000
WORKDIR /app

ARG EXTRAS=""
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev $EXTRAS

COPY app ./app
COPY readinglist.toml ./
# Fetch (about a minute, 3 s between arXiv requests) and index; with the embed extra this also downloads the embedder.
RUN python -m app.cli fetch && python -m app.cli index

RUN useradd --system --home-dir /app app && mkdir -p /app/.cache && chown -R app /app/data /app/.cache
USER app

EXPOSE 8000
CMD ["python", "-m", "app.serve"]
