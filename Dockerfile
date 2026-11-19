FROM python:3.11.18-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# dependencies first, so editing the code does not reinstall torch
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group gpu --no-install-project

COPY services ./services
COPY data/kb ./data/kb
COPY data/lexicon ./data/lexicon
RUN uv sync --frozen --no-dev --no-group gpu

EXPOSE 8000
CMD ["uvicorn", "kotha.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "65536"]
