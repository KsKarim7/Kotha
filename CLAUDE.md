# Kotha: guide for Claude Code

Bangla-English voice AI agent and call analytics for a fictional operator (Nodi Telecom).
Spec: docs/blueprint/. The current task brief defines the scope of any work.

## Stack (ask before adding anything)
Python 3.11, FastAPI + WebSocket, Silero VAD, faster-whisper, bge-m3 + pgvector + rank_bm25,
Gemini / Ollama, MMS-TTS, PostgreSQL 16, Asterisk AudioSocket, Prometheus, MLflow,
Docker Compose, pytest.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest -q`
- Lint and format check: `uv run ruff check . && uv run ruff format --check .`
- Stack: `docker compose up -d`

## How to work
- Do only what the current task brief asks. If something outside it looks wrong, list it at the
  end of your reply; don't fix it.
- Read the files you will touch and state a short plan first. Wait for my OK when the plan
  touches more than 3 files.
- Work in small steps and run the tests after each one. Never leave tests failing.
- Every behaviour change gets a test. Prefer small real fixtures (short WAVs in tests/fixtures)
  over heavy mocks.
- Work on the feature branch. Never commit to main, never force-push, never push without my OK.
- Small commits with Conventional Commit messages, e.g. `feat(vad): add hangover-based turn detector`.
- At the end of each task, list the parts most worth studying so I can add them to docs/STUDY.md.

## Code style
- Write code that reads like a careful engineer wrote it: idiomatic Python, specific names
  (turn_audio, not data or result), small functions.
- Comments: none by default. At most a short one-line comment above a major block, only when
  the reason isn't obvious from the code. No docstrings, section banners, step-numbered comments,
  emojis, or comments that restate what the code does.
- No commented-out code; no TODO without an issue number.
- No abstractions, base classes, wrappers or config options the current task doesn't need.
- Handle errors only where something can actually fail; no blanket try/except.
- Type hints everywhere; Pydantic models at service boundaries.
- Blocking model inference never runs on the event loop: use asyncio.to_thread or a worker.
- Every network call has a timeout. Retry only idempotent calls.
- Internal audio is float32 mono 16 kHz. Convert only at the edges (WebSocket: 16 kHz PCM16 LE;
  AudioSocket: 8 kHz PCM16 LE).

## Hard rules
- Never invent metrics. Results come only from scripts in eval/; I write EVALUATION.md.
- Never commit secrets, .env files, model weights, or audio of people without consent.
- Never log transcripts, phone numbers, PINs or NID numbers. Use the allow-list logger.
- Prices, balances and account actions come only from the BSS service, never from the LLM.
- If unsure how a library behaves, check the installed version and its docs, or write a tiny
  test. Don't guess.

## Docs to keep current
README.md, docs/ARCHITECTURE.md, EVALUATION.md (results only; I write the interpretation),
docs/adr/, CHANGELOG.md.
