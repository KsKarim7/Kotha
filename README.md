# Kotha

A Bangla-English voice AI agent and call-analytics platform for Nodi Telecom, a fictional telecom operator.

[![ci](https://github.com/KsKarim7/Kotha/actions/workflows/ci.yml/badge.svg)](https://github.com/KsKarim7/Kotha/actions/workflows/ci.yml)

## Status

Early development. Nothing runs end to end yet.

## What it will do

These are planned features; none of them is built yet.

- A real-time voice agent that callers reach from a browser (WebSocket) or a phone line (SIP through Asterisk AudioSocket).
- Barge-in, so callers can interrupt the agent, and handoff to a human agent with a summary.
- Post-call analytics: transcription, PII redaction, intent, sentiment, a summary and a QA score in which every criterion quotes the transcript.
- An evaluation harness that measures speech recognition, intent, retrieval, latency and task completion on fixed test sets.
- An on-prem mode in which no call data leaves the machine.

## Fictional data

Nodi Telecom and all telecom data in this repository are fictional. No real operator's name, tariffs or logo are used.

## Development quickstart

Prerequisite: [uv](https://docs.astral.sh/uv/). uv installs Python 3.11 for you.

```bash
uv sync
uv run pre-commit install
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

## Results

There are no results yet. Every number will come from a committed evaluation run and will be reported in [EVALUATION.md](EVALUATION.md).

## Documentation

- [Project specification](docs/blueprint/05_blueprint_spec.md)
- [Evaluation report](EVALUATION.md)
- [Changelog](CHANGELOG.md)
- [Decision records](docs/adr/)

## Licence

The code is released under the MIT licence; see [LICENSE](LICENSE).

Third-party models keep their own licences, and Kotha does not ship model weights. The planned Bangla text-to-speech model, `facebook/mms-tts-ben`, is licensed CC-BY-NC 4.0, which allows non-commercial use only.
