# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project skeleton and tooling: a uv project pinned to Python 3.11, ruff for lint and format, pytest, an MIT licence and the agent guide in `CLAUDE.md`.
- Pre-commit hooks that run ruff and catch trailing whitespace, missing final newlines, broken YAML or TOML, merge conflict markers, private keys and files over 500 KB.
- CI on every pull request and every push to `main`, running pre-commit and the test suite; `main` now requires a pull request and a passing `checks` job before merging.
- A pull request template with the review checklist, and a template for architecture decision records.
- A study log in `docs/STUDY.md` for questions about the code.
- This changelog.
- A README describing the project and its current state, and a skeleton evaluation report.
- Sprint notes in `docs/sprints/`.
- Audio helpers in `kotha.audio.convert`: exact conversion between float32 and little-endian PCM16, resampling, and loading any WAV or FLAC as float32 mono 16 kHz.
- A mu-law telephony simulator in `kotha.audio.telephony`, notes on the audio formats in `docs/ARCHITECTURE.md`, and ruff now sorts `kotha` imports as first-party.
