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
- A script in `eval/` that plots 16 kHz audio against simulated telephony, and the resulting figure in the README.
- A Silero VAD wrapper in `kotha.vad.silero` that returns one speech probability per 32 ms window, with torch installed from the PyTorch CPU index.
- A turn detector in `kotha.vad.turns` that turns 32 ms speech probabilities into speech-started and speech-ended events, with hysteresis, minimum speech, hangover and pre-roll.
- `TurnDetector.flush()` to close a turn still open at the end of a stream, a script in `eval/` that prints the turns Silero finds in an audio file, and VAD notes in `docs/ARCHITECTURE.md`.
- A faster-whisper ASR wrapper in `kotha.asr.whisper` that turns 16 kHz audio into a transcript with language, confidence signals and word timings, running on CPU.
- Local GPU support for faster-whisper: cuBLAS for CUDA 12 in a `gpu` dependency group that CI skips, a setup runbook, ADR 0004, and a script in `eval/` that checks which compute types work on a device.
- A Whisper decoding-stability probe in `eval/` that compares decoding settings (temperature, beam size, repetition controls, a token cap) on the same clips, and an idempotent `LD_LIBRARY_PATH` block in the GPU runbook.
- A text normaliser in `kotha.text.normalize` that WER applies to both references and hypotheses: format characters removed, Unicode NFC, Bangla digits to ASCII, lowercase, punctuation to spaces, whitespace collapsed.
- Conversion steps in the runbook for two Bangla Whisper fine-tunes (whisper-medium and whisper-small based), pinned to a commit and converted to CTranslate2 locally, and a `bengali_share` column in the decoding probe that shows how much of the output is in Bengali script.
- No-timestamps configs (`t0_nots`, `beam5_nots`) in the decoding probe, a `--warmup` option that runs one untimed transcription before timing starts, and columns for clip length, word count, segment count and the first and last word times.
- A WER and CER scoring core in `eval/wer.py`: manifest reading, split selection, and corpus-level WER and CER with latency p50 and p95 per condition and language, on references and hypotheses that both go through the text normaliser; jiwer is a dev dependency.

### Changed

- `uv sync` now installs `kotha` as an editable package, so scripts can import it without `PYTHONPATH`.
- `WhisperAsr` decodes at a single temperature of 0.0 without timestamps, with word timings off unless asked for, and `Transcript` reports the highest compression ratio and the most tokens in one segment (ADR 0005, proposed). The segment-joining rule is now `join_segment_texts`, which the decoding probe imports; probe configs can override its fixed options, and it has two configs without word timings.
