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
- The WER evaluation command, `eval/wer.py`: it transcribes a manifest split with `WhisperAsr` at 16 kHz and through simulated telephony, after one untimed warm-up, prints the run settings and a WER, CER and latency table, and writes a per-utterance CSV that must go outside the repository. The fixed decoding options are one constant, `DECODING_OPTIONS`, in `kotha.asr.whisper`, and the runbook has a "WER evaluation" section.
- A loanword-agnostic score next to the strict one: `data/testsets/loanwords.csv` lists accepted Bengali-script spellings of English loanwords, and `eval/wer.py` reports `lexicon_wer` and `lexicon_cer` after mapping those spellings to Latin on both sides (`--lexicon`, the header, the table and the per-utterance CSV). `wer` and `cer` stay the primary scores and are unchanged.
- A GPU micro-benchmark, `eval/gpu_bench.py`, that times feature extraction, the encoder and a whole transcription per compute type (wall and CPU time, decode time per token) while sampling the GPU's P-state, SM and memory clocks, power, temperature and clock event reasons, and a "GPU benchmark" subsection in the runbook.
- A transcription guideline for the ASR test set in `data/testsets/GUIDELINE.md`, ADR 0002 on consent, storage, retention and publication of test-set recordings, and `card` in the loanword lexicon.
- A manifest validator, `eval/validate_manifest.py`, that checks one or more manifests and their audio (allowed values, speaker codes, unique ids and paths, paths inside `--data-dir`, files that exist and that soundfile can read, duration bounds), warns about container mismatches, low sample rates, extra channels and mixed codecs or sample rates, prints a summary of the set without any reference text, and exits 1 on errors; a "Checking a manifest" section in the runbook.
- A reference-text lint, `eval/lint_references.py`, that checks manifest references against the transcription guideline (empty, not already normalised with the first differing code point and its kind, the taka sign, `lang` against the letters used, loanwords in Bengali script), warns about joined numbers and units, Bangla suffixes attached to Latin words and filler words, never prints reference text, and exits 1 on errors; a "Checking references" section in the runbook.
- Error analysis for WER runs, `eval/error_analysis.py`: `eval/wer.py` now writes the 20 most frequent substituted word pairs per condition, each with a category (`loanword-script`, `number`, `script`, `other`), to `<output stem>.substitutions.csv` next to `--output`, and prints only the category counts per condition; a "Top substitutions" subsection in the runbook.
- Local experiment tracking for WER runs: `eval/wer.py --mlflow-dir DIR` logs one MLflow run to a SQLite store under `DIR` (outside the repository) with the run settings, the scores per condition and language, the git commit and `git_dirty`, and both CSVs as artifacts, with MLflow's telemetry switched off; `mlflow-skinny`, `sqlalchemy` and `alembic` join the dev group, and the runbook has a "Tracking runs with MLflow" subsection.
- ADR 0006 (proposed) for the knowledge base and hybrid retrieval: Markdown KB files per language in `data/kb/`, PostgreSQL + pgvector, bge-m3 dense vectors through sentence-transformers at a pinned revision and loaded offline, our own BM25Okapi, the loanword lexicon as query expansion, RRF fusion, and the retrieval evaluation plan.
- The first eight knowledge-base documents in `data/kb/`, in Bangla and English (internet, voice and combo packages; buying, activating and stopping packs; ways to recharge; recharge problems; checking the balance; validity), all fictional, with invented codes in one "How to reach us" section; `kotha.kb.documents` reads a KB file and refuses one that breaks the format, and `tests/test_kb_content.py` checks the whole folder (translations, sections, script, no money, no real brands, known package codes, short codes and phone numbers).
- The remaining eight knowledge-base documents in `data/kb/`, in Bangla and English (replacing a SIM, a lost or stolen SIM, account PIN and verification, unused data and rollover, international roaming, complaints, refunds, service hours), all fictional and consistent with the first eight; `tests/test_kb_content.py` also checks that ticket numbers are only the allowed examples.
- A knowledge-base chunker, `kotha.kb.chunks`: each `## ` section becomes one or more chunks of at most 300 words (unless a single table or paragraph is longer), split only at blank lines into balanced pieces so a Markdown table stays whole, with deterministic ids `<doc_id>.<lang>.<index>` and the document's title, language and heading on every chunk.
- A local PostgreSQL 16 + pgvector database for the knowledge base: `docker-compose.yml` with the pinned `pgvector/pgvector:0.8.7-pg16` image (credentials from `.env`, port bound to 127.0.0.1), `.env.example`, `kotha.db.schema` with the `kb_chunks` table (1024-dimensional embeddings, exact cosine search), the same image as a CI service container, and database tests that skip without `DATABASE_URL` outside CI (in CI a missing URL fails them); `psycopg[binary]` and `pgvector` join the runtime dependencies.
- Knowledge-base ingestion: `kotha.rag.embedder.Embedder` loads `BAAI/bge-m3` at a pinned revision through sentence-transformers, offline and on the CPU by default, and `python -m kotha.kb.ingest` chunks, embeds and stores the KB in `kb_chunks` in one idempotent transaction; sentence-transformers joins the runtime dependencies.

### Changed

- The loanword lexicon moved from `data/testsets/loanwords.csv` to `data/lexicon/loanwords.csv`, because the running service reads it for query expansion (ADR 0006); the CTranslate2 conversion now uses the project's own transformers, checked to give byte-identical models.
- `uv sync` now installs `kotha` as an editable package, so scripts can import it without `PYTHONPATH`.
- `WhisperAsr` decodes at a single temperature of 0.0 without timestamps, with word timings off unless asked for, and `Transcript` reports the highest compression ratio and the most tokens in one segment (ADR 0005, proposed). The segment-joining rule is now `join_segment_texts`, which the decoding probe imports; probe configs can override its fixed options, and it has two configs without word timings.
- The runbook records the Windows and NVIDIA power settings the local GPU needs, a check with `eval/gpu_bench.py` to run before timing anything on cuda, where benchmark output is kept, and that cuda latency in a WER run counts only after that check.
