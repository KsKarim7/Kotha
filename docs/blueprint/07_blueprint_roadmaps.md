# Kotha: Project Blueprint for AI Engineer (Voice)

## Tab 4 · Phases 7–8: Learning and Development Roadmaps

## Phase 7a: Prerequisites (learn before coding)

Six skills, about 3–5 focused days, mostly overlapping with Milestone 1. Each ends in a small exercise that becomes real project code.

| Skill | Why you need it | Concepts | Exercise | Applied in |
| --- | --- | --- | --- | --- |
| Digital audio basics | Every bug in a voice system starts as a format bug | Sample rate, bit depth, PCM16, little-endian, Nyquist, μ-law, spectrograms | Record yourself; plot waveform and spectrogram at 16 kHz and after simulated 8 kHz; convert WAV → PCM bytes → WAV losslessly | M1, M9 |
| Async Python for streams | A call is a continuous stream with concurrent listening and speaking | Tasks, `TaskGroup`, queues, cancellation, `to_thread`, backpressure | An echo server that streams a WAV back in 20 ms frames and stops instantly on a "stop" message | M8, M9, M12 |
| FastAPI and WebSockets | The JD's API requirement; the browser call path | Routing, dependencies, WebSocket lifecycle, JSON vs binary frames | Build the echo server in FastAPI plus a browser page that records and plays back | M5, M8 |
| ASR and WER basics | The core metric of the job | Whisper at block-diagram level; WER/CER by hand | Transcribe 10 of your own clips with faster-whisper; compute WER for 3 by hand, then with jiwer | M3 |
| Embeddings and vector search | RAG grounding | Cosine similarity, HNSW, BM25 intuition | pgvector in Docker; insert 20 bn + en sentences; query across languages | M4 |
| Docker and Compose | Reproducible stack | Images, volumes, networks, env files, healthchecks | Compose up PostgreSQL + pgvector and a FastAPI hello service | All |

**Resources:** Hugging Face Audio Course (huggingface.co/learn/audio-course, free), FastAPI WebSockets docs (fastapi.tiangolo.com/advanced/websockets), Python asyncio tasks docs (docs.python.org/3/library/asyncio-task.html), pgvector (github.com/pgvector/pgvector), faster-whisper (github.com/SYSTRAN/faster-whisper).

## Phase 7b: Learn While Building

Learn these at the module that needs them; each exercise doubles as a measurable result for `EVALUATION.md`.

| Skill | Why | Concepts | Exercise | Module |
| --- | --- | --- | --- | --- |
| VAD and turn-taking | Cutting callers off or answering late are the two most noticeable voice-bot failures | Hysteresis, hangover, pre-roll, minimum speech | Hangover ablation at 500/700/900 ms | M2 |
| Bangla normalisation and code-mixing | The employer's stated differentiator | Unicode NFC, digit mapping, transliteration, loanword variants | A 50-term telecom loanword lexicon in both scripts; WER with and without normalisation | M3, M4, M7 |
| LLM function calling and dialogue state | Multi-turn agents with safe actions | Tool schemas, state machines, slot filling, server-side guards | Text-mode scenario runner for UC1–UC5 | M6 |
| Intent classifier fine-tuning | "Intent accuracy" and call routing | Tokenisation, encoder fine-tuning, macro-F1, data leakage | xlm-roberta-base vs LLM zero-shot on the same test set | M6 |
| TTS and speech-text normalisation | Natural Bangla output | VITS, real-time factor, number verbalisation | `num_to_bn_words` with unit tests; listening sheet | M7 |
| Telephony | SIP/VoIP integration | SIP call flow, RTP, codecs, dialplan, AudioSocket framing | Two softphones calling each other through Asterisk *before* adding AudioSocket | M9 |
| Diarization and analytics evaluation | Speaker separation, QA scoring | Diarization error rate, Cohen's kappa | DER on mono mixdowns of your stereo calls | M10 |
| Privacy engineering | Consent, PII, residency | PII patterns, data minimisation, retention, egress control | Planted-PII scan returns zero hits | M11 |
| Observability and load testing | Scale and production monitoring | Metric types, percentiles, Little's law, event-loop lag | Concurrency ramp test with a bottleneck write-up | M12 |
| ASR fine-tuning with LoRA | "Develop, fine-tune" ASR | PEFT/LoRA, augmentation, mixed precision, checkpoint conversion | A 500-step smoke run on a small subset before the full run | M13 |
| Experiment tracking | JD-preferred MLflow; reproducibility | Runs, params, metrics, artifacts | Log every M3 and M13 run | M3, M13 |
| Agile practice | JD Agile/Scrum | Sprint goal, backlog, definition of done, review | Weekly sprint notes in `docs/sprints/` | M0 |

### Needs separate study (the project won't teach it)

- **Telecom domain vocabulary:** prepaid vs postpaid, USSD, OSS vs BSS, call detail records (CDRs), number portability, IVR menu design. Read operator FAQ pages and a telecom-OSS/BSS primer; write a one-page glossary.
- **Commercial contact-centre and CTI platforms:** what Genesys, Avaya or Cisco contact-centre suites do and where a voice bot plugs in. Conceptual only.
- **Regulation:** read the official text of Bangladesh's Personal Data Protection Ordinance and any BTRC guidance relevant to call recording, rather than relying on summaries.

## Phase 8: Development Milestones

A working browser call ships in week 1; each later milestone deepens one layer (about 7 weeks full-time):

| Milestone | Weeks | Focus |
| --- | --- | --- |
| MS1 · Browser call works | W1 | VAD, ASR, RAG, LLM, TTS end to end (interview-ready MVP) |
| MS2 · Conversational agent | W2 | tools, verification, handoff, barge-in |
| MS3 · Phone calls + 8 kHz | W3 | Asterisk AudioSocket, robustness tests |
| MS4 · Analytics + privacy | W4–W5 | diarize, redact, QA scores, consent |
| MS5 · Production hardening | W6 | metrics, load test, CI gates, on-prem |
| MS6 · Model work + stretch | W7 | Whisper LoRA, one stretch feature |

Each milestone depends on the previous one's definition of done. The one exception is M13 training, which can run on Kaggle in the background from week 3. Part-time, stretch each week to about 1.5–2 weeks. If you fall behind, cut features, never the evaluation.

### MS1 · Browser call works (week 1) — depends on: prerequisites

- **Tasks and features:** M0 repo and CI; M1 audio utilities, a 150-utterance test set and the transcription guideline; M2 VAD; M3 ASR + evaluation CLI (2 models × 2 audio conditions); M4 a 15-document KB + hybrid retriever; M6 answer-only (no tools yet); M7 Bangla TTS; M8 WebSocket loop + browser client.
- **Concepts:** audio formats, VAD, WER/CER, embeddings and RRF, asyncio.
- **Deliverables:** demo video #1 (a 60-second code-mixed browser call); `EVALUATION.md` v1 with the WER, retrieval and latency tables.
- **Definition of done:** a fresh clone runs with `docker compose up`; a spoken code-mixed policy question (UC4) gets a correct, cited answer; every number in the README comes from a committed evaluation run. This is the same scope as the 1-week plan in your CV document.

### MS2 · Conversational agent (week 2) — depends on: MS1

- **Tasks and features:** M5 mock BSS/CRM; M6 full (verification, confirmation, handoff summary, intent-classifier comparison); M8 barge-in and sentence-streamed TTS.
- **Concepts:** function calling, state machines, idempotency, task cancellation.
- **Deliverables:** scenario runner passing UC1–UC5 in text and audio; prompt-injection suite; TTFA before and after sentence streaming; barge-in demo clip.
- **Definition of done:** tests prove no write action can happen without verification and a spoken yes; handoff summaries validate against the Pydantic schema.

### MS3 · Phone calls and 8 kHz robustness (week 3) — depends on: MS2

- **Tasks and features:** Asterisk container and softphones; AudioSocket server and transport refactor; DTMF fallback; handoff via the dialplan; 30 utterances recorded through the real phone path; 8 kHz experiments (model choice, domain prompt).
- **Concepts:** SIP and RTP, codecs, dialplan, real-time frame pacing.
- **Deliverables:** demo video #2 (a softphone call that hands off to the agent softphone); a simulated-vs-real 8 kHz WER table.
- **Definition of done:** UC1 and UC5 pass over SIP; phone-path latency is measured separately from the browser path.

### MS4 · Post-call analytics and privacy (weeks 4–5) — depends on: MS3

- **Tasks and features:** record 10–15 stereo role-play calls with consenting friends; M10 analytics worker; M11 consent, redaction, retention, RBAC dashboard.
- **Concepts:** diarization and DER, structured output, Cohen's kappa, PII engineering.
- **Deliverables:** a dashboard with per-call analytics and redacted audio; evaluation rows for DER, QA agreement, quote verification and PII recall.
- **Definition of done:** a planted-PII scan of logs and database returns zero hits; every QA score shows a verified quote or is marked "na".

### MS5 · Production hardening (week 6) — depends on: MS1–MS4

- **Tasks and features:** Prometheus and Grafana; JSON logs; load generator and concurrency ramp; one measured optimisation; CI evaluation gates; on-prem mode with Ollama and a local-vs-cloud LLM comparison.
- **Concepts:** metric types, percentiles, bottleneck analysis, egress control.
- **Deliverables:** concurrency chart; Grafana screenshot; demo video #3 (a full call with the network switched off).
- **Definition of done:** CI blocks a deliberately regressing pull request; the on-prem egress check passes.

### MS6 · Model work and stretch (week 7) — depends on: the MS1 evaluation harness

- **Tasks and features:** M13 LoRA fine-tuning and comparison; one M14 feature (voice biometrics recommended); final README, `ARCHITECTURE.md` and a 3-minute walkthrough video.
- **Concepts:** PEFT/LoRA, augmentation, equal error rate.
- **Deliverables:** published MLflow comparison; the final evaluation report.
- **Definition of done:** a reviewer understands the system, its numbers and its limits after five minutes with the README.
