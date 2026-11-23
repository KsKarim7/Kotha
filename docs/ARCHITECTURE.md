# Architecture

This document describes what exists at Milestone 1 (a browser call works) and what has been
decided. Later milestones (the phone path, tools, analytics) are not described until they exist.

## Overview

```
browser (mic, AudioWorklet)                          docker compose on one host
   |  PCM16 16 kHz frames, JSON start/stop
   v
/v1/calls/stream  (FastAPI, one CallSession per call, anyio task group)
   receive loop -> Silero VAD + TurnDetector  --(confirmed speech only)--> turn queue
   turn handler -> Whisper (faster-whisper) -> decode gate
                -> AnswerPipeline: account routing -> hybrid retrieval (pgvector + BM25, RRF)
                                   -> Gemini -> citation, number, language and claim checks
                -> speakable text -> MMS-TTS, one sentence at a time
   send loop    -> JSON events + PCM16 audio back to the browser
PostgreSQL 16 + pgvector: kb_chunks (126 chunks of 16 bilingual KB documents)
```

Every model call runs in a worker thread, never on the event loop. Models load and warm once at
startup, before the app accepts connections (`/v1/ready`).

## Audio formats

### Internal format

All audio inside Kotha is float32, mono, 16 kHz, with samples in [-1, 1]. Modules exchange
audio in this format. Sample-rate and PCM conversion happen at the edges and inside the
telephony simulator.

### Edge formats

| Edge | Module | Wire format | State |
| --- | --- | --- | --- |
| WebSocket | M8 | 16 kHz PCM16, little-endian | built |
| Asterisk AudioSocket | M9 | 8 kHz PCM16, little-endian | planned |

A 20 ms frame is 320 samples (640 bytes) at 16 kHz and 160 samples (320 bytes) at 8 kHz.

### Where conversion happens

`kotha.audio.convert` holds all format conversion:

- `pcm16_to_float` and `float_to_pcm16` convert between little-endian PCM16 bytes and
  float32. Both scale by 32768, so every int16 value survives the round trip exactly.
- `resample` changes the sample rate with a polyphase filter.
- `load_mono_16k` reads a WAV or FLAC file, averages its channels and resamples it to 16 kHz.

### Telephony simulation

`kotha.audio.telephony.simulate_telephony` turns 16 kHz audio into what it would sound like
after a phone line, and returns it at 16 kHz with the same length:

- it models the band limit: resampling to 8 kHz removes everything above 4 kHz;
- it models mu-law companding: each 8 kHz sample is quantised to a sign and a 7-bit
  magnitude on a logarithmic scale, then decoded.

It does not model codec packet loss, jitter, line noise or echo. The companding is a
continuous mu-law curve, not bit-exact G.711, which uses segmented tables.

## Voice activity detection

### Speech probability

`kotha.vad.silero.SileroVad` wraps the `silero-vad` 6.x package (the blueprint names v5). It
runs on CPU-only torch from the PyTorch CPU index (ADR 0003), and the model weights ship inside
the package, so nothing is downloaded at runtime. It returns one speech probability for each
512-sample (32 ms) window at 16 kHz. The model is stateful: each call gets its own instance,
and it is not reset between turns.

### Turns

`kotha.vad.turns.TurnDetector` turns those probabilities into turns. It takes any probability
function, so its logic is tested without the model.

- Hysteresis: a turn starts on a window at or above 0.5; after that, any window at or above
  0.35 counts as speech and restarts the hangover.
- Minimum speech is 250 ms, hangover 700 ms and pre-roll 300 ms, each rounded up to whole
  32 ms windows.
- `SpeechStarted` is emitted only once minimum speech is confirmed, so every start is followed
  by exactly one `SpeechEnded` (from the hangover or from `flush()`), and shorter bursts
  produce no events.
- `SpeechEnded.end_sample` marks the true end of speech; its audio also includes the pre-roll
  before the start and the hangover tail after the end.
- `flush()` closes a turn that is still open when the stream ends.

`eval/vad_turns.py` runs both over an audio file and prints each turn's start, end and
duration.

These defaults are the blueprint's; the M2-4 hangover comparison will revisit them. VAD on
8 kHz phone audio is planned for M9.

## Speech recognition

`kotha.asr.whisper.WhisperAsr` wraps faster-whisper (CTranslate2, no torch). It loads one
model per instance and turns 16 kHz float32 audio into a `Transcript`: text, language, the
worst segment `avg_logprob`, the highest `no_speech_prob`, the highest compression ratio, the
most tokens in one segment, and word timings when asked for.

- By default it runs on CPU with int8 weights, which is also what CI tests.
- On the local GPU it needs cuBLAS for CUDA 12 from the `gpu` dependency group and
  `LD_LIBRARY_PATH` set before Python starts; see the "Local GPU setup" section of
  `docs/runbook.md` and ADR 0004. `eval/asr_device_check.py` reports which compute types load
  and transcribe on a device.
- Decoding follows ADR 0005 (proposed). Every call passes `temperature=0.0` (no fallback),
  `without_timestamps=True` (the prompt both Bangla fine-tunes were trained with),
  `condition_on_previous_text=False` and `vad_filter=False`, since the turn detector has
  already cut the audio; `beam_size` defaults to 1 and there is no token cap.
- Word timestamps are off by default, so a turn under 30 s decodes in one window. With
  `word_timestamps=True` faster-whisper starts the next window at the last word's end, which
  decodes the tail of the audio again; only post-call redaction (M10) asks for them.
- `max_compression_ratio` and `max_segment_tokens` are raw loop signals, not a verdict. A
  window stops at 224 tokens at most, and the caller (M8) decides what to reject.
- `join_segment_texts` strips each segment's text, skips empty ones and joins the rest with
  a space; the decoding probe uses the same function.
- When Whisper returns no segments, the transcript is empty with `worst_avg_logprob` set to
  -inf, `max_no_speech_prob` to 1.0, `max_compression_ratio` to 0.0 and
  `max_segment_tokens` to 0; these are the wrapper's values, not the model's.

The app takes its model from `KOTHA_ASR_MODEL`; the Docker stack defaults to
`models/bn-tugstugi-medium-ct2` on the CPU with int8. M3-6 confirms or replaces that choice from
WER and the device check.

## Text normalisation

`kotha.text.normalize.normalize` turns a transcript into the form that WER compares. It
applies these rules in order:

1. Delete format characters (Unicode category Cf: ZWJ, ZWNJ, zero-width space, BOM, soft
   hyphen, bidi marks). They are invisible and keyboards insert them inconsistently; deleting
   them first lets the next rule compose letters they had split.
2. Unicode NFC, so precomposed and decomposed spellings of the same letter (`ো` against
   `ে` + `া`) compare equal.
3. Bangla digits ০-৯ to ASCII 0-9, so a number written in either script compares equal.
4. Lowercase, so Latin case is not counted as an error.
5. Replace every punctuation character (any category starting with P, including the danda
   `।`) with a space, because the transcription guideline has no punctuation. Symbols such as
   `৳` and `+` are kept.
6. Collapse all whitespace (including no-break and thin spaces) to single spaces and strip.

The same function is applied to references and hypotheses, and the reference lint (M1-3)
reuses it, so a reference is checked in the form it is scored in.

It differs from the build guide's sketch in two deliberate ways:

- it removes all Unicode punctuation instead of a fixed list, so dashes, ellipses, curly
  quotes and the double danda `॥` are removed too;
- it deletes format characters, which the sketch leaves in place.

Some side effects apply equally to both sides: `"NT-30GB"` becomes two words (`nt 30gb`),
`"1.5"` becomes `1 5`, and `"50%"` becomes `50`.

It does not transliterate. A hypothesis in the wrong script (for example Devanagari for
Bangla, or `mobile` written in Bangla script) stays an error: later steps receive that text
as written, so transliterating it here would hide an error they still see.

Any change to `normalize` changes every WER number, so all evaluations are rerun after one.

## Knowledge base and retrieval

The knowledge base is 16 fictional documents, each in Bangla and English, as Markdown files in
`data/kb/` (ADR 0006). They hold descriptions, rules and product facts, never prices or balances.
`kotha.kb.chunks` splits each document at its `## ` headings into chunks of up to 300 words (a
single longer paragraph or table stays whole), never splitting a table, with ids `<doc_id>.<lang>.<index>`. `python -m kotha.kb.ingest` embeds
each chunk with bge-m3 (pinned, offline, CPU) and stores it in `kb_chunks` in one idempotent
transaction.

`kotha.rag.retriever.HybridRetriever.search` takes the top 20 chunks by pgvector cosine distance
and the top 20 by our own BM25 (`kotha.rag.bm25`, with the loanword lexicon expanding Latin and
Bangla spellings of English terms), fuses the two rankings by reciprocal rank fusion (k = 60) and
returns 4 chunks. BM25 also scores every table body row on its own and ranks a chunk by its best
unit, so a package code listed once in a long table is found (ADR 0006 update).
`kotha.rag.citations.verify_answer` checks that every sentence of an answer ends
with a citation of a retrieved chunk and that every number in it appears in a cited chunk.

## Answer pipeline

`kotha.dialogue.answer.AnswerPipeline.reply` turns one transcript into one reply (ADR 0008):

1. Code, not the prompt, routes account requests (own balance, buying or cancelling a pack) to a
   fixed "not available yet" reply; MS1 has no account tools.
2. Hybrid retrieval returns 4 chunks; nothing retrieved (a question with no search tokens) gives
   "please repeat", and a database or embedding error gives the handoff reply.
3. Gemini (ADR 0007) gets a constant system prompt and the passages, with the question last. The
   reply language follows the caller: Bangla if the question has any Bengali letter.
4. Code checks the answer (citations, numbers, no money, language, at most two sentences, no
   claim about the caller's account) and regenerates once with a note naming the problems. A
   second failure, `NO_ANSWER`, an LLM error or a slow first attempt gives the handoff reply.

The reply carries its cited sources and the time spent in retrieval and the LLM.

## Text-to-speech

Replies are spoken by two MMS VITS voices, `facebook/mms-tts-ben` and `facebook/mms-tts-eng`
(ADR 0009). Their tokenizers silently drop every character outside their vocabularies, so the
text a voice receives must contain only characters it knows.

- **Speakable text** (`kotha.tts.speakable`). The dialogue keeps digits, because the citation
  verifier checks them; only the TTS input turns them into words. `speakable_sentences` removes
  citation markers, splits the reply into sentences, and writes each one as words: numbers in
  Bangla or English words, short codes and phone numbers digit by digit, `*` and `#` as words,
  Latin terms in Bangla script for the Bangla voice. The last step keeps only characters in the
  voice's vocabulary.
- **Voices** (`kotha.tts.mms.MmsVoice`). Each voice loads once, offline, at its pinned revision.
  Every synthesis sets the torch seed to 0 under one process-wide lock, so the same text gives
  the same audio.
- **Service** (`kotha.tts.service.TtsService`). It routes a reply to the voice for its language
  and gives one float32 array at 16 kHz per sentence. `speak` is an async iterator that runs one
  worker-thread call (`asyncio.to_thread`) per sentence, never on the event loop, so the first
  sentence can play before the rest is ready, and a cancelled reply stops after the sentence in
  progress. `load_tts_service` blocks while it loads both voices and warms the cache, so the app
  must call it in a worker thread at startup. The greeting and the fixed replies (handoff,
  "please repeat", "not available yet") are synthesised once at start and served from a
  read-only cache; answers are not cached. `to_telephony_rate` resamples to 8 kHz for the phone
  path.

## Browser call: WebSocket protocol

The app (`kotha.api.app`, FastAPI) serves the blueprint's endpoint, `/v1/calls/stream`. One
connection is one call. Messages are Pydantic models in `kotha.api.protocol`.

**Client to server.**

- JSON `{"type":"start","sample_rate":16000,"consent":true}` starts the call. `consent` is the
  caller's agreement that their speech is transcribed and their question is sent to the LLM;
  without it the server sends `error` `consent_required` and closes. Audio sent before `start`
  is refused with `not_started` and never reaches the models.
- Binary frames of PCM16 little-endian mono samples at 16 kHz. The blueprint's frame is 20 ms
  (640 bytes); the server accepts any even length up to one second (32,000 bytes) and buffers a
  remainder that does not fill a 32 ms VAD window.
- JSON `{"type":"stop"}`, or closing the socket, ends the call.

**Server to client.** JSON events and binary frames of the bot's audio in the same PCM16 format.

| Event | When | Fields |
| --- | --- | --- |
| `ready` | on connect, before `start` (an addition to the blueprint) | `sample_rate`, `audio_format` |
| `vad` | the turn detector confirmed speech, or ended a turn | `event`: `speech_start` or `speech_end` |
| `transcript` | a turn passed ASR and the gate | `text`, `final` (always true in MS1) |
| `bot_text` | before the bot speaks | `kind` (`greeting`, `answer`, `account_unavailable`, `handoff`, `repeat`), `lang`, `text`, `citations` (`doc_id`, `title`, one per document) |
| `reply_audio_end` | after the last audio frame of a reply (an addition) | none |
| `handoff` | after a handoff reply has been spoken, before the turn's `timing` | `reason` (`no_answer`, `rejected_twice`, `no_time_to_retry`, `llm_timeout`, `llm_unavailable`, `llm_rate_limited`, `llm_error`, `retrieval_failed`, `failed_turns`), also logged on the `reply` event and shown on the page; the blueprint's summary comes with M6-full |
| `error` | a message was refused or TTS failed | `code` |
| `playback_stop` | not sent in MS1; it belongs to barge-in (MS2) | |

Every reply is one `bot_text`, then one binary frame per sentence, then `reply_audio_end`. A
reply to a caller's turn ends with a `timing` event (an addition; after `reply_audio_end` and any
`handoff`, so it is always the turn's last event) with `asr_seconds`,
`retrieval_seconds`, `llm_seconds` and `tts_first_audio_seconds`, each `null` when that stage did
not run; `eval/latency.py` reads it, and each value is also logged as `event=stage`. It also
carries the reply `kind`, the `handoff_reason` (`null` unless a handoff) and `rejections`, the
verifier rules every rejected answer of the turn broke, in order, so a rule broken by both
attempts appears twice (each is also logged as `event=answer_rejected reason=<rule>`, never the
answer itself). Error
codes: `bad_control_message`, `not_started`, `already_started`, `consent_required`,
`odd_length_frame`, `frame_too_large`, `too_many_bad_frames` (after 20 refused messages the call
is closed), `tts_failed` (the text was sent but the audio stopped).

**Readiness.** `GET /v1/ready` returns `{"status":"ready","startup_seconds":…}`. The app loads
and warms every model in its lifespan before uvicorn accepts connections (`kotha.api.app.warm_up`:
Silero, Whisper on synthesised speech, one retrieval search; the voices are warmed by the cached
prompts; the Gemini client is constructed without an API call), so a client that polls this
endpoint never reaches a cold model. A failed load or warm-up stops startup (uvicorn exits), so
the app never serves calls half-loaded. Latency runs measure warm calls only and report cold start
separately.

**Browser client.** The app serves one page at `/` (`kotha/api/static/`), with no external
scripts, fonts or styles. After the caller ticks the consent box and presses Start, the page polls
`/v1/ready`, opens the microphone with echo cancellation, noise suppression and automatic gain
control, and connects to `/v1/calls/stream`. It sends `start` first and only then the audio: an
AudioWorklet (`pcm-worklet.js`) resamples the microphone from the context rate to 16 kHz by
linear interpolation and posts 20 ms frames of PCM16 (320 samples). There is no low-pass filter
before the down-sampling, so energy above 8 kHz can fold into the band; speech carries little
there. Bot audio is played back in arrival order at 16 kHz. The transcript panel shows the
caller's transcript and each bot reply with the titles of its cited documents. End sends `stop`.

**Deployment (MS1).** `docker compose` runs two containers on one host: `db` (PostgreSQL 16 with
pgvector) and `app` (this FastAPI app, CPU-only, models read from read-only mounts). Both ports are
bound to `127.0.0.1`; the app reaches the database at host `db`.

## Browser call: the call loop

`kotha.api.session.CallSession` runs one anyio task group per call with three tasks:

- **Receive loop.** Reads messages, handles `start` and `stop`, refuses malformed ones with an
  `error` event, converts PCM16 to float32, and runs Silero and the turn detector on whole 32 ms
  windows in a worker thread. The detector cuts a turn at 30 s, so a monologue cannot grow
  without limit. Only audio the turn detector has accepted as speech ever reaches ASR.
- **Turn handler.** Waits for `start`, speaks the greeting, then takes one turn at a time: ASR
  in a worker thread, then the gate. A turn fails when ASR raises, or the transcript is empty,
  has a compression ratio above 2.4 or a segment at Whisper's 224-token ceiling (ADR 0005). The
  first failed turn gets "please repeat" in the call's current language; a second failed turn
  in a row gets the handoff reply and a `handoff` event (FR-10). A good turn goes to the client
  as `transcript` and to `AnswerPipeline.reply` in a worker thread, and the reply is spoken
  sentence by sentence through `TtsService.speak`. If TTS fails mid-reply, the client gets
  `tts_failed` and the reply still ends with `reply_audio_end`.
- **Send loop.** The only task that writes to the socket, so events and audio never interleave.
  It also closes the socket after `stop`, a refused consent or too many bad messages.

A disconnect ends the receive or send loop, which ends the task group and cancels the other
tasks. Work already running in a worker thread finishes in the background and its result is
dropped. The task group is anyio's, not `asyncio.TaskGroup` (ADR 0010).

Both queues are bounded. The outgoing queue holds 64 items, so a client that stops reading slows
the call down instead of filling memory. MS1 has no barge-in, so at most one turn waits while the
bot is busy; a further turn is dropped and logged, and the caller gets its `vad` events but no
reply. The browser's echo cancellation keeps the bot's own voice out of the microphone. The limit
of 20 refused messages counts over the whole call, including audio sent before `start`, so a
client should send `start` before it opens the microphone. Closing after a refused consent or too
many refused messages uses code 1008 (policy violation); `stop` closes with 1000. A client that
sends nothing for 30 s (a browser sends a frame every 20 ms) is closed with 1001 (going away), so
any client must keep sending, during the greeting too (the latency harness sends silence then);
and a send that does not finish within 10 s, because the client stopped reading, ends the call.
Both log `event=call_timeout` with the reason `client_silent` or `client_not_reading`. A reply
sentence is one frame (a 6 s sentence is about 192 KB), so a link slower than about 150 kbit/s
could hit the send limit. After a handoff for failed turns the count starts again, so the call
goes on if the caller keeps talking.

Silero, ASR, the pipeline and TTS all share the default `asyncio.to_thread` pool. Under several
concurrent calls, a long LLM call or an abandoned thread can delay another call's VAD step; M8-3
measures one call at a time.

Logs go through `kotha.logs.log_event`, which accepts only allow-listed field names, and string
values only as short tokens that start with a letter. This keeps transcripts and plain phone
numbers out of the logs; it cannot judge what a number in an allowed numeric field means, so
callers must only pass counts and timings there.
