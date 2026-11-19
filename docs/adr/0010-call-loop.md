# 0010: One anyio task group per call, with a decode gate before the dialogue
Status: accepted · Date: 2026-11-19

## Context
M8 connects VAD, ASR, the answer pipeline and TTS into a live call over the WebSocket protocol in
the blueprint (`/v1/calls/stream`, a `start` message with consent, binary PCM16 frames, JSON
events). The blueprint asks for one task group per call (receive loop, turn handling, send loop),
clean shutdown on disconnect, malformed frames that never crash a session, a fallback phrase and
handoff instead of silence, and handoff after two failed turns (FR-10). ADR 0005 says non-speech
must never reach Whisper and that a looping decode (compression ratio above 2.4, or a segment at
the 224-token ceiling) is unusable. Every model call blocks, so none may run on the event loop.

## Decision
- **Task group.** `CallSession` runs its three tasks in an anyio task group. A disconnect, `stop`,
  a refused consent or too many bad messages end the group and cancel the other tasks.
- **Worker threads.** Creating Silero, the turn detector step, Whisper, `AnswerPipeline.reply`
  and each TTS sentence run in `asyncio.to_thread`; only the send loop writes to the socket.
- **Gate and failed turns.** A turn reaches ASR only after the turn detector has confirmed
  speech, and the detector cuts a turn at 30 s. A turn fails when ASR raises or the transcript is
  empty, loops (ratio above 2.4) or hits 224 tokens. The first failed turn gets "please repeat";
  the second in a row gets the handoff reply and a `handoff` event. A TTS error ends the reply's
  audio with a `tts_failed` event; the reply text has already been sent.
- **Bounds.** The outgoing queue holds 64 items; at most one turn waits while the bot is busy
  (no barge-in in MS1), and a further turn is dropped and logged. After 20 refused messages the
  call is closed.
- **One VAD per call.** Silero keeps state for one audio stream, so each call creates its own
  instance in a worker thread; all other models load once at startup.
- **Greeting.** Every call opens with the Bangla greeting after `start`, because the operator is
  Bangladeshi and the caller's language is unknown until they speak. Later fixed replies follow
  the language of the last reply.
- **Protocol additions.** `ready` (sent on connect, so the client knows the audio format before
  `start`) and `reply_audio_end` (so the client knows when a reply's audio is complete) are added
  to the blueprint's events. Frames of any even length up to one second are accepted, a superset
  of the blueprint's 20 ms frames. `handoff` carries the reason only; the structured summary
  comes with M6-full, and `playback_stop` with barge-in in MS2.

## Alternatives considered
- **`asyncio.TaskGroup`.** Starlette's test client ends a WebSocket handler by cancelling it
  through an anyio cancel scope. In a probe with that client, a handler built on
  `asyncio.TaskGroup` ended with `CancelledError` on the first close, while the same handler on an
  anyio task group closed cleanly 20 times out of 20. Under uvicorn a disconnect is a message, not
  a cancellation, but server shutdown does cancel handlers, and anyio works in both cases. anyio
  runs on asyncio here, so the code still uses asyncio queues and `asyncio.to_thread`.
- **Re-asking the LLM to repair a bad transcript.** It would spend an LLM call on text that
  ADR 0005 already marks unusable; asking the caller again is faster and cannot invent a
  question.
- **Unbounded queues.** Simpler, but a client that stops reading, or keeps talking over a long
  reply, would grow the server's memory and get answers to old speech.

## Consequences
- A disconnect never leaves a task running on the loop, but a model call already in a worker
  thread runs to the end and its result is dropped.
- With the Bangla Whisper fine-tune, an English question can come back in Bangla script and get
  a Bangla answer (seen in the first live run); the live ASR model is still open (M3-6).
- Without barge-in, a caller who speaks over a long reply loses all but the latest turn.
- `fastapi`, `uvicorn`, `websockets` (uvicorn's WebSocket implementation) and `anyio` are
  declared runtime dependencies; all four were already installed through other packages.
