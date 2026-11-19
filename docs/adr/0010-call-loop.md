# 0010: One anyio task group per call, with a decode gate before the dialogue
Status: accepted · Date: 2026-11-19

## Context
M8 connects VAD, ASR, the answer pipeline and TTS into a live call over a WebSocket. The blueprint
asks for one task group per call (receive loop, turn handling, send loop), clean shutdown on
disconnect, and malformed frames that never crash a session. ADR 0005 says non-speech must never
reach Whisper and that a looping decode (compression ratio above 2.4, or a segment at the
224-token ceiling) must be treated as unusable. Every model call blocks, so none may run on the
event loop.

## Decision
- **Task group.** `CallSession` runs its three tasks in an anyio task group. The receive loop
  raises `CallEnded` on disconnect, which ends the group and cancels the other tasks.
- **Worker threads.** Silero with the turn detector, Whisper, `AnswerPipeline.reply` and each TTS
  sentence run in `asyncio.to_thread`; only the send loop writes to the socket.
- **Gate.** A turn reaches ASR only after the turn detector has confirmed speech. A transcript
  that is empty, has a compression ratio above 2.4 or a segment of 224 tokens gets the fixed
  "please repeat" reply in the call's current language and never reaches the pipeline.
- **One VAD per call.** Silero keeps state for one audio stream, so each call loads its own
  instance (about 0.04 s, measured) instead of sharing one per process; all other models load
  once at startup.
- **Greeting.** Every call opens with the Bangla greeting, because the operator is Bangladeshi
  and the caller's language is unknown until they speak. Later fixed replies follow the
  language of the last reply.
- **No barge-in in MS1.** Turns that arrive while the bot speaks are queued and answered in order.

## Alternatives considered
- **`asyncio.TaskGroup`.** Starlette ends a WebSocket handler through anyio cancel scopes. In a
  probe with Starlette's test client, a handler built on `asyncio.TaskGroup` ended with
  `CancelledError` on the first close, while the same handler on an anyio task group closed
  cleanly 20 times out of 20. anyio runs on asyncio here, so the code still uses asyncio queues
  and `asyncio.to_thread`.
- **Re-asking the LLM to repair a bad transcript.** It would spend an LLM call and a few seconds
  on text that ADR 0005 already marks unusable; asking the caller again is faster and cannot
  invent a question.
- **Answering in the language of the transcript only.** Done for answers (the pipeline replies in
  the caller's language); the greeting has no transcript yet.

## Consequences
- A disconnect never leaves a task running on the loop, but a model call already in a worker
  thread runs to the end and its result is dropped.
- With the Bangla Whisper fine-tune, an English question can come back in Bangla script and get
  a Bangla answer (seen in the first live run); the live ASR model is still open (M3-6).
- Without barge-in, a caller who speaks over a long reply waits until it ends; MS2 adds barge-in.
- `anyio` is now a declared runtime dependency (it was already installed through Starlette).
