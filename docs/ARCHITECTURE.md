# Architecture

Kotha is at an early stage. This document describes only what exists or has been decided.

## Audio formats

### Internal format

All audio inside Kotha is float32, mono, 16 kHz, with samples in [-1, 1]. Modules exchange
audio in this format. Sample-rate and PCM conversion happen at the edges and inside the
telephony simulator.

### Edge formats (planned)

| Edge | Module | Wire format |
| --- | --- | --- |
| WebSocket | M8 | 16 kHz PCM16, little-endian |
| Asterisk AudioSocket | M9 | 8 kHz PCM16, little-endian |

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
worst segment `avg_logprob`, the highest `no_speech_prob`, and word timings.

- By default it runs on CPU with int8 weights, which is also what CI tests.
- On the local GPU it needs cuBLAS for CUDA 12 from the `gpu` dependency group and
  `LD_LIBRARY_PATH` set before Python starts; see the "Local GPU setup" section of
  `docs/runbook.md` and ADR 0004. `eval/asr_device_check.py` reports which compute types load
  and transcribe on a device.
- Every call passes `condition_on_previous_text=False`, `word_timestamps=True` and
  `vad_filter=False`, since the turn detector has already cut the audio.
- When Whisper returns no segments, the transcript is empty with `worst_avg_logprob` set to
  -inf and `max_no_speech_prob` to 1.0; these are the wrapper's values, not the model's.

The live model and compute type are chosen in M3-6, from WER and the device check.

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
