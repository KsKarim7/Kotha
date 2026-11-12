# 0005: Decode Whisper at a single temperature without timestamps, word timings on request
Status: proposed · Date: 2026-11-12

## Context
`WhisperAsr` used faster-whisper's defaults: a temperature fallback list, timestamp mode and
word timestamps on every call. Four probes with `eval/asr_decoding_probe.py` (one speaker, the
3 s fixture cut mid-sentence and the 6.4 s K8 clip) showed what that costs:

- **M3-2b (PR #25), `small`.** The fallback (`current`) climbed to temperature 1.0 on every
  clip and produced nonsense in 61-274 s. Every temperature-0.0 row was repeatable. Greedy
  decoding looped to the 224-token window ceiling on both clips; `beam5_t0` avoided the loop
  on K8 only. A `max_new_tokens` cap cut loops but also changed beam output (K8 `beam5_cap`
  looped at N = 130). `small` wrote no Bengali script.
- **M3-4a (PR #27).** Two Bangla fine-tunes, A (`bengaliAI/tugstugi_bengaliai-asr_whisper-medium`)
  and B (`arif11/bangla-ASR-v7`, whisper-small), were converted to CTranslate2. Both were
  trained with the prompt `<|startoftranscript|> <|bn|> <|transcribe|> <|notimestamps|>`, but
  faster-whisper decodes in timestamp mode by default. Both write Bengali script.
- **M3-4b (PR #28).** With `without_timestamps=True`, A's stray leading "া" disappeared
  (A / fixture and K8 / `t0` against `t0_nots`). A's trailing words and B's loops remained.
  They came from a tail window: with `word_timestamps=True` faster-whisper starts the next
  window at the last word's end (`faster_whisper/transcribe.py:1288-1291`), so the last
  0.02-0.32 s was decoded again. B / fixture / `beam5_t0` looped in the main window (224
  tokens, ratio 3.43). Both models produced text on 2 s of silence and of white noise in every
  config, and A's `no_speech_prob` was 0.000 in every row.
- **M3-4c (this change), cuda `int8_float16`, 2 runs after a warm-up.** With
  `word_timestamps=False` every clip decoded in exactly one window for both models (counted
  from faster-whisper's debug log). `t0_nots_nw` on both speech clips gave one segment, was
  repeatable, and matched the M3-4b main-window text exactly, with no tail words:

  | model | clip | tokens | compression ratio | text |
  | --- | --- | --- | --- | --- |
  | A | fixture | 27 | 1.27 | আমার মোবাইল ব্যালেন্স কত আর থার্টি সি |
  | A | K8 | 58 | 1.80 | আমার মোবাইল ব্যালেন্স কত আর থার্টি জিবি ইন্টারনেট প্যাকেজটা অ্যাক্টিভ করতে চাই |
  | B | fixture | 69 | 1.26 | আমার ম্ভাইল ব্যালেন্স কত আর থার্টি- |
  | B | K8 | 158 | 1.70 | আমার ম্ভাইল ব্যালেঞ্জ কতো, আর থার্টি জীবি ইন্টারনেট প্যাকেজ টাক্টিক করতে চায়ি। |

  On silence and noise, B still looped to 224 tokens (ratio 5.89 and 8.44). A wrote 5 words
  on silence (ratio 2.20) and "শ্রী শ্রী" on noise (8 tokens, ratio 1.00).

## Decision
`WhisperAsr.transcribe` always passes `temperature=0.0` (one value, no fallback list),
`without_timestamps=True`, `condition_on_previous_text=False` and `vad_filter=False`, with
`beam_size=1` by default and no `max_new_tokens`. Word timestamps are off by default and
turned on only when a caller asks (`word_timestamps=True`). `Transcript` reports the raw loop
signals `max_compression_ratio` and `max_segment_tokens`; the wrapper does not judge them.

## Alternatives considered
- **Timestamp mode without word timings (`t0_nw`).** One window as well, but A keeps the stray
  "া" on both speech clips (A / fixture and K8 / `t0_nw`). B was clean in it, so a per-model
  setting would work, but one setting that passes the gate for both models is simpler.
- **Word timestamps on every call, with `hallucination_silence_threshold`.** Brings back the
  tail window, and the live path does not use word timings.
- **Beam search (`beam_size=5`).** B looped on the fixture with it (M3-4b), and in M3-2b it
  helped `small` on one clip only, at a higher cost.
- **The temperature fallback.** It retries a looping window at higher temperatures, which can
  rescue a window but makes the output random and slow; in M3-2b it produced nonsense.
- **A `max_new_tokens` cap.** It changed beam output in M3-2b and must be sized per model: B's
  clean K8 output is already 158 tokens, above the 131 tokens M3-2b's cap allowed.
- **`repetition_penalty` or `no_repeat_ngram_size`.** Did not stop the loops, or turned the
  output into fragments (M3-2b).

## Consequences
- **No fallback, so a loop is not retried.** M8 must treat a transcript with
  `max_compression_ratio > 2.4` or `max_segment_tokens` at the 224-token ceiling as unusable
  (re-prompt or hand off). The gate does not catch short hallucinations such as A's
  "শ্রী শ্রী" on noise.
- **Word timings are off in the live path.** M10 (post-call redaction) turns them on and must
  handle the tail window that comes back with them, and check word-time accuracy: in M3-4b
  the first word started at 0.00 s in every row, against Silero's speech onset at 0.35 s.
- **Non-speech must never reach the model.** Both fine-tunes produce text on silence and noise,
  and A's `no_speech_prob` is always 0, so faster-whisper's no-speech skip never fires for A.
  The turn detector has to keep silence and noise out.
- **No token cap.** A window stops at the 224-token ceiling at most.
- **`small` is decoded the same way,** although it was only probed in timestamp mode (M3-2b).

To be confirmed or revised on the dev split in M3-6 (WER per model, the live model and compute
type), and then marked accepted.
