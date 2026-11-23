# Evaluation

Results for Milestone 1. Every result in this report comes from a committed evaluation run; a value that has not been measured yet is shown as [measure]. Each result states the size of the test set it used and the hardware it ran on.

## Hardware

- CPU: AMD Ryzen 5 5600H
- RAM: 16 GB, of which 10 GB is available to WSL
- GPU: NVIDIA GTX 1650, 4 GB, with about 2.5-2.7 GB free for models because the GPU also drives the screen

## Test sets

| Test set | Contents | Size |
| --- | --- | --- |
| `asr_test` | Utterances with speaker, language (bn / en / mix), noise condition (clean / street / fan) and reference text | [measure] |
| `intent_test` | Utterance text with an intent label | [measure] |
| `rag_test` | Typed questions with the expected document id, written by the knowledge base's author and awaiting Khaled's review | 72 questions: 24 bn, 24 en, 24 mix |
| `calls` | Stereo role-play calls: caller on the left channel, agent on the right | [measure] |

## VAD hangover ablation

| Hangover | Turns cut mid-utterance | False starts on noise | Added latency |
| --- | --- | --- | --- |
| 500 ms | [measure] | [measure] | [measure] |
| 700 ms | [measure] | [measure] | [measure] |
| 900 ms | [measure] | [measure] | [measure] |

Needs: the recorded `asr_test` set with turn boundaries marked, and noise-only clips; `eval/vad_turns.py` per hangover setting; and a scoring script that compares the detected turns with the marked boundaries.

Interpretation: [to be written by Khaled]

## ASR

| Model | Audio | Lang | WER | CER | p50 latency | p95 latency |
| --- | --- | --- | --- | --- | --- | --- |
| [model] | [audio] | [lang] | [measure] | [measure] | [measure] | [measure] |

The text normalisation applied to references and hypotheses will be stated with the results.

Needs: the recorded `asr_test` set (consented speakers, ADR 0002) with references checked by `eval/lint_references.py`; then `eval/wer.py` for each model at 16 kHz and through simulated telephony (runbook, "WER evaluation"). The same recordings give the ASR-transcribed retrieval column.

Interpretation: [to be written by Khaled]

## Retrieval

Current retriever (BM25 also scores table body rows, ADR 0006 update):

| Retriever | Typed questions: hit@3 / MRR | ASR-transcribed questions: hit@3 / MRR |
| --- | --- | --- |
| Dense only | 0.958 / 0.893 | [measure] |
| BM25 only | 0.750 / 0.672 | [measure] |
| Hybrid (RRF) | 0.903 / 0.840 | [measure] |
| Hybrid + lexicon expansion | 0.889 / 0.831 | [measure] |

Typed questions, old and new, hit@3 / MRR. "Old" is MS1's retriever, which scored whole chunks only; "new" also scores each table body row alone:

| Retriever | Old: all | New: all | Old: bn | New: bn | Old: en | New: en | Old: mix | New: mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense only | 0.958 / 0.893 | 0.958 / 0.893 | 0.958 / 0.913 | 0.958 / 0.913 | 0.917 / 0.865 | 0.917 / 0.865 | 1.000 / 0.903 | 1.000 / 0.903 |
| BM25 only | 0.694 / 0.663 | 0.750 / 0.672 | 0.625 / 0.571 | 0.708 / 0.593 | 0.625 / 0.644 | 0.667 / 0.645 | 0.833 / 0.774 | 0.875 / 0.777 |
| Hybrid (RRF) | 0.931 / 0.832 | 0.903 / 0.840 | 0.875 / 0.759 | 0.875 / 0.787 | 0.917 / 0.814 | 0.875 / 0.810 | 1.000 / 0.924 | 0.958 / 0.923 |
| Hybrid + lexicon expansion | 0.903 / 0.830 | 0.889 / 0.831 | 0.875 / 0.759 | 0.875 / 0.813 | 0.917 / 0.814 | 0.875 / 0.810 | 0.917 / 0.917 | 0.917 / 0.869 |

Why the rows changed: a bare package code (`NT-30GB`) did not reach the document that lists it, which BM25 is meant to handle (ADR 0006). Each table body row is now also scored on its own, and a chunk counts with its best unit (ADR 0006 update). The rows join BM25's index statistics, so any BM25 ranking can change; the gold document's BM25 rank changed for 19 of the 72 questions (the two runs' per-question CSVs, logged as their artifacts). Hybrid hit@3 fell by two questions (one en, one mix). Overall hybrid MRR rose (bn rose; en and mix fell slightly), and hybrid + lexicon MRR on mix fell from 0.917 to 0.869. The change was chosen on the package-code regression test, not on these questions. Dense search is unchanged.

Values are the runs' metrics rounded to 3 decimals. Both runs: `eval/retrieval.py` (runbook, "Retrieval evaluation"), `git_dirty` false, experiment `kotha-retrieval`, 72 typed questions (`rag_test.csv`, SHA-256 `82958d25…`), 126 chunks, bge-m3 at revision `5617a9f6` on CPU (the script's default device), 20 candidates per retriever, RRF k = 60, lexicon `loanwords.csv` with 22 spellings.
- Old: commit `75815fd`, MLflow run `248a1bcae2ad4aef834b73797ffad89a`.
- New: commit `e1454e1`, MLflow run `39b2a864ab744d049f478d5e7de0c7ae`.

The ASR-transcribed column needs the recorded test set and the M3 ASR run.

The typed questions were written by the knowledge base's author and will be reviewed by Khaled, which tends to raise the scores. Each question has one gold document, although the knowledge base repeats some facts across documents, so a chunk from another document that also answers the question counts as a miss.

Interpretation: [to be written by Khaled]

## TTS

| Measure | Before normalisation | After normalisation |
| --- | --- | --- |
| Listening score | [measure] | [measure] |

Real-time factor (synthesis time / audio length, lower is faster), CPU:

| Voice | Sentences | RTF p50 | RTF p95 |
| --- | --- | --- | --- |
| bn (`facebook/mms-tts-ben`) | 14 | 0.230 | 0.242 |
| en (`facebook/mms-tts-eng`) | 6 | 0.224 | 0.227 |

Run: `eval/tts_probe.py --sentences listening` (runbook, "Text-to-speech voices") on commit `46d3bed`, `git_dirty` false, MLflow run `ebbb1c6999684979b92e275149a5b4fb` in the experiment `kotha-tts`; the 20 listening-sheet sentences after the speakable-text step, one synthesis per sentence after one untimed warm-up per voice, seed 0, 6 torch threads, nothing else running. The en p95 rests on six sentences.

Needs: three listeners to rate the 40 blind clips from `eval/listening_sheet.py` (runbook, "Listening test") and a scoring script.

Interpretation: [to be written by Khaled]

## Latency

Warm calls, one call at a time, in seconds, p50 / p95. Each column is one configuration and one run. Stage rows count answered turns only (not split, not failed); `ttfa_all_turns` counts every heard turn, handoffs included: the time until the caller hears anything.

| Stage | MS1: CPU, tugstugi `int8` | CPU, tugstugi `int8`, primary rest | CPU, tugstugi `int8` | cuda, tugstugi `int8_float16` | cuda, arif11 `int8_float16` (quota ran out) |
| --- | --- | --- | --- | --- | --- |
| vad (true end of speech to `speech_end`, 700 ms hangover included) | 0.746 / 0.770 | 0.747 / 0.767 | 0.747 / 0.762 | 0.749 / 0.767 | 0.750 / 0.766 |
| asr | 5.261 / 9.357 | 6.508 / 8.764 | 5.223 / 9.045 | 1.621 / 1.891 | 1.111 / 1.332 |
| retrieval | 0.317 / 0.594 | 0.694 / 1.594 | 0.340 / 0.966 | 0.319 / 0.491 | 0.313 / 0.422 |
| llm (whole generation; no streaming, so first-token time is [measure]) | 1.722 / 9.958 | 2.262 / 8.790 | 5.702 / 9.950 | 4.124 / 8.847 | 5.383 / 9.507 |
| tts_first_audio (reply text to the first synthesised sentence) | 4.496 / 7.826 | 4.702 / 9.395 | 4.180 / 10.532 | 4.190 / 9.960 | 4.293 / 8.913 |
| TTFA, from the true end of speech | 12.742 / 24.036 | 16.867 / 26.716 | 16.981 / 27.149 | 11.928 / 20.113 | 12.064 / 18.255 |
| TTFA, from the VAD decision | 12.002 / 23.278 | 16.123 / 26.007 | 16.268 / 26.417 | 11.178 / 19.390 | 11.314 / 17.505 |
| TTFA over all heard turns (`ttfa_all_turns`) | not computed (added later) | 14.852 / 26.693 (70) | 16.115 / 24.340 (70) | 10.239 / 19.132 (69) | 8.201 / 16.625 (70) |
| Counted turns (answers, not split, not failed) | 56 | 44 | 45 | 44 | 27 |
| Replies of 72: answers / handoffs / other | 58 / 14 / 0 | 44 / 28 / 0 | 46 / 26 / 0 | 46 / 26 / 0 | 27 / 44 / 1 repeat |
| Split / failed turns | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 1 (`en09` timeout) | 2 / 0 |
| Commit | `aeb6516` | `27a682a` | `46d3bed` | `46d3bed` | `46d3bed` |
| MLflow run (`kotha-latency`) | `fd75c9a5215c447e87ef24cce5cfcc58` | `d4a5309656f24934bff891afd188a03b` | `fba66a2fea3f4488b5c2ca186907307d` | `64d884be816f4b769cee3525650266bc` | `0b9f98d263854426959ade776678eee2` |

Handoff reasons (from the per-turn CSV; the MS1 run did not record them):

| Reason | Primary rest (`d4a53096`) | CPU (`fba66a2f`) | cuda tugstugi (`64d884be`) | cuda arif11 (`0b9f98d2`) |
| --- | --- | --- | --- | --- |
| `no_answer` | 24 (bn 6, en 10, mix 8) | 21 (bn 3, en 11, mix 7) | 23 (bn 5, en 12, mix 6) | 26 (bn 5, en 16, mix 5) |
| `llm_timeout` | 3 | 3 | 2 | 1 |
| `llm_unavailable` | 1 | 0 | 0 | 0 |
| `llm_rate_limited` | 0 | 0 | 0 | 17 |
| `no_time_to_retry` | 0 | 2 | 1 | 0 |
| Verifier rules on those turns | none | `unsupported_number` 2, `missing_citation` 1, `too_long` 1 | `missing_citation` 1, `unsupported_number` 1, `too_long` 1 | none |

Why the columns differ, as facts of the runs:
- **Primary rest.** Commit `27a682a` let `gemini-3.1-flash-lite` rest for 60 s after a failure (#64). That run's handoffs rose to 28 (24 `no_answer`, against 14 handoffs in MS1), and the rest was removed in #76 (ADR 0007, second update). The other new columns run on `46d3bed`, without the rest.
- **Gemini on that day.** In diagnostic probes (not measurements) `gemini-3.1-flash-lite` returned 503 to most requests, so most turns were answered by the retry model; the model that answered is not logged per turn. The two models answer differently (ADR 0007), so `llm` and the handoff counts depend on Google's load, not only on the code.
- **arif11 column.** The key's daily Gemini quota ran out during this run (17 `llm_rate_limited`, from `mx04` on, in the last third of the run); it has 27 counted turns, below the blueprint's 30, so only its `vad` and `asr` rows compare with the other columns.
- **`en09` failure (cuda tugstugi).** The app logged an `answer` reply and then nothing within the harness's 120 s turn limit: the reply's speech did not finish in time. The cause is not known yet (the answer text is not logged).

Run conditions, all columns: `eval/latency.py --launch` (runbook, "Latency evaluation"), `git_dirty` false; the 72 `rag_test` questions (`rag_test.csv`, SHA-256 `82958d25…`) spoken by the MMS voices, not people, streamed in real time as 20 ms frames, one call at a time, after one uncounted warm-up turn; LLM: Gemini through `GeminiClient`; voices and the embedder on the CPU; values are the runs' metrics rounded to 3 decimals. The cuda runs ran after the runbook's GPU check passed in the same session (`P0`, every run at `sm_mhz_limit`, no clock event reasons). The runs on `46d3bed` ran from a clean worktree of that commit with the main checkout's virtual environment (the worktree's own `uv sync` stalled on a download), with nothing else heavy on the machine. Leaving handoffs out of the stage rows also leaves out slow LLM paths that end in a handoff, which biases `llm` and TTFA low; `ttfa_all_turns` includes them. A recorded speaker may give different ASR times and turn ends.

### Cold start

One cold start on its own (nothing else loading models), CPU, tugstugi `int8`, in seconds:

| Step | Seconds |
| --- | --- |
| `asr_load` | 35.688 |
| `llm_client` | 0.021 |
| `embedder_load` (includes importing torch and transformers) | 9.125 |
| `retriever_index` (chunks and BM25) | 0.157 |
| `tts_load_bn` / `tts_load_en` | 0.808 / 1.088 |
| `tts_prompts_bn` / `tts_prompts_en` (fixed replies synthesised and cached) | 6.243 / 5.802 |
| `vad_warm_up` | 0.114 |
| `asr_warm_up_speech` / `asr_warm_up` | 1.389 / 5.401 |
| `retrieval_warm_up` (first search) | 0.472 |
| Not covered by a step (`other_startup`) | 0.407 |
| App startup (`startup_seconds`) | 66.715 |
| Before startup (Python, uvicorn, the app module, up to 0.5 s of readiness polling) | 1.365 |
| Cold start, process start to `/v1/ready` | 68.080 |

Run: `eval/cold_start.py` (runbook, "Cold start") on commit `46d3bed`, `git_dirty` false, MLflow run `ca512651fdfd407cb142e4a6875e2e71` in the experiment `kotha-cold-start`. It is one run; the operating system's file cache was warm from the earlier runs that day. The cold starts the latency runs report (MS1 188.9 s, then 115.7, 260.0, 184.9 and 44.1 s) were each measured right after the harness had loaded the MMS voices and synthesised 72 clips in its own process, so they are not comparable with this one or with each other.

Interpretation: [to be written by Khaled]
