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

Needs: the recorded `asr_test` set with turn boundaries marked, and noise-only clips; then `eval/vad_turns.py` per hangover setting.

Interpretation: [to be written]

## ASR

| Model | Audio | Lang | WER | CER | p50 latency | p95 latency |
| --- | --- | --- | --- | --- | --- | --- |
| [model] | [audio] | [lang] | [measure] | [measure] | [measure] | [measure] |

The text normalisation applied to references and hypotheses will be stated with the results.

Needs: the recorded `asr_test` set (consented speakers, ADR 0002) with references checked by `eval/lint_references.py`; then `eval/wer.py` for each model at 16 kHz and through simulated telephony (runbook, "WER evaluation"). The same recordings give the ASR-transcribed retrieval column.

Interpretation: [to be written]

## Retrieval

| Retriever | Typed questions: hit@3 / MRR | ASR-transcribed questions: hit@3 / MRR |
| --- | --- | --- |
| Dense only | 0.958 / 0.893 | [measure] |
| BM25 only | 0.694 / 0.663 | [measure] |
| Hybrid (RRF) | 0.931 / 0.832 | [measure] |
| Hybrid + lexicon expansion | 0.903 / 0.830 | [measure] |

Typed questions by language, hit@3 / MRR (24 questions each):

| Retriever | bn | en | mix |
| --- | --- | --- | --- |
| Dense only | 0.958 / 0.913 | 0.917 / 0.865 | 1.000 / 0.903 |
| BM25 only | 0.625 / 0.571 | 0.625 / 0.644 | 0.833 / 0.774 |
| Hybrid (RRF) | 0.875 / 0.759 | 0.917 / 0.814 | 1.000 / 0.924 |
| Hybrid + lexicon expansion | 0.875 / 0.759 | 0.917 / 0.814 | 0.917 / 0.917 |

Values are the run's metrics rounded to 3 decimals. Run: `eval/retrieval.py` (runbook, "Retrieval evaluation") on commit `75815fd`, `git_dirty` false, MLflow run `248a1bcae2ad4aef834b73797ffad89a` in the experiment `kotha-retrieval`; 72 typed questions (`rag_test.csv`, SHA-256 `82958d25…`), 126 chunks, bge-m3 at revision `5617a9f6` on CPU (the script's default device), 20 candidates per retriever, RRF k = 60, lexicon `loanwords.csv` with 22 spellings. The ASR-transcribed column needs the recorded test set and the M3 ASR run.

The typed questions were written by the knowledge base's author and will be reviewed by Khaled, which tends to raise the scores. Each question has one gold document, although the knowledge base repeats some facts across documents, so a chunk from another document that also answers the question counts as a miss.

Interpretation: [to be written]

## TTS

| Measure | Before normalisation | After normalisation |
| --- | --- | --- |
| Listening score | [measure] | [measure] |

| Measure | Value |
| --- | --- |
| RTF | [measure] |

Needs: three listeners to rate the 40 blind clips from `eval/listening_sheet.py` (runbook, "Listening test") and a scoring script; RTF from a committed run, which `eval/tts_probe.py` can provide once it logs its timings.

Interpretation: [to be written]

## Latency

Warm calls, one call at a time, on the CPU, in seconds (56 counted turns):

| Stage | p50 | p95 |
| --- | --- | --- |
| vad (true end of speech to the server's `speech_end` event, 700 ms hangover included) | 0.746 | 0.770 |
| asr | 5.261 | 9.357 |
| retrieval | 0.317 | 0.594 |
| llm (whole generation, summed over a regeneration; MS1 does not stream, so first-token time is [measure]) | 1.722 | 9.958 |
| tts_first_audio (reply text to the first synthesised sentence) | 4.496 | 7.826 |
| TTFA, from the true end of speech | 12.742 | 24.036 |
| TTFA, from the VAD decision | 12.002 | 23.278 |

Cold start (process start to `/v1/ready`), reported once and not part of the table: 188.9 s, of which the app's own loading and warm-up (`startup_seconds`) took 179.0 s.

Values are the run's metrics rounded to 3 decimals. Run: `eval/latency.py --launch` (runbook, "Latency evaluation") on commit `aeb6516`, `git_dirty` false, MLflow run `fd75c9a5215c447e87ef24cce5cfcc58` in the experiment `kotha-latency`. ASR: `models/bn-tugstugi-medium-ct2` on the CPU. LLM: Gemini through `GeminiClient` (`gemini-3.1-flash-lite`, retrying on `gemini-3.5-flash-lite`; the model that answered is not logged per turn). Voices: MMS on the CPU.

- **Synthetic inputs.** The callers are the 72 `rag_test` questions (`rag_test.csv`, SHA-256 `82958d25…`) spoken by the MMS voices, not people, and streamed in real time as 20 ms frames. A recorded speaker may give different ASR times and turn ends.
- **What is counted.** One uncounted warm-up turn ran first (question `en01`, which ended in a handoff). The table counts the 56 turns that got an answer, were not split by the VAD and did not fail. Of the 72 turns, 58 got an answer and 14 a handoff; 2 answered turns were split and left out; none failed. Leaving handoffs out also leaves out slow LLM paths that end in a handoff, which biases `llm` and TTFA low.
- **Cold start.** The app was started right after the harness had loaded the MMS voices and synthesised all 72 clips in its own process, on the machine under Hardware. It is one run.

Interpretation: [to be written]
