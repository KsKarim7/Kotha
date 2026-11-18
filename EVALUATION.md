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

Interpretation: [to be written]

## ASR

| Model | Audio | Lang | WER | CER | p50 latency | p95 latency |
| --- | --- | --- | --- | --- | --- | --- |
| [model] | [audio] | [lang] | [measure] | [measure] | [measure] | [measure] |

The text normalisation applied to references and hypotheses will be stated with the results.

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

Interpretation: [to be written]

## Latency

| Stage | p50 | p95 |
| --- | --- | --- |
| vad | [measure] | [measure] |
| asr | [measure] | [measure] |
| retrieval | [measure] | [measure] |
| llm_first_token | [measure] | [measure] |
| tts_first_audio | [measure] | [measure] |
| TTFA | [measure] | [measure] |

Interpretation: [to be written]
