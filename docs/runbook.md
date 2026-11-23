# Runbook

## Local GPU setup

faster-whisper runs on the GPU through CTranslate2, which loads cuBLAS for CUDA 12
(`libcublas.so.12`) at runtime. uv installs it with the `gpu` dependency group, which is a
default group locally and is turned off in CI (`UV_NO_GROUP: gpu`). The Windows NVIDIA
driver provides the rest: never install a driver or the CUDA toolkit inside Ubuntu.

One-time setup, in an Ubuntu terminal:

```bash
cd ~/kotha && uv sync
grep -qF 'nvidia/cublas/lib' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
_kotha_cublas="$HOME/kotha/.venv/lib/python3.11/site-packages/nvidia/cublas/lib"
case ":${LD_LIBRARY_PATH:-}:" in
  *":$_kotha_cublas:"*) ;;
  *) export LD_LIBRARY_PATH="$_kotha_cublas${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
unset _kotha_cublas
EOF
```

The block adds the path only if it is not already there, so re-sourcing `~/.bashrc` or
opening nested shells doesn't repeat it.

Then open a new terminal, and in VS Code run "Developer: Reload Window" so that its terminals
and extensions pick up the variable.

Check that it worked; every compute type that the GPU supports should show `ok` as `yes`:

```bash
cd ~/kotha && uv run python eval/asr_device_check.py --model tiny
```

If transcribing fails with `Library libcublas.so.12 is not found or cannot be loaded`:

- run `uv sync` again; a sync with `--no-default-groups`, `--no-group gpu` or `UV_NO_GROUP`
  set removes the gpu group;
- run `echo $LD_LIBRARY_PATH` in a new terminal; it must contain the `nvidia/cublas/lib`
  path above.

### Power settings (Windows side)

Timings on `cuda` depend on Windows keeping the GPU at full clock. Without these settings, the
GPU ran in a power-saving state that made cuda slower than cpu (PR #33, PR #34).

- The laptop charger is plugged in.
- Windows Settings > System > Power & battery > Power mode: "Plugged in" = Best Performance.
- NVIDIA Control Panel > 3D Settings > Manage 3D settings > Global Settings tab: scroll the
  Settings list to "Power management mode", choose "Prefer maximum performance", then Apply.
  On some machines the same setting is in the NVIDIA App under Graphics > Global Settings.
- Restart Windows after changing them; a Windows restart also restarts WSL.

#### Check before timing anything on the GPU

```bash
cd ~/kotha && uv run python eval/gpu_bench.py --model models/bn-arif11-v7-small-ct2 \
  --device cuda --compute-types int8_float16 --stages encoder --runs 30 \
  --clip /mnt/g/Kotha/data/k8_spectrogram.m4a
```

Healthy means `pstates` is `P0` and every value in `sm_mhz_runs` equals `sm_mhz_limit`. The
first one or two runs may show `-` because `nvidia-smi` hasn't sent its first sample yet; that
doesn't count against the check. If any run shows 855 MHz or 1380 MHz, the settings above
aren't in effect.

### GPU benchmark

`eval/gpu_bench.py` times the stages of one transcription and records the GPU's state while
they run. It loads the model once per compute type, does one untimed warm-up transcription,
and then times `--runs` rounds of three stages: `features` (the log-Mel features of the clip),
`encoder` (the encoder alone, on one padded 30 s window) and `whole` (a full transcription with
the same settings as `WhisperAsr`).

```bash
cd ~/kotha && uv run python eval/gpu_bench.py --model models/bn-arif11-v7-small-ct2 \
  --device cuda --compute-types float16 int8_float16 \
  --clip /mnt/g/Kotha/data/k8_spectrogram.m4a
```

- `wall_*_ms` is elapsed time, and `wall_ms_runs` lists every run in order, so a slowdown
  over time shows. `cpu_median_ms` is the CPU time of all threads of the process.
- `cpu_wall_ratio` is CPU time over wall time. Above 1 means several threads were busy. Near 1
  on cuda means one thread was busy the whole time: either the GPU path is CPU-bound (with
  low `util_pct_median`) or a thread spin-waits for the GPU (with high utilisation).
- `decode_ms_*` is the `whole` time minus the `encoder` and `features` times, an estimate of
  the autoregressive decoder; `decode_ms_per_token` divides it by the tokens produced.
- On cuda, `nvidia-smi` is sampled every `--sample-ms` (default 100). Each stage gets the
  P-states seen, the SM clock (median, max, limit), power, utilisation, temperature and the
  active clock event reasons (for example `sw_power_cap` or `sw_thermal_slowdown`); the
  `*_runs` columns repeat clock, power, temperature and reasons per run. A stage shorter than
  one sample interval may have no samples.
- `mem_mhz_*` and `mem_mhz_runs` give the memory clock (median, max, limit, and per run) in
  the same way as the SM clock; it follows the P-state, for example 6000 MHz in P0 and
  405 MHz in P8 on the GTX 1650.
- Starting `nvidia-smi` briefly wakes the GPU, so `--sample-ms 0` turns sampling off for a
  control run. `--stages` limits the run to some stages, for example
  `--stages encoder --runs 30` to watch the GPU under sustained load.
- No transcript text is printed. These timings describe one machine in one state; they are
  not latency results for `EVALUATION.md`.
- Save output worth keeping outside the repository with
  `| tee /mnt/g/Kotha/data/eval/gpu/<date>-<what>.tsv`, because `/tmp` is cleared when WSL
  restarts. The reference run with the settings above is
  `2026-11-13-nvidia-max-{burst,B-whole,A-whole}.tsv`.

## Local database

The knowledge base lives in PostgreSQL 16 with pgvector, from the pinned image
`pgvector/pgvector:0.8.7-pg16` (ADR 0006). Docker Desktop must be running with WSL integration on.

One-time setup: copy `.env.example` to `.env` and replace the placeholders. `.env` is git-ignored;
never commit it. Make the password with `openssl rand -hex 24`: characters such as `/`, `@`, `%`,
`$` or spaces break `DATABASE_URL` or the shell. `POSTGRES_PASSWORD` must match the password inside
`DATABASE_URL`. The credentials take effect only when the volume is first created; after changing
them, delete the volume (`docker compose down -v`) or change the role's password in the database.

```bash
cd ~/kotha && cp -n .env.example .env   # then edit .env
docker compose up -d db
docker compose ps db                     # wait until it says "healthy"
```

- The port is bound to `127.0.0.1` only, so the database is not reachable from the network.
- Compose passes only `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB` from `.env` into the
  container (not the other keys); without `POSTGRES_PASSWORD` the image refuses to start.
- Data is kept in the named volume `kotha_pgdata` (Compose adds the project name);
  `docker compose down -v` deletes it.
- `kotha.db.schema.prepare_database` creates the `vector` extension and the `kb_chunks` table if
  they are missing, inside the caller's transaction: the caller commits. Creating the extension
  needs a superuser role, which the local and CI databases use. Search is exact cosine search with
  no index while the KB is small (ADR 0006).

Database tests read `DATABASE_URL` and are skipped when it is not set, so `uv run pytest -q` works
without Docker. To run them against the local database (each test rolls back its changes):

```bash
cd ~/kotha && set -a && . ./.env && set +a && uv run pytest -q tests/test_db.py
```

CI runs the same tests against a service container of the same image. There `CI` is set, so a
missing `DATABASE_URL` makes the tests fail instead of being skipped.

## Bangla Whisper models

Two Bangla fine-tunes of Whisper, converted to CTranslate2 so that faster-whisper can load
them from a local path (`WhisperModel("models/<name>-ct2", ...)`, or `--model` in the `eval/`
scripts):

| Output | Source repo | Pinned SHA | Licence | Base model | Quantization | Size |
| --- | --- | --- | --- | --- | --- | --- |
| `models/bn-tugstugi-medium-ct2` | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` | `da605cc1bd2f60a18d8e440e977ddfa921a88e63` | Apache-2.0 | whisper-medium (from its `config.json`) | float16 | 1460 MiB |
| `models/bn-arif11-v7-small-ct2` | `arif11/bangla-ASR-v7` | `9fd541fa69c40ae93b10266dfe7fc6476d72e447` | Apache-2.0 | openai/whisper-small | float16 | 466 MiB |

`models/` is git-ignored, and the `check-added-large-files` pre-commit hook (500 KB) would
block model files anyway.

The conversion uses the project's own transformers, a runtime dependency since
sentence-transformers was added (ADR 0006). The models were first converted with transformers
5.9.0, the version CTranslate2 4.9.0's own tests pin. On 2026-11-18 both conversions were repeated
with the project's transformers 5.23.0, and every output file was byte-identical. Don't use
`uv run --with` for these steps: an overlay can replace the project's CPU-only torch.

Download each source at its pinned SHA first. `ct2-transformers-converter --revision` pins
the weights but not the files that `--copy_files` fetches, which come from `main`.
`training_args.bin` and the training logs are not downloaded.

```bash
cd ~/kotha && uv run python - <<'PY'
from huggingface_hub import snapshot_download

files = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json", "normalizer.json",
    "model.safetensors", "pytorch_model.bin",
]
snapshot_download(
    "bengaliAI/tugstugi_bengaliai-asr_whisper-medium",
    revision="da605cc1bd2f60a18d8e440e977ddfa921a88e63",
    local_dir="models/src/bn-tugstugi-medium",
    allow_patterns=files,
)
snapshot_download(
    "arif11/bangla-ASR-v7",
    revision="9fd541fa69c40ae93b10266dfe7fc6476d72e447",
    local_dir="models/src/bn-arif11-v7-small",
    allow_patterns=files,
)
PY
```

`arif11/bangla-ASR-v7` has no `tokenizer.json`, which faster-whisper needs. Build it from
`vocab.json` and `merges.txt`, and copy only that file into the source directory:

```bash
uv run python -c 'from transformers import WhisperTokenizerFast; WhisperTokenizerFast.from_pretrained("models/src/bn-arif11-v7-small").save_pretrained("models/src/bn-arif11-v7-small-tokenizer")'
cp models/src/bn-arif11-v7-small-tokenizer/tokenizer.json models/src/bn-arif11-v7-small/
```

Its weights are a pickle (`pytorch_model.bin`). transformers loads them with
`torch.load(..., weights_only=True)`, which refuses arbitrary Python objects; never turn that
off.

Convert both:

```bash
for name in bn-tugstugi-medium bn-arif11-v7-small; do
  uv run ct2-transformers-converter --model "models/src/$name" \
    --output_dir "models/$name-ct2" --copy_files tokenizer.json preprocessor_config.json \
    --quantization float16
done
```

## Embedding model and knowledge-base ingestion

Retrieval embeds text with `BAAI/bge-m3` (MIT) through sentence-transformers, pinned at revision
`5617a9f61b028005a4858fdac845db406aefb181` (ADR 0006). `kotha.rag.embedder.Embedder` loads it
only from the local Hugging Face cache (`local_files_only=True`), on the CPU by default, and sets
`DISABLE_SAFETENSORS_CONVERSION=true` first, so it never contacts the Hub and never downloads
weights from a bot pull request. Download it once (about 2.2 GB, only the files the model needs):

```bash
cd ~/kotha && uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download("BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181", allow_patterns=["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json", "1_Pooling/config.json", "config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"])'
```

Then chunk, embed and store the knowledge base in the local database (see "Local database"):

```bash
cd ~/kotha && set -a && . ./.env && set +a && uv run python -m kotha.kb.ingest
```

- It reads `data/kb/` (`--kb-dir` to change it), chunks every document (`kotha.kb.chunks`),
  embeds "title, heading, text" for each chunk, upserts the chunks into `kb_chunks` and deletes
  chunks that are no longer in the KB, all in one transaction. Running it again gives the same
  table, so it is safe to rerun after editing the KB.
- It prints how many chunks were stored and removed. On the CPU it takes a few minutes and about
  3 GB of RAM.
- The test that loads the real model is skipped when the model is not in the cache, as in CI.

## Text-to-speech voices

Replies are spoken by `facebook/mms-tts-ben` (Bangla) and `facebook/mms-tts-eng` (English), both
VITS models licensed CC-BY-NC 4.0 (non-commercial only), pinned in ADR 0009. Download them once
(about 277 MiB, `model.safetensors` and the small JSON files only):

```bash
cd ~/kotha && uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download("facebook/mms-tts-ben", revision="0da99de6074c8829121cdabfbdba423af18e8e56", allow_patterns=["*.json", "model.safetensors"]); snapshot_download("facebook/mms-tts-eng", revision="c71de0fe7204c83f1c10820a7d696d0b450048ba", allow_patterns=["*.json", "model.safetensors"])'
```

`eval/tts_probe.py` synthesises a fixed set of bn and en sentences with both voices at seed 0 and
prints, per sentence, the audio length, the synthesis time, the RTF, the text the tokenizer keeps
and every character it drops. It writes one WAV per sentence to `--output-dir`, which must be
outside the repository. `--asr-model` transcribes each clip back with faster-whisper as a rough
hint of what the audio sounds like; it is not a pronunciation score. One model transcribes every
clip, so run it twice: the Bangla fine-tune for the bn clips and multilingual `medium` for the en
clips (see "Bangla Whisper models" for the local model):

```bash
cd ~/kotha && HF_HUB_OFFLINE=1 uv run python eval/tts_probe.py \
  --output-dir ~/kotha-runs/tts-probe/bn-asr --asr-model models/bn-tugstugi-medium-ct2
cd ~/kotha && HF_HUB_OFFLINE=1 uv run python eval/tts_probe.py \
  --output-dir ~/kotha-runs/tts-probe/en-asr --asr-model medium
```

- The tokenizer drops every character outside the voice's `vocab.json` without a warning, so a
  `dropped:` line means that character is never spoken. The blank token, `6` for Bangla and `k`
  for English, is listed as dropped too: the tokenizer gives it the blank's id (ADR 0009).
- The timings are from one run on whatever the CPU is doing at the time; they are not results for
  `EVALUATION.md`.

### Listening test

`data/listening/tts_sentences.csv` holds 20 fixed sentences (14 bn, 6 en; data sizes, times, codes,
phone numbers and English terms; no prices). `eval/listening_sheet.py` renders each one twice with
the real voices: `before` sends the raw text to the voice, `after` sends `kotha.tts.speakable`
output. The 40 clips are shuffled with a fixed seed and named `clip-01.wav` to `clip-40.wav`, so a
listener cannot tell the versions apart:

```bash
cd ~/kotha && HF_HUB_OFFLINE=1 uv run python eval/listening_sheet.py \
  --output-dir ~/kotha-runs/listening/v2/clips --key ~/kotha-runs/listening/v2/key.csv
```

- `ratings.csv` is the sheet for listeners: clip, language, the intended sentence and one empty
  column per listener. Each listener plays every clip and writes a score from 1 (cannot understand
  it) to 5 (clear and natural), judging whether the clip says the intended sentence, numbers and
  codes included.
- `--key` writes the CSV that maps each clip to its sentence, category, version and the exact text
  sent to the voice. It must be outside `--output-dir`, because that folder is what listeners get.
  Whoever renders the clips has seen the key and should not be one of the three listeners.
- Both paths must be outside the repository. Scores go into `EVALUATION.md` only after three
  listeners have rated every clip.

## Retrieval evaluation

`eval/retrieval.py` asks every question in `data/testsets/rag_test.csv` and scores four
retrievers: dense only, BM25 only, hybrid, and hybrid with lexicon expansion. A question counts
as found at the rank of the first chunk from its gold document, within the top 20. The script
prints hit@3 and MRR overall and per language, and writes one row per question and retriever to
`--output`, which must be outside the repository.

It needs Docker Desktop running, the knowledge base ingested and bge-m3 in the Hugging Face cache
(see "Local database" and "Embedding model and knowledge-base ingestion"):

```bash
cd ~/kotha && docker compose up -d --wait db && set -a && . ./.env && set +a \
  && HF_HUB_OFFLINE=1 uv run python -m kotha.kb.ingest \
  && HF_HUB_OFFLINE=1 uv run python eval/retrieval.py \
    --output ~/kotha-runs/retrieval/typed.csv --mlflow-dir ~/kotha-runs/mlflow
```

- With `--mlflow-dir`, the run goes to the experiment `kotha-retrieval`, with metrics named like
  `hybrid_lexicon.mix.hit_at_3`, and the script prints `mlflow run: <run id>`.
- A number counts for `EVALUATION.md` only if the run has `git_dirty` false.
- The script refuses to run when `kb_chunks` differs from `data/kb/`, and it logs the SHA-256 of
  the test set, so a run can be tied to the exact questions and knowledge base. The check compares
  chunk text, not vectors, so run the ingestion again after changing the embedding model.
- The questions were written by the author of the knowledge base, which tends to raise the scores.
  Khaled reviews them. Change a question only to fix its wording or its gold document, never
  because of how it scored, and rerun the evaluation after any change.

## Running the app

The app serves the browser call on `/v1/calls/stream` (protocol in `docs/ARCHITECTURE.md`). It needs the
database with the knowledge base ingested, bge-m3, the MMS voices, a converted Whisper model and
the Gemini key in `.env` (see the sections above):

```bash
cd ~/kotha && docker compose up -d --wait db && set -a && . ./.env && set +a \
  && KOTHA_ASR_MODEL=models/bn-tugstugi-medium-ct2 HF_HUB_OFFLINE=1 \
  uv run uvicorn kotha.api.app:app --host 127.0.0.1 --port 8000 --ws-max-size 65536
```

- Open http://127.0.0.1:8000 in Chrome or Firefox on the same machine (a browser only allows the
  microphone on `localhost` or HTTPS), tick the consent box, press Start, wait for the greeting,
  then speak. Use headphones if echo cancellation lets the bot's voice back in.
- `KOTHA_ASR_MODEL` is required: a faster-whisper model name or a local CTranslate2 folder.
  `KOTHA_ASR_DEVICE` defaults to `cpu`; use `cuda` only after the GPU check above, with
  `KOTHA_ASR_COMPUTE_TYPE=int8_float16` (the default `int8` is the CPU setting).
- `--ws-max-size 65536` keeps uvicorn from buffering frames far larger than the one-second limit
  the app enforces (32,000 bytes).
- Startup loads every model and warms it: Silero once, the voices through the fixed prompts,
  Whisper on one synthesised Bangla sentence, and one retrieval search (no LLM call). uvicorn
  accepts no connection until this is done, so no call pays for loading a model. If loading or
  warm-up fails, uvicorn logs "Application startup failed" and exits with code 3, and
  `/v1/ready` never answers.
- `GET /v1/ready` answers `{"status":"ready","startup_seconds":…}` once startup is complete;
  before that the connection is refused. Clients (the browser page, `eval/latency.py`) poll it
  before they open a call. `startup_seconds` covers loading and warm-up inside the app; the time
  from process start to ready also includes Python imports and is measured from outside.
- Logs carry only event names, call ids, reply kinds, languages, handoff reasons, verifier rule
  names and error codes; never what the caller or the bot said.
- When an answer of a handed-off turn was rejected (always for `rejected_twice` and
  `no_time_to_retry`, and for an `llm_*` or `no_answer` handoff after a retry), the rules it broke
  are logged as `event=answer_rejected reason=<rule>` lines with the same `call_id`.
- After a handoff the page says why, and the log has it:
  `docker compose logs app | grep 'event=reply.*reason='` (or the uvicorn output). An LLM reason
  names how the last attempt failed; earlier attempts may have failed differently. They point
  outside the code:
  - `llm_unavailable` (the page says "overloaded or unreachable"): the last attempt got a 5xx, a
    connection failure or an unreadable response. On the free tier this is usually Google's
    capacity (503 "high demand"). Wait a minute and ask again; if it persists, check the network
    and Google's status page, or move to a paid tier.
  - `llm_rate_limited`: a rate limit or quota was reached (429). A per-minute limit clears within
    a minute; daily quotas reset at midnight Pacific time. The limits are in Google AI Studio.
  - `llm_timeout`: the last attempt timed out (or Gemini returned 408); usually overload as well.
  - `llm_error`: the request was refused (4xx, for example a revoked key) or the answer was cut
    off. Check `GEMINI_API_KEY` in `.env` and that the app started without errors.

## Docker stack

`docker compose up` starts the database and the app. The app image (`Dockerfile`) is CPU-only: it
installs the locked runtime dependencies without the `gpu` group, so torch comes from the PyTorch
CPU index (ADR 0003). The image holds no model weights. The app reads them from two read-only
mounts, the Hugging Face hub cache (`HF_HUB_CACHE_DIR`, default `~/.cache/huggingface/hub`; only
the hub folder, so a saved Hugging Face login token is not exposed) and `./models`, and runs with
`HF_HUB_OFFLINE=1`, so download them first (see "Embedding model and knowledge-base ingestion",
"Text-to-speech voices" and "Bangla Whisper models"). Create `models/` before the first
`docker compose up`; if it is missing, Docker creates it owned by root.

```bash
cd ~/kotha && docker compose up -d --wait db
docker compose run --rm app python -m kotha.kb.ingest
docker compose up -d --wait app
```

- The ingestion runs once per database volume, and again after `data/kb/` changes; it is not
  repeated at every start because it re-embeds every chunk. Without it the app still starts, but
  retrieval finds nothing, so every policy question gets "please repeat".
- `DATABASE_URL` is built from `POSTGRES_PASSWORD` without encoding, so the password must be
  URL-safe (the `openssl rand -hex 24` password from "Local database" is).
- Inside the stack the app reaches the database at host `db`; compose builds `DATABASE_URL` from
  `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB` in `.env`. `GEMINI_API_KEY` comes from
  `.env` too. `KOTHA_ASR_MODEL` defaults to `/app/models/bn-tugstugi-medium-ct2`.
- The app is healthy once `/v1/ready` answers; the health check allows 10 minutes for startup.
  `--wait` returns when it is healthy, or fails if it is not.
- Both ports are bound to `127.0.0.1` only.
- `docker compose config` prints `.env` values, including the key and the password; do not paste
  its output anywhere.
- Rebuild after changing the code or the lock file: `docker compose build app`.

## Latency evaluation

`eval/latency.py` measures warm calls end to end. It speaks every question of
`data/testsets/rag_test.csv` with the MMS voices (synthetic input: the TTS voice, not a person),
cuts each clip at its last audible sample, and streams it to the app in real time as 20 ms frames,
followed by silence, one call per question. One extra turn (the first question) runs first and is
not counted, because the first turn of a process may still pay first-use costs; its stages are
printed so any cold effect is visible. The percentiles count only turns that got an answer (`kind`
`answer`), were not split and did not fail. A fixed reply (handoff, "please repeat", account) is
served from the TTS cache, so its `tts_first_audio` is near zero, and some of those turns never
reach the LLM; mixing them in would hide the answer path. The cost: a slow LLM path that ends in a
handoff (`no_answer`, `rejected_twice`, an `llm_*` reason) is left out too, which biases `llm` and
`ttfa` low. The script prints the counted turns, the split turns, the reply kinds, the handoff
reasons and the failures (timeout, connection closed, connect failed, bad event, no audio), warns
when fewer than 30 turns are counted (the blueprint's minimum), and writes one row per turn to
`--output`, which must be outside the repository. Each row has the reply `kind`, its
`handoff_reason` and the verifier `rejections` (rule names joined by `;`) from the `timing` event;
MLflow gets one `handoffs.<reason>` count per reason. A failed turn is recorded and the run goes on.

```bash
cd ~/kotha && docker compose up -d --wait db && set -a && . ./.env && set +a \
  && KOTHA_ASR_MODEL=models/bn-tugstugi-medium-ct2 HF_HUB_OFFLINE=1 uv run python eval/latency.py \
    --launch --port 8766 --output ~/kotha-runs/latency/run.csv --mlflow-dir ~/kotha-runs/mlflow
```

- `--launch` starts the app itself and reports cold start (process start to `/v1/ready`) once,
  separately; without it the script waits for `/v1/ready` on `--port` and reports no cold start.
- Stages, all in seconds:
  - `vad`: true end of speech to the server's `speech_end` event; it includes the 700 ms hangover.
  - `asr`, `retrieval`, `llm`, `tts_first_audio`: measured in the server and sent in the turn's
    `timing` event. `llm` is the whole generation (summed over a regeneration), because MS1 does not
    stream tokens. `tts_first_audio` is from the reply text to its first synthesised sentence; a
    cached fixed reply makes it close to zero.
  - `ttfa`: true end of speech to the first byte of bot audio at the client (the blueprint's TTFA,
    measured from the true end of speech). On localhost, receiving is effectively the same moment
    as the server sending. `ttfa_from_vad` is the same from the `speech_end` event.
  - The true end of speech is the end of the last 20 ms frame of the clip on the paced timeline,
    after the clip was cut at its last sample above 1% of its peak.
  - `ttfa_all_turns`: `ttfa` over every turn that was heard, whatever the reply (answers,
    handoffs, "please repeat"), leaving out only split and failed turns: what a caller waits
    before hearing anything. The other rows count answers only.
- On the GPU, after the check above: add `KOTHA_ASR_DEVICE=cuda KOTHA_ASR_COMPUTE_TYPE=int8_float16`
  (ADR 0005's compute type) before `uv run`. `KOTHA_ASR_COMPUTE_TYPE` defaults to `int8`, the CPU
  setting; both are logged as run parameters.
- `--launch` refuses a port that is already in use, stops waiting if the app exits during
  startup, and kills the app if it does not exit within 60 s at the end.
- A number counts for `EVALUATION.md` only from a run with `git_dirty` false. Timings on cuda count
  only after the GPU check in this runbook passed in the same session; `KOTHA_ASR_DEVICE` defaults
  to `cpu`.
- Gemini's free tier limits requests per minute; the run makes one call at a time, so it stays
  well under that.

## Text chat and scenarios

Both commands run the real answer pipeline (ADR 0008): hybrid retrieval over the database, bge-m3
on CPU and Gemini. They need the database up and ingested, as for the retrieval evaluation, and
`GEMINI_API_KEY` in `.env`. Everything typed goes to the Gemini free tier, so type only made-up
test questions, never a real caller's words.

```bash
cd ~/kotha && docker compose up -d --wait db && set -a && . ./.env && set +a \
  && HF_HUB_OFFLINE=1 uv run python -m kotha.dialogue.chat
```

Type a question after `you>`; Ctrl-D ends the chat. Each reply shows its kind (`answer`,
`account_unavailable`, `handoff: <reason>` or `repeat`) and the cited chunks.

The scenario runner replays the scripted conversations in `data/scenarios/text_scenarios.json`
and checks each reply's kind, language and, for answers, the cited document. In MS1 the pipeline
keeps no state between turns, so every turn must make sense on its own. It prints every turn and
exits with status 1 if any turn fails:

```bash
cd ~/kotha && set -a && . ./.env && set +a \
  && HF_HUB_OFFLINE=1 uv run python -m kotha.dialogue.scenarios > ~/kotha-runs/scenarios.txt
```

- The output holds the bot's answers, so keep it outside the repository.
- Gemini's answers vary between runs; a failing turn is worth reading before changing anything.
  Change a scenario only when its expectation is wrong, never to make a run pass.

## Checking a manifest

`eval/validate_manifest.py` checks one or more manifests and their audio before any
evaluation run, and prints a summary of the set. For the public manifest alone:

```bash
cd ~/kotha && uv run python eval/validate_manifest.py --manifest data/testsets/asr_test.csv \
  --data-dir /mnt/g/Kotha/data/testsets/asr
```

With the private manifest that sits next to the audio on G: (ADR 0002), give both; ids and
paths must then be unique across the two:

```bash
cd ~/kotha && uv run python eval/validate_manifest.py \
  --manifest data/testsets/asr_test.csv /mnt/g/Kotha/data/testsets/asr/asr_private.csv \
  --data-dir /mnt/g/Kotha/data/testsets/asr
```

- Errors are printed as `<manifest>:<line> <id>: <problem>`: a value outside the allowed
  `lang`, `noise`, `split` or speaker code, an empty id, text or path, a repeated id or path,
  a path that is absolute, contains `..` or leads outside `--data-dir`, a missing file, a file
  soundfile can't read (a real AAC `.m4a` can't be read; convert it to WAV or FLAC first), and
  a duration outside `--min-s` and `--max-s` (default 0.5 to 30 s, both ends allowed).
- Warnings don't fail the check: a container that doesn't match the file extension (for
  example MP3 data in a `.m4a` file), a sample rate below 16000 Hz, more than one channel,
  and more than one codec or sample rate in the whole set.
- The summary gives rows, speakers and minutes, a lang x noise table, clips and minutes per
  speaker and per split, and the codecs and sample rates seen. Minutes count only files that
  soundfile can read, and values outside the allowed ones are grouped as `other`.
- The exit code is 1 if there is any error, else 0.
- The output never contains reference text or the invalid values themselves; open the
  manifest at the reported line to see them.

## Checking references

`eval/lint_references.py` checks the `text` column against `data/testsets/GUIDELINE.md`, so
that references are already in the form the text normaliser produces. For the public
manifest alone, and with the private one:

```bash
cd ~/kotha && uv run python eval/lint_references.py --manifest data/testsets/asr_test.csv
cd ~/kotha && uv run python eval/lint_references.py \
  --manifest data/testsets/asr_test.csv /mnt/g/Kotha/data/testsets/asr/asr_private.csv
```

Problems are printed as `<manifest>:<line> <id>: <problem>`. Errors:

- `empty reference`.
- `not normalised at position N: U+XXXX NAME (kind)` (rule 1): the reference differs from
  `normalize(reference)`, first at code point N (counted from 1; an editor may count Bangla
  characters differently). The kind is punctuation, Bangla digit, uppercase letter, extra or
  odd whitespace, invisible character, or not NFC. A precomposed য় (U+09DF), ড় or ঢ় is not
  NFC; type the letter and its nukta (়) as two code points, as a Unicode keyboard does.
  Only the first difference is shown; run the lint again after fixing it.
- `taka sign ৳, write the word টাকা (rule 3)`.
- `lang bn contains Latin letters`, `lang en contains Bengali letters`, `lang mix needs both
  Bengali and Latin letters` (rule 7; digits don't count).
- `word N: loanword in Bengali script, write it as '<latin>' (rule 2)`: the word is a
  spelling in the loanword lexicon (`--lexicon`, default `data/lexicon/loanwords.csv`).

Warnings, which don't change the exit code:

- `word N: number and unit joined (rule 3)`, for example `30gb`; digits alone are fine.
- `word N: Latin word with a Bangla suffix attached ... (rule 4)`, for example `packageটা`.
- `word N: filler word, leave it out (rule 5)`: আ, উম, এ, uh, um or hmm as a whole word.
  এ is also an ordinary Bangla word, so check the audio before removing it.

Words are counted in the normalised reference. The last line is `N errors, M warnings`, and
the exit code is 1 if there is any error. The output never contains reference text, so open
the manifest at the reported line to see the problem.

## WER evaluation

`eval/wer.py` transcribes one split of a manifest with `WhisperAsr`, using the decoding
settings of ADR 0005, and scores it with the text normaliser. The manifest will be committed at
`data/testsets/asr_test.csv` (M1-6), and the audio stays on G:. The manifest's `path` column is
relative to the audio folder given as `--data-dir`.

A dev run on the GPU, at 16 kHz and through simulated telephony:

```bash
cd ~/kotha && uv run python eval/wer.py --manifest data/testsets/asr_test.csv \
  --data-dir /mnt/g/Kotha/data/testsets/asr --model models/bn-tugstugi-medium-ct2 \
  --device cuda --compute-type int8_float16 --condition 16k 8k --split dev \
  --output /mnt/g/Kotha/data/eval/dev-tugstugi-medium-cuda.csv
```

The same run on the CPU:

```bash
cd ~/kotha && uv run python eval/wer.py --manifest data/testsets/asr_test.csv \
  --data-dir /mnt/g/Kotha/data/testsets/asr --model models/bn-tugstugi-medium-ct2 \
  --device cpu --compute-type int8 --condition 16k 8k --split dev \
  --output /mnt/g/Kotha/data/eval/dev-tugstugi-medium-cpu.csv
```

- `--language bn` (the default) forces Bangla for every row; `--language auto` lets Whisper
  detect the language. Report which one a result used.
- The script runs one untimed transcription first, so the latencies leave out model warm-up.
  They time only the transcription, not loading the audio or simulating telephony.
- Latency from a cuda run counts only if the GPU check (Local GPU setup) passed in the same
  session. Every report of latency states the device, the compute type and the GPU state.
- It prints the run settings, then a table of WER, CER and latency p50 and p95 per condition
  and language, with an `all` row per condition. CER counts Unicode code points including
  spaces; say so wherever a CER is reported.
- `--output` is a per-utterance CSV with the normalised reference and hypothesis. It holds
  transcripts of consented test audio, so write it outside the repository (for example under
  `/mnt/g/Kotha/data/eval/`) and never commit it. The script refuses an output path inside
  the repository and creates missing folders.
- Only numbers from a committed evaluation run go into `EVALUATION.md`.
- Rows of speakers who said No to publishing their text are kept in a private manifest next to
  the audio on G: and are never committed. A full evaluation passes that private manifest to
  `--manifest` (or a merged copy kept outside the repository), and `EVALUATION.md` states how
  many rows are private (ADR 0002).

### Top substitutions

Next to `--output`, the script writes `<output stem>.substitutions.csv` (columns `condition`,
`ref`, `hyp`, `count`, `category`): the 20 most frequent substituted word pairs per condition,
taken from the normalised text before the loanword lexicon is applied. After the table it
prints only how many of those pairs fall in each category, for example
`16k substitutions: loanword-script <n>, script <n>, number <n>, other <n>`; it never prints
words.
Each pair gets the first category that matches:

- `loanword-script`: the reference has a Latin word from the lexicon and the hypothesis one of
  its listed Bengali spellings. These are the errors `lexicon_wer` already forgives, so a high
  count here is about spelling convention, not recognition.
- `number`: either word contains a digit, for example `30` against `তিরিশ`.
- `script`: one word is in Latin and the other in Bengali script, but the pair isn't in the
  lexicon.
- `other`: everything else, mostly real recognition errors.

The file holds words from the transcripts, so like `--output` it stays outside the repository
and follows the retention rules of ADR 0002.

### Tracking runs with MLflow

With `--mlflow-dir DIR`, the script also logs the run to a local MLflow store: a SQLite file
`DIR/mlflow.db` and artifacts under `DIR/artifacts`, in the experiment `kotha-asr`. It logs
the run settings as params, every WER, CER, lexicon score and latency per condition and
language as metrics (named like `16k.mix.wer`), the git commit and `git_dirty` as tags, and
both CSVs as artifacts, then prints `mlflow run: <run id>`. The model, lexicon and manifest are
logged by name only, never as full paths. The script switches MLflow's usage telemetry off and
keeps its logs at WARNING unless `MLFLOW_LOGGING_LEVEL` is already set.

```bash
cd ~/kotha && uv run python eval/wer.py --manifest data/testsets/asr_test.csv \
  --data-dir /mnt/g/Kotha/data/testsets/asr --model models/bn-tugstugi-medium-ct2 \
  --device cuda --compute-type int8_float16 --condition 16k 8k --split dev \
  --output /mnt/g/Kotha/data/eval/dev-tugstugi-medium-cuda.csv \
  --mlflow-dir ~/kotha-runs/mlflow
```

- Keep the store in the WSL home, for example `~/kotha-runs/mlflow`, not on `/mnt/g`: SQLite's
  file locking on Windows drives under WSL is unreliable. This is a precaution, not something
  measured here. The script refuses a `--mlflow-dir` inside the repository.
- The git state is read before transcription starts, so the tags describe the code that ran,
  and a git failure stops the run early. Untracked files in the repository also make
  `git_dirty` true.
- A number counts for `EVALUATION.md` only if its run has `git_dirty` false.
- The artifacts contain transcripts, so the store follows the retention rules of ADR 0002 like
  the CSVs themselves.

To browse the runs, start the UI on this machine only. The first start takes a while, because
uv downloads full MLflow into a temporary environment; `pyproject.toml` stays unchanged:

```bash
cd ~/kotha && MLFLOW_DISABLE_TELEMETRY=true uv run --with mlflow mlflow ui \
  --backend-store-uri sqlite:///$HOME/kotha-runs/mlflow/mlflow.db --host 127.0.0.1
```

`--host 127.0.0.1` keeps the UI unreachable from the network, because the store holds
transcripts.

### Loanword lexicon and `lexicon_wer`

The references write English loanwords in Latin script ("mobile balance"), while the Bangla
fine-tunes write them in Bengali script ("মোবাইল ব্যালেন্স"). Strict WER counts every such
word as an error, even when the model heard it right.

- `wer` and `cer` are the primary scores. `lexicon_wer` and `lexicon_cer` are a secondary,
  loanword-agnostic score, and must always be reported with that label, never as "the WER".
- For the secondary score, both the reference and the hypothesis are normalised as usual, and
  then every whole word listed in the lexicon is replaced by its Latin form. With an empty
  lexicon the two scores are equal. `lexicon_cer` counts the code points of the mapped text.
- The lexicon is `data/lexicon/loanwords.csv` (columns `latin,bangla`, one row per accepted
  Bengali spelling), passed with `--lexicon` (default: that file). It is part of the
  transcription guideline (K2), which owns it; the text normaliser itself never
  transliterates.
- Only list loanwords. An ordinary Bangla word that also stands for an English one (such as
  নম্বর for "number") must not be listed, because mapping it would hide real errors. কল and
  সিম are also ordinary Bangla words, kept for the telecom domain.
- The lexicon replaces whole words only, so these still count as errors: attached suffixes
  ("প্যাকেজটা"), number words ("থার্টি" for 30), and spellings not in the list.
