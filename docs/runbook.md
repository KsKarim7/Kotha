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

The conversion uses transformers 5.9.0, the version CTranslate2 4.9.0's own tests pin. It
runs in a throwaway environment (`uv run --with`), so `pyproject.toml` and `uv.lock` don't
change.

Download each source at its pinned SHA first. `ct2-transformers-converter --revision` pins
the weights but not the files that `--copy_files` fetches, which come from `main`.
`training_args.bin` and the training logs are not downloaded.

```bash
cd ~/kotha && uv run --with transformers==5.9.0 python - <<'PY'
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
uv run --with transformers==5.9.0 python -c 'from transformers import WhisperTokenizerFast; WhisperTokenizerFast.from_pretrained("models/src/bn-arif11-v7-small").save_pretrained("models/src/bn-arif11-v7-small-tokenizer")'
cp models/src/bn-arif11-v7-small-tokenizer/tokenizer.json models/src/bn-arif11-v7-small/
```

Its weights are a pickle (`pytorch_model.bin`). transformers 5.9.0 loads them with
`torch.load(..., weights_only=True)`, which refuses arbitrary Python objects; never turn that
off.

Convert both:

```bash
for name in bn-tugstugi-medium bn-arif11-v7-small; do
  uv run --with transformers==5.9.0 ct2-transformers-converter --model "models/src/$name" \
    --output_dir "models/$name-ct2" --copy_files tokenizer.json preprocessor_config.json \
    --quantization float16
done
```

## WER evaluation

`eval/wer.py` transcribes one split of a manifest with `WhisperAsr`, using the decoding
settings of ADR 0005, and scores it with the text normaliser. The manifest is committed at
`data/testsets/asr_test.csv`, and the audio stays on G:. The manifest's `path` column is
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
- It prints the run settings, then a table of WER, CER and latency p50 and p95 per condition
  and language, with an `all` row per condition. CER counts Unicode code points including
  spaces; say so wherever a CER is reported.
- `--output` is a per-utterance CSV with the normalised reference and hypothesis. It holds
  transcripts of consented test audio, so write it outside the repository (for example under
  `/mnt/g/Kotha/data/eval/`) and never commit it. The script refuses an output path inside
  the repository and creates missing folders.
- Only numbers from a committed evaluation run go into `EVALUATION.md`.

### Loanword lexicon and `lexicon_wer`

The references write English loanwords in Latin script ("mobile balance"), while the Bangla
fine-tunes write them in Bengali script ("মোবাইল ব্যালেন্স"). Strict WER counts every such
word as an error, even when the model heard it right.

- `wer` and `cer` are the primary scores. `lexicon_wer` and `lexicon_cer` are a secondary,
  loanword-agnostic score, and must always be reported with that label, never as "the WER".
- For the secondary score, both the reference and the hypothesis are normalised as usual, and
  then every whole word listed in the lexicon is replaced by its Latin form. With an empty
  lexicon the two scores are equal. `lexicon_cer` counts the code points of the mapped text.
- The lexicon is `data/testsets/loanwords.csv` (columns `latin,bangla`, one row per accepted
  Bengali spelling), passed with `--lexicon` (default: that file). It is part of the
  transcription guideline (K2), which owns it; the text normaliser itself never
  transliterates.
- Only list loanwords. An ordinary Bangla word that also stands for an English one (such as
  নম্বর for "number") must not be listed, because mapping it would hide real errors. কল and
  সিম are also ordinary Bangla words, kept for the telecom domain.
- The lexicon replaces whole words only, so these still count as errors: attached suffixes
  ("প্যাকেজটা"), number words ("থার্টি" for 30), and spellings not in the list.
