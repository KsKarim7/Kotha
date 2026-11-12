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
