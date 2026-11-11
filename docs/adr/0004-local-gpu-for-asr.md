# 0004: Run faster-whisper on the local GPU with cuBLAS from a uv dependency group
Status: proposed · Date: 2026-11-11

## Context
The live path should run ASR on the local GTX 1650 (4 GB, compute capability 7.5, Windows
driver under WSL2). CTranslate2 4.9.0 already sees the GPU, but it loads `libcublas.so.12` by
name at runtime, so transcribing on `cuda` fails with "Library libcublas.so.12 is not found"
until cuBLAS for CUDA 12 is installed and its directory is on `LD_LIBRARY_PATH` before Python
starts. CI has no GPU and must not install the NVIDIA packages.

## Decision
Add a `gpu` dependency group with `nvidia-cublas-cu12` for Linux only, and make it a default
group (`[tool.uv] default-groups = ["dev", "gpu"]`) so a plain `uv sync` keeps it locally. CI
sets `UV_NO_GROUP: gpu` for the whole job, because `uv run` also syncs default groups.
`LD_LIBRARY_PATH` is exported from `~/.bashrc` after its interactive check, which the Ubuntu
terminal, VS Code terminals and VS Code's extension host all read (see `docs/runbook.md`).

cuDNN is not installed: CTranslate2 4.9.0 contains no reference to cuDNN, and after
transcribing on `cuda` only `libcublas.so.12`, `libcublasLt.so.12` and `libcuda.so.1` are
loaded. This must be rechecked on any CTranslate2 upgrade. cuBLAS 12.9 itself depends on
`nvidia-cuda-nvrtc-cu12`, which comes with it.

## Alternatives considered
- CUDA toolkit inside WSL: a system-wide apt install outside `uv.lock`, much larger than
  cuBLAS alone, and it must be the WSL-specific package so no Linux driver replaces the
  Windows one, which already provides `libcuda`.
- An NVIDIA CUDA Docker image now: heavier than needed before the services are containerised.
- CPU only: works and stays the default, but leaves the GPU unused for the live path.
- `uv sync --group gpu` with no default group: no CI change, but every plain `uv sync` would
  remove the GPU libraries.

## Consequences
- Locally, `uv sync` and `uv run` keep cuBLAS; CI never installs it.
- A sync with `--no-default-groups`, `--no-group gpu` or `UV_NO_GROUP` removes it; the runbook
  says how to restore it.
- `LD_LIBRARY_PATH` points into `~/kotha/.venv`, so moving the repo means updating the line.
- On this GPU, small and medium both load and run with float16, int8_float16 and int8; none
  of these failed or ran out of memory. Which one the live path uses is decided in M3-6.
