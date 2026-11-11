# Runbook

## Local GPU setup

faster-whisper runs on the GPU through CTranslate2, which loads cuBLAS for CUDA 12
(`libcublas.so.12`) at runtime. uv installs it with the `gpu` dependency group, which is a
default group locally and is turned off in CI (`UV_NO_GROUP: gpu`). The Windows NVIDIA
driver provides the rest: never install a driver or the CUDA toolkit inside Ubuntu.

One-time setup, in an Ubuntu terminal:

```bash
cd ~/kotha && uv sync
grep -qF 'nvidia/cublas/lib' ~/.bashrc || echo 'export LD_LIBRARY_PATH="$HOME/kotha/.venv/lib/python3.11/site-packages/nvidia/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> ~/.bashrc
```

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
