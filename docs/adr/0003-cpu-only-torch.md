# 0003: Install CPU-only torch from the PyTorch index
Status: accepted · Date: 2026-11-11

## Context
Silero VAD needs torch: the `silero-vad` 6.2.3 package depends on it. On Linux, PyPI's torch
2.15.0 is a CUDA build that pulls in cuda-toolkit, triton and the nvidia-*-cu13 packages, about
3.0 GB of wheels for Python 3.11 on x86-64. VAD and MMS-TTS run on CPU in our plan, ASR uses
CTranslate2 without torch, and fine-tuning runs on Kaggle or Colab.

## Decision
Take torch from the explicit `pytorch-cpu` index (`https://download.pytorch.org/whl/cpu`)
through `[tool.uv.sources]`, and list torch as a direct dependency: uv applied the source only
once torch was direct; as a transitive dependency of silero-vad it still resolved from PyPI.

## Alternatives considered
- PyPI torch with CUDA: about 3.0 GB of wheels, and the 4 GB GPU is reserved for ASR.
- onnxruntime only: the silero-vad package still requires torch, and MMS-TTS needs torch anyway.
- A torch-free fork of Silero VAD from a third party: it would not be maintained by Silero.

## Consequences
- The torch wheel is 209.5 MB instead of about 3.0 GB for the CUDA set, so local installs and
  CI download less. The local `.venv` grew from 330 MB to 1.2 GB.
- torch must stay a direct dependency, or the CPU source stops applying.
- GPU torch would need a deliberate change to the source entry later.
- torchvision or torchaudio would each need the same source entry if ever added.
- The lock has two torch entries from the CPU index: `2.15.0+cpu` for Linux and Windows, and
  `2.15.0` for macOS.
