import argparse
import gc
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kotha.asr.whisper import WhisperAsr
from kotha.audio.convert import load_mono_16k

DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "speech_16k.wav"


@dataclass(frozen=True)
class ComputeTypeResult:
    compute_type: str
    ok: bool
    load_s: float | None
    median_s: float | None
    repeatable: bool | None
    gpu_mem_mb: int | None
    error: str


def gpu_memory_used_mb() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return int(completed.stdout.splitlines()[0])


def check_compute_type(
    audio_16k: np.ndarray,
    model: str,
    device: str,
    compute_type: str,
    runs: int,
    read_gpu_memory: Callable[[], int | None],
) -> ComputeTypeResult:
    # An unsupported compute type or an out-of-memory error is a result, not a crash.
    try:
        started = time.perf_counter()
        asr = WhisperAsr(model, device=device, compute_type=compute_type)
        load_s = time.perf_counter() - started
        gpu_mem_mb = read_gpu_memory()
        asr.transcribe(audio_16k, language="bn")
        transcripts = []
        durations = []
        for _ in range(runs):
            started = time.perf_counter()
            transcripts.append(asr.transcribe(audio_16k, language="bn"))
            durations.append(time.perf_counter() - started)
        del asr
    except Exception as error:
        message = " ".join(str(error).split())
        return ComputeTypeResult(
            compute_type, False, None, None, None, None, f"{type(error).__name__}: {message}"
        )
    return ComputeTypeResult(
        compute_type=compute_type,
        ok=all(transcript.text for transcript in transcripts),
        load_s=load_s,
        median_s=statistics.median(durations),
        repeatable=all(transcript == transcripts[0] for transcript in transcripts),
        gpu_mem_mb=gpu_mem_mb,
        error="",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, nargs="?", default=DEFAULT_AUDIO)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-types", default="float16,int8_float16,int8")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    audio_16k = load_mono_16k(args.audio)
    read_gpu_memory = gpu_memory_used_mb if args.device == "cuda" else lambda: None
    print("compute_type\tok\tload_s\tmedian_s\trepeatable\tgpu_mem_mb\terror")
    for compute_type in args.compute_types.split(","):
        result = check_compute_type(
            audio_16k, args.model, args.device, compute_type, args.runs, read_gpu_memory
        )
        gc.collect()
        cells = [
            result.compute_type,
            "yes" if result.ok else "no",
            "-" if result.load_s is None else f"{result.load_s:.2f}",
            "-" if result.median_s is None else f"{result.median_s:.2f}",
            "-" if result.repeatable is None else ("yes" if result.repeatable else "no"),
            "-" if result.gpu_mem_mb is None else str(result.gpu_mem_mb),
            result.error or "-",
        ]
        print("\t".join(cells))


if __name__ == "__main__":
    main()
