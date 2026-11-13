import argparse
import gc
import statistics
import subprocess
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time
from typing import IO, TypeVar

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim

from kotha.asr.whisper import DECODING_OPTIONS
from kotha.audio.convert import load_mono_16k

STAGES = ("features", "encoder", "whole")
REASONS = (
    "gpu_idle",
    "sw_power_cap",
    "hw_slowdown",
    "hw_thermal_slowdown",
    "sw_thermal_slowdown",
    "hw_power_brake_slowdown",
)
SAMPLE_FIELDS = (
    "pstate",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "clocks.max.mem",
    "power.draw",
    "utilization.gpu",
    "temperature.gpu",
    *(f"clocks_event_reasons.{reason}" for reason in REASONS),
)
COLUMNS = (
    "compute_type",
    "stage",
    "runs",
    "wall_median_ms",
    "wall_max_ms",
    "wall_ms_runs",
    "cpu_median_ms",
    "cpu_wall_ratio",
    "tokens",
    "decode_ms_median",
    "decode_ms_per_token",
    "mem_before_load_mib",
    "samples",
    "pstates",
    "sm_mhz_median",
    "sm_mhz_max",
    "sm_mhz_limit",
    "mem_mhz_median",
    "mem_mhz_max",
    "mem_mhz_limit",
    "power_w_median",
    "power_w_max",
    "util_pct_median",
    "temp_c_max",
    "active_reasons",
    "sm_mhz_runs",
    "mem_mhz_runs",
    "power_w_runs",
    "temp_c_runs",
    "reasons_runs",
)

T = TypeVar("T")


@dataclass(frozen=True)
class GpuSample:
    at: float
    pstate: str | None
    sm_mhz: float | None
    sm_mhz_limit: float | None
    mem_mhz: float | None
    mem_mhz_limit: float | None
    power_w: float | None
    util_pct: float | None
    temp_c: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StageRun:
    start: float
    end: float
    wall_ms: float
    cpu_ms: float


@dataclass(frozen=True)
class GpuSummary:
    samples: int
    pstates: tuple[str, ...]
    sm_mhz_median: float | None
    sm_mhz_max: float | None
    sm_mhz_limit: float | None
    mem_mhz_median: float | None
    mem_mhz_max: float | None
    mem_mhz_limit: float | None
    power_w_median: float | None
    power_w_max: float | None
    util_pct_median: float | None
    temp_c_max: float | None
    reasons: tuple[str, ...]


def number(value: str) -> float | None:
    return None if value.strip() in ("[N/A]", "N/A", "") else float(value)


def parse_sample(line: str, at: float) -> GpuSample:
    values = [value.strip() for value in line.split(",")]
    pstate, sm_mhz, sm_mhz_limit, mem_mhz, mem_mhz_limit, power_w, util_pct, temp_c, *flags = values
    return GpuSample(
        at=at,
        pstate=None if pstate in ("[N/A]", "N/A") else pstate,
        sm_mhz=number(sm_mhz),
        sm_mhz_limit=number(sm_mhz_limit),
        mem_mhz=number(mem_mhz),
        mem_mhz_limit=number(mem_mhz_limit),
        power_w=number(power_w),
        util_pct=number(util_pct),
        temp_c=number(temp_c),
        reasons=tuple(
            reason for reason, flag in zip(REASONS, flags, strict=True) if flag == "Active"
        ),
    )


def read_samples(stream: IO[str], samples: list[GpuSample]) -> None:
    for line in stream:
        if line.strip():
            samples.append(parse_sample(line, perf_counter()))


@contextmanager
def gpu_sampler(sample_ms: int) -> Iterator[list[GpuSample]]:
    samples: list[GpuSample] = []
    if sample_ms <= 0:
        yield samples
        return
    process = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(SAMPLE_FIELDS)}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(sample_ms),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    reader = threading.Thread(target=read_samples, args=(process.stdout, samples), daemon=True)
    reader.start()
    try:
        yield samples
    finally:
        process.terminate()
        process.wait(timeout=10)
        reader.join(timeout=10)


def gpu_memory_mib() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    used, total = (value.strip() for value in completed.stdout.splitlines()[0].split(","))
    return f"{used}/{total}"


def samples_between(samples: Sequence[GpuSample], start: float, end: float) -> list[GpuSample]:
    return [sample for sample in samples if start <= sample.at <= end]


def median_or_none(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def max_or_none(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def summarise_samples(samples: Sequence[GpuSample]) -> GpuSummary | None:
    if not samples:
        return None
    return GpuSummary(
        samples=len(samples),
        pstates=tuple(sorted({sample.pstate for sample in samples if sample.pstate})),
        sm_mhz_median=median_or_none([sample.sm_mhz for sample in samples]),
        sm_mhz_max=max_or_none([sample.sm_mhz for sample in samples]),
        sm_mhz_limit=max_or_none([sample.sm_mhz_limit for sample in samples]),
        mem_mhz_median=median_or_none([sample.mem_mhz for sample in samples]),
        mem_mhz_max=max_or_none([sample.mem_mhz for sample in samples]),
        mem_mhz_limit=max_or_none([sample.mem_mhz_limit for sample in samples]),
        power_w_median=median_or_none([sample.power_w for sample in samples]),
        power_w_max=max_or_none([sample.power_w for sample in samples]),
        util_pct_median=median_or_none([sample.util_pct for sample in samples]),
        temp_c_max=max_or_none([sample.temp_c for sample in samples]),
        reasons=tuple(reason for reason in REASONS if any(reason in s.reasons for s in samples)),
    )


def decode_ms(whole_ms: float, encoder_ms: float, features_ms: float) -> float:
    return whole_ms - encoder_ms - features_ms


def decode_ms_per_token(decode: float, tokens: int) -> float | None:
    return decode / tokens if tokens else None


def timed(action: Callable[[], T]) -> tuple[T, StageRun]:
    wall_start = perf_counter()
    cpu_start = process_time()
    value = action()
    cpu_ms = (process_time() - cpu_start) * 1000
    wall_end = perf_counter()
    return value, StageRun(wall_start, wall_end, (wall_end - wall_start) * 1000, cpu_ms)


def transcribe_options() -> dict:
    return {
        "language": "bn",
        "beam_size": 1,
        "initial_prompt": None,
        "word_timestamps": False,
        **DECODING_OPTIONS,
    }


def transcribe_tokens(model: WhisperModel, audio_16k: np.ndarray) -> int:
    segments, _ = model.transcribe(audio_16k, **transcribe_options())
    return sum(len(segment.tokens) for segment in segments)


def run_stages(
    model: WhisperModel, audio_16k: np.ndarray, stages: Sequence[str], runs: int
) -> tuple[dict[str, list[StageRun]], list[int]]:
    window = pad_or_trim(
        model.feature_extractor(audio_16k)[:, : model.feature_extractor.nb_max_frames]
    )
    timings: dict[str, list[StageRun]] = {stage: [] for stage in stages}
    tokens: list[int] = []
    for _ in range(runs):
        if "features" in stages:
            _, run = timed(lambda: model.feature_extractor(audio_16k))
            timings["features"].append(run)
        if "encoder" in stages:
            _, run = timed(lambda: model.encode(window))
            timings["encoder"].append(run)
        if "whole" in stages:
            run_tokens, run = timed(lambda: transcribe_tokens(model, audio_16k))
            timings["whole"].append(run)
            tokens.append(run_tokens)
    return timings, tokens


def cell(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def run_reasons(samples: Sequence[GpuSample]) -> str:
    if not samples:
        return "-"
    summary = summarise_samples(samples)
    return "+".join(summary.reasons) if summary and summary.reasons else "none"


def per_run_cells(runs: Sequence[StageRun], samples: Sequence[GpuSample]) -> list[str]:
    windows = [samples_between(samples, run.start, run.end) for run in runs]
    return [
        ",".join(cell(median_or_none([s.sm_mhz for s in window])) for window in windows),
        ",".join(cell(median_or_none([s.mem_mhz for s in window])) for window in windows),
        ",".join(cell(median_or_none([s.power_w for s in window]), 1) for window in windows),
        ",".join(cell(max_or_none([s.temp_c for s in window])) for window in windows),
        ",".join(run_reasons(window) for window in windows),
    ]


def summary_cells(summary: GpuSummary | None) -> list[str]:
    if summary is None:
        return ["0"] + ["-"] * 12
    return [
        str(summary.samples),
        ",".join(summary.pstates) or "-",
        cell(summary.sm_mhz_median),
        cell(summary.sm_mhz_max),
        cell(summary.sm_mhz_limit),
        cell(summary.mem_mhz_median),
        cell(summary.mem_mhz_max),
        cell(summary.mem_mhz_limit),
        cell(summary.power_w_median, 1),
        cell(summary.power_w_max, 1),
        cell(summary.util_pct_median),
        cell(summary.temp_c_max),
        "+".join(summary.reasons) or "none",
    ]


def decode_cells(tokens: Sequence[int], decodes: Sequence[float]) -> list[str]:
    if not decodes:
        return [cell(statistics.median(tokens)) if tokens else "-", "-", "-"]
    per_token = [
        decode_ms_per_token(decode, count) for decode, count in zip(decodes, tokens, strict=True)
    ]
    return [
        cell(statistics.median(tokens)),
        cell(statistics.median(decodes), 1),
        cell(median_or_none(per_token), 2),
    ]


def format_row(
    compute_type: str,
    stage: str,
    runs: Sequence[StageRun],
    samples: Sequence[GpuSample],
    memory: str | None,
    tokens: Sequence[int] = (),
    decodes: Sequence[float] = (),
) -> str:
    wall = [run.wall_ms for run in runs]
    cpu_median = statistics.median(run.cpu_ms for run in runs)
    wall_median = statistics.median(wall)
    if samples:
        stage_samples = [s for run in runs for s in samples_between(samples, run.start, run.end)]
        gpu_cells = summary_cells(summarise_samples(stage_samples)) + per_run_cells(runs, samples)
    else:
        gpu_cells = ["-"] * (len(COLUMNS) - COLUMNS.index("samples"))
    return "\t".join(
        [
            compute_type,
            stage,
            str(len(runs)),
            cell(wall_median, 1),
            cell(max(wall), 1),
            ",".join(cell(value) for value in wall),
            cell(cpu_median, 1),
            cell(cpu_median / wall_median, 2),
            *decode_cells(tokens, decodes),
            memory or "-",
            *gpu_cells,
        ]
    )


def bench_compute_type(
    args: argparse.Namespace, compute_type: str, audio_16k: np.ndarray
) -> list[str]:
    on_cuda = args.device == "cuda"
    memory = gpu_memory_mib() if on_cuda else None
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    transcribe_tokens(model, audio_16k)
    with gpu_sampler(args.sample_ms if on_cuda else 0) as samples:
        timings, tokens = run_stages(model, audio_16k, args.stages, args.runs)
    del model
    gc.collect()
    decodes = (
        [
            decode_ms(whole.wall_ms, encoder.wall_ms, features.wall_ms)
            for whole, encoder, features in zip(
                timings["whole"], timings["encoder"], timings["features"], strict=True
            )
        ]
        if set(STAGES) <= set(args.stages)
        else []
    )
    return [
        format_row(
            compute_type,
            stage,
            timings[stage],
            samples,
            memory,
            tokens if stage == "whole" else (),
            decodes if stage == "whole" else (),
        )
        for stage in args.stages
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-types", nargs="+", required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    args.stages = [stage for stage in STAGES if stage in args.stages]
    audio_16k = load_mono_16k(args.clip)
    print("\t".join(COLUMNS))
    for compute_type in args.compute_types:
        for row in bench_compute_type(args, compute_type, audio_16k):
            print(row, flush=True)


if __name__ == "__main__":
    main()
