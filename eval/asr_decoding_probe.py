import argparse
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from faster_whisper import WhisperModel

from kotha.audio.convert import load_mono_16k

FIXED_OPTIONS = {
    "language": "bn",
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "vad_filter": False,
}

CONFIGS: dict[str, dict] = {
    "current": {"beam_size": 1},
    "t0": {"beam_size": 1, "temperature": 0.0},
    "beam5_t0": {"beam_size": 5, "temperature": 0.0},
    "t0_rep": {"beam_size": 1, "temperature": 0.0, "repetition_penalty": 1.1},
    "t0_ngram": {"beam_size": 1, "temperature": 0.0, "no_repeat_ngram_size": 3},
    "t0_cap": {"beam_size": 1, "temperature": 0.0, "max_new_tokens": None},
    "beam5_cap": {"beam_size": 5, "temperature": 0.0, "max_new_tokens": None},
}

COLUMNS = (
    "clip",
    "device",
    "config",
    "runs",
    "median_s",
    "max_s",
    "repeatable",
    "max_tokens",
    "max_compression_ratio",
    "temperatures",
    "worst_avg_logprob",
    "max_no_speech_prob",
    "empty",
    "bengali_share",
    "text",
)


@dataclass(frozen=True)
class RunStats:
    seconds: float
    text: str
    max_tokens: int
    max_compression_ratio: float
    temperatures: tuple[float, ...]
    worst_avg_logprob: float
    max_no_speech_prob: float
    empty: bool


@dataclass(frozen=True)
class ProbeRow:
    clip: str
    device: str
    config: str
    runs: int
    median_s: float
    max_s: float
    repeatable: bool | None
    max_tokens: int
    max_compression_ratio: float
    temperatures: tuple[float, ...]
    worst_avg_logprob: float
    max_no_speech_prob: float
    empty: bool
    bengali_share: float | None
    text: str


def run_once(model: WhisperModel, audio_16k: np.ndarray, options: dict) -> RunStats:
    started = perf_counter()
    segment_stream, _ = model.transcribe(audio_16k, **FIXED_OPTIONS, **options)
    segments = list(segment_stream)
    seconds = perf_counter() - started
    # Same joining rule as kotha.asr.whisper.WhisperAsr, so the texts are comparable.
    segment_texts = (segment.text.strip() for segment in segments)
    text = " ".join(text for text in segment_texts if text)
    if not segments:
        return RunStats(seconds, "", 0, 0.0, (), float("-inf"), 1.0, True)
    return RunStats(
        seconds=seconds,
        text=text,
        max_tokens=max(len(segment.tokens) for segment in segments),
        max_compression_ratio=max(segment.compression_ratio for segment in segments),
        temperatures=tuple(sorted({segment.temperature for segment in segments})),
        worst_avg_logprob=min(segment.avg_logprob for segment in segments),
        max_no_speech_prob=max(segment.no_speech_prob for segment in segments),
        empty=not text,
    )


def bengali_share(text: str) -> float | None:
    letters = [char for char in text if unicodedata.category(char)[0] in "LM"]
    if not letters:
        return None
    return sum("\u0980" <= char <= "\u09ff" for char in letters) / len(letters)


def summarize(clip: str, device: str, config: str, runs: list[RunStats]) -> ProbeRow:
    seconds = [run.seconds for run in runs]
    return ProbeRow(
        clip=clip,
        device=device,
        config=config,
        runs=len(runs),
        median_s=statistics.median(seconds),
        max_s=max(seconds),
        repeatable=all(run.text == runs[0].text for run in runs) if len(runs) > 1 else None,
        max_tokens=max(run.max_tokens for run in runs),
        max_compression_ratio=max(run.max_compression_ratio for run in runs),
        temperatures=tuple(sorted({t for run in runs for t in run.temperatures})),
        worst_avg_logprob=min(run.worst_avg_logprob for run in runs),
        max_no_speech_prob=max(run.max_no_speech_prob for run in runs),
        empty=any(run.empty for run in runs),
        bengali_share=bengali_share(runs[0].text),
        text=" ".join(runs[0].text.split()),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--budget-s", type=float, default=float("inf"))
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    capped = [name for name in args.configs if "max_new_tokens" in CONFIGS[name]]
    if capped and args.max_new_tokens is None:
        parser.error(f"--max-new-tokens is required for {', '.join(capped)}")

    started = perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("\t".join(COLUMNS))
    for clip in args.clips:
        audio_16k = load_mono_16k(clip)
        for config in args.configs:
            if perf_counter() - started > args.budget_s:
                skipped = [clip.name, args.device, config, "skipped (budget)"]
                print("\t".join(skipped + ["-"] * (len(COLUMNS) - len(skipped))))
                continue
            options = dict(CONFIGS[config])
            if "max_new_tokens" in options:
                options["max_new_tokens"] = args.max_new_tokens
            run_count = 1 if config == "current" else args.runs
            runs = [run_once(model, audio_16k, options) for _ in range(run_count)]
            row = summarize(clip.name, args.device, config, runs)
            print(
                "\t".join(
                    [
                        row.clip,
                        row.device,
                        row.config,
                        str(row.runs),
                        f"{row.median_s:.2f}",
                        f"{row.max_s:.2f}",
                        "-" if row.repeatable is None else ("yes" if row.repeatable else "no"),
                        str(row.max_tokens),
                        f"{row.max_compression_ratio:.2f}",
                        ",".join(f"{t:.1f}" for t in row.temperatures) or "-",
                        f"{row.worst_avg_logprob:.3f}",
                        f"{row.max_no_speech_prob:.3f}",
                        "yes" if row.empty else "no",
                        "-" if row.bengali_share is None else f"{row.bengali_share:.2f}",
                        row.text or "-",
                    ]
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
