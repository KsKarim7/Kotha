import argparse
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from wer import ManifestRow, read_manifest

LANGS = ("bn", "en", "mix")
NOISES = ("clean", "street", "fan")
SPLITS = ("dev", "test")
SPEAKER_PATTERN = re.compile(r"spk[0-9]{2}")
OTHER = "other"
MIN_SAMPLE_RATE_HZ = 16_000
EXPECTED_CONTAINERS = {
    ".wav": ("WAV", "WAVEX", "RF64"),
    ".flac": ("FLAC",),
    ".mp3": ("MP3",),
    ".ogg": ("OGG",),
}
AAC_SUFFIXES = (".m4a", ".aac", ".mp4")


@dataclass(frozen=True)
class AudioInfo:
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration_s: float


@dataclass(frozen=True)
class Clip:
    row: ManifestRow
    audio: AudioInfo | None


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)


def field_errors(row: ManifestRow) -> list[str]:
    errors = []
    if not row.id:
        errors.append("empty id")
    if not row.text:
        errors.append("empty text")
    if row.lang not in LANGS:
        errors.append(f"lang must be one of {', '.join(LANGS)}")
    if row.noise not in NOISES:
        errors.append(f"noise must be one of {', '.join(NOISES)}")
    if row.split not in SPLITS:
        errors.append(f"split must be one of {', '.join(SPLITS)}")
    if not SPEAKER_PATTERN.fullmatch(row.speaker or ""):
        errors.append("speaker must be spk followed by two digits")
    return errors


def path_error(relative_path: str, data_dir: Path) -> str | None:
    if not relative_path:
        return "empty path"
    path = Path(relative_path)
    if path.is_absolute():
        return "path is absolute"
    if ".." in path.parts:
        return "path contains .."
    if not (data_dir / path).resolve().is_relative_to(data_dir.resolve()):
        return "path resolves outside --data-dir"
    return None


def read_audio_info(audio_path: Path) -> AudioInfo:
    info = sf.info(audio_path)
    return AudioInfo(
        container=info.format,
        codec=f"{info.format}/{info.subtype}",
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_s=info.duration,
    )


def unreadable_message(suffix: str) -> str:
    if suffix.lower() in AAC_SUFFIXES:
        return "unreadable by soundfile (real AAC .m4a?)"
    return "unreadable by soundfile"


def audio_warnings(audio: AudioInfo, suffix: str) -> list[str]:
    warnings = []
    if audio.container not in EXPECTED_CONTAINERS.get(suffix.lower(), ()):
        warnings.append(f"{audio.container} data in a file named {suffix or '(no extension)'}")
    if audio.sample_rate < MIN_SAMPLE_RATE_HZ:
        warnings.append(f"sample rate {audio.sample_rate} Hz is below {MIN_SAMPLE_RATE_HZ} Hz")
    if audio.channels > 1:
        warnings.append(f"{audio.channels} channels, downmixed to mono")
    return warnings


def check_file(
    relative_path: str, data_dir: Path, min_s: float, max_s: float
) -> tuple[AudioInfo | None, list[str], list[str]]:
    problem = path_error(relative_path, data_dir)
    if problem:
        return None, [problem], []
    audio_path = data_dir / relative_path
    if not audio_path.is_file():
        return None, ["file missing"], []
    try:
        audio = read_audio_info(audio_path)
    except sf.LibsndfileError:
        return None, [unreadable_message(audio_path.suffix)], []
    errors = []
    if not min_s <= audio.duration_s <= max_s:
        errors.append(f"duration {audio.duration_s:.2f} s is outside [{min_s}, {max_s}] s")
    return audio, errors, audio_warnings(audio, audio_path.suffix)


def validate(
    manifest_paths: Sequence[Path], data_dir: Path, min_s: float, max_s: float
) -> Validation:
    validation = Validation()
    first_ids: dict[str, str] = {}
    first_paths: dict[str, str] = {}
    for manifest_path in manifest_paths:
        try:
            rows = read_manifest(manifest_path)
        except ValueError as error:
            validation.errors.append(f"{manifest_path.name}: {error}")
            continue
        for line, row in enumerate(rows, start=2):
            where = f"{manifest_path.name}:{line}"
            errors = field_errors(row)
            if row.id in first_ids:
                errors.append(f"id repeated (first at {first_ids[row.id]})")
            elif row.id:
                first_ids[row.id] = where
            path_key = os.path.normpath(row.path) if row.path else ""
            if path_key in first_paths:
                errors.append(f"path repeated (first at {first_paths[path_key]})")
            elif path_key:
                first_paths[path_key] = where
            audio, file_errors, warnings = check_file(row.path or "", data_dir, min_s, max_s)
            location = f"{where} {row.id or '-'}"
            validation.errors += [f"{location}: {error}" for error in errors + file_errors]
            validation.warnings += [f"{location}: warning: {warning}" for warning in warnings]
            validation.clips.append(Clip(row, audio))
    validation.warnings += set_warnings(validation.clips)
    return validation


def format_counts(counts: Counter[str] | Counter[int], unit: str = "") -> str:
    if not counts:
        return "none"
    return ", ".join(f"{value}{unit} ({count})" for value, count in sorted(counts.items()))


def set_warnings(clips: Sequence[Clip]) -> list[str]:
    audios = [clip.audio for clip in clips if clip.audio]
    codecs = Counter(audio.codec for audio in audios)
    sample_rates = Counter(audio.sample_rate for audio in audios)
    warnings = []
    if len(codecs) > 1:
        warnings.append(f"warning: the set mixes {len(codecs)} codecs: {format_counts(codecs)}")
    if len(sample_rates) > 1:
        warnings.append(
            f"warning: the set mixes {len(sample_rates)} sample rates: "
            f"{format_counts(sample_rates, ' Hz')}"
        )
    return warnings


def clip_minutes(clip: Clip) -> float:
    return clip.audio.duration_s / 60 if clip.audio else 0.0


def category(value: str | None, allowed: Iterable[str]) -> str:
    return value if value in allowed else OTHER


def speaker_label(clip: Clip) -> str:
    speaker = clip.row.speaker or ""
    return speaker if SPEAKER_PATTERN.fullmatch(speaker) else OTHER


def tally(
    clips: Iterable[Clip], label: Callable[[Clip], Hashable]
) -> dict[Hashable, tuple[int, float]]:
    counts: Counter[Hashable] = Counter()
    minutes: defaultdict[Hashable, float] = defaultdict(float)
    for clip in clips:
        key = label(clip)
        counts[key] += 1
        minutes[key] += clip_minutes(clip)
    return {key: (counts[key], minutes[key]) for key in counts}


def with_other(allowed: Sequence[str], seen: Iterable[Hashable]) -> list[str]:
    return [*allowed, OTHER] if OTHER in set(seen) else list(allowed)


def grid_lines(clips: Sequence[Clip]) -> list[str]:
    cells = tally(
        clips, lambda clip: (category(clip.row.lang, LANGS), category(clip.row.noise, NOISES))
    )
    langs = with_other(LANGS, (lang for lang, _ in cells))
    noises = with_other(NOISES, (noise for _, noise in cells))
    lines = [
        "lang x noise (clips / minutes)",
        "lang".ljust(8) + "".join(n.rjust(14) for n in noises),
    ]
    for lang in langs:
        counts = [cells.get((lang, noise), (0, 0.0)) for noise in noises]
        lines.append(
            lang.ljust(8) + "".join(f"{n} / {minutes:.2f}".rjust(14) for n, minutes in counts)
        )
    return lines


def table_lines(
    title: str, totals: dict[Hashable, tuple[int, float]], keys: Sequence[str]
) -> list[str]:
    lines = [f"{title:<8}{'clips':>7}{'minutes':>9}"]
    for key in keys:
        count, minutes = totals.get(key, (0, 0.0))
        lines.append(f"{key:<8}{count:>7}{minutes:>9.2f}")
    return lines


def summary_lines(clips: Sequence[Clip]) -> list[str]:
    speakers = tally(clips, speaker_label)
    splits = tally(clips, lambda clip: category(clip.row.split, SPLITS))
    speaker_codes = sorted(str(key) for key in speakers if key != OTHER)
    audios = [clip.audio for clip in clips if clip.audio]
    total_minutes = sum(clip_minutes(clip) for clip in clips)
    return [
        f"{len(clips)} rows, {len(speaker_codes)} speakers, {total_minutes:.2f} minutes",
        "",
        *grid_lines(clips),
        "",
        *table_lines("speaker", speakers, with_other(speaker_codes, speakers)),
        "",
        *table_lines("split", splits, with_other(SPLITS, splits)),
        "",
        f"codecs: {format_counts(Counter(audio.codec for audio in audios))}",
        f"sample rates: {format_counts(Counter(audio.sample_rate for audio in audios), ' Hz')}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check ASR test-set manifests and their audio, and summarise the set. "
            "Reference text is never printed."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help="manifest CSVs, for example the public one and an optional private one",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="folder that the manifests' path column is relative to",
    )
    parser.add_argument(
        "--min-s", type=float, default=0.5, help="shortest allowed clip in seconds (%(default)s)"
    )
    parser.add_argument(
        "--max-s", type=float, default=30.0, help="longest allowed clip in seconds (%(default)s)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation = validate(args.manifest, args.data_dir, args.min_s, args.max_s)
    for line in [*validation.errors, *validation.warnings]:
        print(line)
    if validation.errors or validation.warnings:
        print()
    for line in summary_lines(validation.clips):
        print(line)
    print()
    print(f"{len(validation.errors)} errors, {len(validation.warnings)} warnings")
    return 1 if validation.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
