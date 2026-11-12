import argparse
import csv
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter

import jiwer

from kotha.asr.whisper import DECODING_OPTIONS, WhisperAsr
from kotha.audio.convert import load_mono_16k
from kotha.audio.telephony import simulate_telephony
from kotha.text.normalize import normalize

SPLITS = ("dev", "test", "all")
CONDITIONS = ("16k", "8k")
BEAM_SIZE = 1
NORMALISER = "kotha.text.normalize.normalize"
REPO_ROOT = Path(__file__).resolve().parent.parent
LEXICON_PATH = REPO_ROOT / "data" / "testsets" / "loanwords.csv"
LEXICON_COLUMNS = ("latin", "bangla")
BENGALI_BLOCK = range(0x0980, 0x0A00)


@dataclass(frozen=True)
class ManifestRow:
    id: str
    path: str
    speaker: str
    lang: str
    noise: str
    split: str
    text: str


MANIFEST_COLUMNS = tuple(field.name for field in fields(ManifestRow))


@dataclass(frozen=True)
class Result:
    row: ManifestRow
    condition: str
    hypothesis: str
    latency_s: float


@dataclass(frozen=True)
class GroupScore:
    condition: str
    lang: str
    n: int
    wer: float
    cer: float
    lexicon_wer: float
    lexicon_cer: float
    latency_p50_s: float
    latency_p95_s: float


@dataclass(frozen=True)
class UtteranceScore:
    id: str
    lang: str
    noise: str
    condition: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    lexicon_wer: float
    lexicon_cer: float
    latency_s: float


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = [column for column in MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest {path} is missing columns: {', '.join(missing)}")
        return [
            ManifestRow(**{column: row[column] for column in MANIFEST_COLUMNS}) for row in reader
        ]


def select_split(rows: Sequence[ManifestRow], split: str) -> list[ManifestRow]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {split!r}")
    if split == "all":
        return list(rows)
    return [row for row in rows if row.split == split]


def read_lexicon(path: Path) -> dict[str, str]:
    lexicon: dict[str, str] = {}
    first_lines: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as lexicon_file:
        reader = csv.DictReader(lexicon_file)
        missing = [column for column in LEXICON_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"lexicon {path} is missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            location = f"lexicon {path}, line {line}"
            latin = lexicon_word(row["latin"], "latin", location)
            bangla = lexicon_word(row["bangla"], "bangla", location)
            if any(ord(char) in BENGALI_BLOCK for char in latin):
                raise ValueError(
                    f"{location}: latin must not contain Bengali letters, got {latin!r}"
                )
            if lexicon.get(bangla, latin) != latin:
                raise ValueError(
                    f"{location}: {bangla!r} maps to {latin!r} here and to {lexicon[bangla]!r}"
                    f" on line {first_lines[bangla]}"
                )
            lexicon[bangla] = latin
            first_lines.setdefault(bangla, line)
    return lexicon


def lexicon_word(value: str | None, column: str, location: str) -> str:
    word = normalize(value or "")
    if len(word.split()) != 1:
        raise ValueError(f"{location}: {column} must be one word after normalising, got {word!r}")
    return word


def apply_lexicon(normalised_text: str, lexicon: Mapping[str, str]) -> str:
    return " ".join(lexicon.get(token, token) for token in normalised_text.split())


def normalised_pairs(results: Sequence[Result]) -> list[tuple[str, str]]:
    pairs = [(normalize(result.row.text), normalize(result.hypothesis)) for result in results]
    empty_ids = [
        result.row.id
        for result, (reference, _) in zip(results, pairs, strict=True)
        if not reference
    ]
    if empty_ids:
        raise ValueError(f"empty reference after normalisation for ids: {', '.join(empty_ids)}")
    return pairs


def latency_percentiles(latencies: Sequence[float]) -> tuple[float, float]:
    if len(latencies) == 1:
        return latencies[0], latencies[0]
    p95 = statistics.quantiles(latencies, n=20, method="inclusive")[18]
    return statistics.median(latencies), p95


def map_pair(pair: tuple[str, str], lexicon: Mapping[str, str]) -> tuple[str, str]:
    reference, hypothesis = pair
    return apply_lexicon(reference, lexicon), apply_lexicon(hypothesis, lexicon)


def error_rates(pairs: Sequence[tuple[str, str]]) -> tuple[float, float]:
    references = [reference for reference, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    return float(jiwer.wer(references, hypotheses)), float(jiwer.cer(references, hypotheses))


def score(results: Sequence[Result], lexicon: Mapping[str, str]) -> list[GroupScore]:
    pairs = normalised_pairs(results)
    lexicon_pairs = [map_pair(pair, lexicon) for pair in pairs]
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, result in enumerate(results):
        groups[(result.condition, result.row.lang)].append(index)
        groups[(result.condition, "all")].append(index)
    ordered_keys = sorted(groups, key=lambda key: (key[0], key[1] == "all", key[1]))
    return [group_score(key, groups[key], results, pairs, lexicon_pairs) for key in ordered_keys]


def group_score(
    key: tuple[str, str],
    indices: list[int],
    results: Sequence[Result],
    pairs: Sequence[tuple[str, str]],
    lexicon_pairs: Sequence[tuple[str, str]],
) -> GroupScore:
    wer, cer = error_rates([pairs[index] for index in indices])
    lexicon_wer, lexicon_cer = error_rates([lexicon_pairs[index] for index in indices])
    p50, p95 = latency_percentiles([results[index].latency_s for index in indices])
    condition, lang = key
    return GroupScore(
        condition=condition,
        lang=lang,
        n=len(indices),
        wer=wer,
        cer=cer,
        lexicon_wer=lexicon_wer,
        lexicon_cer=lexicon_cer,
        latency_p50_s=p50,
        latency_p95_s=p95,
    )


def utterance_scores(results: Sequence[Result], lexicon: Mapping[str, str]) -> list[UtteranceScore]:
    pairs = normalised_pairs(results)
    return [
        utterance_score(result, pair, lexicon) for result, pair in zip(results, pairs, strict=True)
    ]


def utterance_score(
    result: Result, pair: tuple[str, str], lexicon: Mapping[str, str]
) -> UtteranceScore:
    wer, cer = error_rates([pair])
    lexicon_wer, lexicon_cer = error_rates([map_pair(pair, lexicon)])
    reference, hypothesis = pair
    return UtteranceScore(
        id=result.row.id,
        lang=result.row.lang,
        noise=result.row.noise,
        condition=result.condition,
        reference=reference,
        hypothesis=hypothesis,
        wer=wer,
        cer=cer,
        lexicon_wer=lexicon_wer,
        lexicon_cer=lexicon_cer,
        latency_s=result.latency_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--condition", nargs="+", choices=CONDITIONS, default=["16k"])
    parser.add_argument("--language", choices=("bn", "auto"), default="bn")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, default=LEXICON_PATH)
    return parser


def transcribe_row(
    asr: WhisperAsr, row: ManifestRow, data_dir: Path, condition: str, language: str | None
) -> Result:
    audio_16k = load_mono_16k(data_dir / row.path)
    if condition == "8k":
        audio_16k = simulate_telephony(audio_16k)
    started = perf_counter()
    transcript = asr.transcribe(audio_16k, language=language, beam_size=BEAM_SIZE)
    latency_s = perf_counter() - started
    return Result(row, condition, transcript.text, latency_s)


def transcribe_all(
    asr: WhisperAsr,
    rows: Sequence[ManifestRow],
    data_dir: Path,
    conditions: Sequence[str],
    language: str | None,
) -> list[Result]:
    asr.transcribe(load_mono_16k(data_dir / rows[0].path), language=language, beam_size=BEAM_SIZE)
    return [
        transcribe_row(asr, row, data_dir, condition, language)
        for condition in conditions
        for row in rows
    ]


def print_summary(
    args: argparse.Namespace, row_count: int, lexicon_size: int, scores: Sequence[GroupScore]
) -> None:
    decoding = {"beam_size": BEAM_SIZE, **DECODING_OPTIONS}
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"compute_type: {args.compute_type}")
    print(f"language: {args.language}")
    print(f"split: {args.split}")
    print(f"rows: {row_count}")
    print(f"decoding: {', '.join(f'{key}={value}' for key, value in decoding.items())}")
    print(f"normaliser: {NORMALISER}")
    print(f"lexicon: {args.lexicon} ({lexicon_size} spellings)")
    print()
    print("condition\tlang\tn\twer\tcer\tlexicon_wer\tlexicon_cer\tlatency_p50_s\tlatency_p95_s")
    for group in scores:
        print(
            f"{group.condition}\t{group.lang}\t{group.n}\t{group.wer:.4f}\t{group.cer:.4f}"
            f"\t{group.lexicon_wer:.4f}\t{group.lexicon_cer:.4f}"
            f"\t{group.latency_p50_s:.2f}\t{group.latency_p95_s:.2f}"
        )
    print()
    print("CER counts Unicode code points including spaces")
    print(
        "wer/cer are the primary scores; lexicon_wer/lexicon_cer map the lexicon's "
        "Bengali-script loanword spellings to Latin on both sides"
    )


def write_utterances(path: Path, scores: Sequence[UtteranceScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=[field.name for field in fields(UtteranceScore)]
        )
        writer.writeheader()
        writer.writerows(asdict(utterance) for utterance in scores)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(REPO_ROOT):
        parser.error(
            f"--output {args.output} is inside the repository; the CSV holds transcripts of "
            "test audio and must be written outside it"
        )
    rows = select_split(read_manifest(args.manifest), args.split)
    if not rows:
        parser.error(f"no rows in split {args.split!r} of {args.manifest}")
    lexicon = read_lexicon(args.lexicon)
    conditions = list(dict.fromkeys(args.condition))
    language = None if args.language == "auto" else "bn"
    asr = WhisperAsr(args.model, device=args.device, compute_type=args.compute_type)
    results = transcribe_all(asr, rows, args.data_dir, conditions, language)
    print_summary(args, len(rows), len(lexicon), score(results, lexicon))
    write_utterances(output, utterance_scores(results, lexicon))


if __name__ == "__main__":
    main()
