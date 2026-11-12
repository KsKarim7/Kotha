import csv
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import jiwer

from kotha.text.normalize import normalize

SPLITS = ("dev", "test", "all")


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


def score(results: Sequence[Result]) -> list[GroupScore]:
    pairs = normalised_pairs(results)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, result in enumerate(results):
        groups[(result.condition, result.row.lang)].append(index)
        groups[(result.condition, "all")].append(index)
    ordered_keys = sorted(groups, key=lambda key: (key[0], key[1] == "all", key[1]))
    return [group_score(key, groups[key], results, pairs) for key in ordered_keys]


def group_score(
    key: tuple[str, str],
    indices: list[int],
    results: Sequence[Result],
    pairs: Sequence[tuple[str, str]],
) -> GroupScore:
    references = [pairs[index][0] for index in indices]
    hypotheses = [pairs[index][1] for index in indices]
    p50, p95 = latency_percentiles([results[index].latency_s for index in indices])
    condition, lang = key
    return GroupScore(
        condition=condition,
        lang=lang,
        n=len(indices),
        wer=float(jiwer.wer(references, hypotheses)),
        cer=float(jiwer.cer(references, hypotheses)),
        latency_p50_s=p50,
        latency_p95_s=p95,
    )


def utterance_scores(results: Sequence[Result]) -> list[UtteranceScore]:
    return [
        UtteranceScore(
            id=result.row.id,
            lang=result.row.lang,
            noise=result.row.noise,
            condition=result.condition,
            reference=reference,
            hypothesis=hypothesis,
            wer=float(jiwer.wer(reference, hypothesis)),
            cer=float(jiwer.cer(reference, hypothesis)),
            latency_s=result.latency_s,
        )
        for result, (reference, hypothesis) in zip(results, normalised_pairs(results), strict=True)
    ]
