import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from wer import (
    GroupScore,
    ManifestRow,
    Result,
    UtteranceScore,
    read_manifest,
    score,
    select_split,
    utterance_scores,
)

HEADER = ["id", "path", "speaker", "lang", "noise", "split", "text"]
MANIFEST_ROWS = [
    ["bn-1", "audio/bn-1.wav", "s01", "bn", "clean", "dev", "আমার নম্বর কত?"],
    ["en-1", "audio/en-1.wav", "s02", "en", "street", "test", "Check my balance."],
    ["mix-1", "audio/mix-1.wav", "s01", "mix", "fan", "dev", "আমার 30GB প্যাকেজ activate করো"],
]


def write_manifest(
    path: Path, header: list[str], rows: list[list[str]], encoding: str = "utf-8"
) -> Path:
    with path.open("w", newline="", encoding=encoding) as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def manifest_row(
    id: str = "u1", lang: str = "bn", text: str = "আমার নম্বর", split: str = "dev"
) -> ManifestRow:
    return ManifestRow(id, f"audio/{id}.wav", "s01", lang, "clean", split, text)


def result(
    text: str,
    hypothesis: str,
    *,
    id: str = "u1",
    lang: str = "bn",
    condition: str = "16k",
    latency_s: float = 1.0,
) -> Result:
    return Result(manifest_row(id, lang, text), condition, hypothesis, latency_s)


def test_read_manifest_reads_rows_and_ignores_extra_columns(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "manifest.csv",
        HEADER + ["notes"],
        [values + ["recorded at home"] for values in MANIFEST_ROWS],
    )
    assert read_manifest(path) == [ManifestRow(*values) for values in MANIFEST_ROWS]


def test_read_manifest_accepts_a_byte_order_mark(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "manifest.csv", HEADER, MANIFEST_ROWS, "utf-8-sig")
    assert [row.id for row in read_manifest(path)] == ["bn-1", "en-1", "mix-1"]


def test_read_manifest_names_the_missing_columns(tmp_path: Path) -> None:
    kept = [index for index, column in enumerate(HEADER) if column not in ("speaker", "split")]
    path = write_manifest(
        tmp_path / "manifest.csv",
        [HEADER[index] for index in kept],
        [[values[index] for index in kept] for values in MANIFEST_ROWS],
    )
    with pytest.raises(ValueError, match="missing columns: speaker, split$"):
        read_manifest(path)


@pytest.mark.parametrize(
    ("split", "ids"),
    [("dev", ["bn-1", "mix-1"]), ("test", ["en-1"]), ("all", ["bn-1", "en-1", "mix-1"])],
)
def test_select_split(split: str, ids: list[str]) -> None:
    rows = [ManifestRow(*values) for values in MANIFEST_ROWS]
    assert [row.id for row in select_split(rows, split)] == ids


def test_select_split_rejects_an_unknown_split() -> None:
    with pytest.raises(ValueError, match="'train'"):
        select_split([manifest_row()], "train")


def test_score_reports_corpus_wer_per_lang_and_overall() -> None:
    scores = score(
        [
            result("আমার নম্বর", "আমার নাম্বার", id="bn-1", lang="bn"),
            result("check my data balance", "check my data balanced", id="en-1", lang="en"),
        ]
    )
    assert [(group.lang, group.n, group.wer) for group in scores] == [
        ("bn", 1, 0.5),
        ("en", 1, 0.25),
        ("all", 2, pytest.approx(2 / 6)),
    ]


@pytest.mark.parametrize(
    ("text", "hypothesis", "cer"),
    [("আমার", "আমর", 0.25), ("ab cd", "abcd", 0.2), ("abcd", "abed", 0.25)],
)
def test_score_counts_cer_over_code_points_including_spaces(
    text: str, hypothesis: str, cer: float
) -> None:
    [lang_group, _] = score([result(text, hypothesis)])
    assert lang_group.cer == cer


def test_score_ignores_punctuation_digits_and_case() -> None:
    scores = score(
        [
            result("আমার ৩০ জিবি প্যাকেজ।", "আমার 30 জিবি প্যাকেজ", id="bn-1", lang="bn"),
            result("Check my balance.", "check my BALANCE", id="en-1", lang="en"),
        ]
    )
    assert [(group.wer, group.cer) for group in scores] == [(0.0, 0.0)] * 3
    assert all(type(group.wer) is float and type(group.cer) is float for group in scores)


def test_score_groups_by_condition_then_lang_with_all_last() -> None:
    scores = score(
        [
            result("check my data balance", "check my data balance", id="en", lang="en"),
            result("আমার নম্বর", "আমার নম্বর", id="bn", lang="bn"),
            result("আমার নম্বর", "আমার নাম্বার", id="bn", lang="bn", condition="8k"),
            result("check my data balance", "check my data balance", lang="en", condition="8k"),
        ]
    )
    assert [(group.condition, group.lang, group.n, group.wer) for group in scores] == [
        ("16k", "bn", 1, 0.0),
        ("16k", "en", 1, 0.0),
        ("16k", "all", 2, 0.0),
        ("8k", "bn", 1, 0.5),
        ("8k", "en", 1, 0.0),
        ("8k", "all", 2, pytest.approx(1 / 6)),
    ]


def test_score_latency_percentiles() -> None:
    results = [
        result("আমার নম্বর", "আমার নম্বর", id=f"u{seconds}", latency_s=seconds)
        for seconds in (5.0, 1.0, 4.0, 2.0, 3.0)
    ]
    assert score(results) == [
        GroupScore("16k", "bn", 5, 0.0, 0.0, 3.0, pytest.approx(4.8)),
        GroupScore("16k", "all", 5, 0.0, 0.0, 3.0, pytest.approx(4.8)),
    ]


def test_score_latency_of_a_single_result() -> None:
    [lang_group, all_group] = score([result("আমার নম্বর", "আমার নম্বর", latency_s=2.5)])
    assert (lang_group.latency_p50_s, lang_group.latency_p95_s) == (2.5, 2.5)
    assert (all_group.latency_p50_s, all_group.latency_p95_s) == (2.5, 2.5)


def test_utterance_scores_keep_input_order_and_normalise_both_sides() -> None:
    scores = utterance_scores(
        [
            result("আমার নম্বর", "আমার নাম্বার।", id="bn-1", lang="bn", latency_s=1.5),
            result("Check my balance.", "check my BALANCE", id="en-1", lang="en", latency_s=0.5),
        ]
    )
    assert scores == [
        UtteranceScore("bn-1", "bn", "clean", "16k", "আমার নম্বর", "আমার নাম্বার", 0.5, 0.2, 1.5),
        UtteranceScore(
            "en-1", "en", "clean", "16k", "check my balance", "check my balance", 0.0, 0.0, 0.5
        ),
    ]
    assert all(type(item.wer) is float and type(item.cer) is float for item in scores)


def test_empty_hypothesis_is_scored_as_all_deletions() -> None:
    [utterance] = utterance_scores([result("আমার নম্বর কত", "")])
    assert (utterance.hypothesis, utterance.wer, utterance.cer) == ("", 1.0, 1.0)
    [lang_group, _] = score([result("আমার নম্বর কত", ""), result("check my balance", "check")])
    assert lang_group.wer == pytest.approx(5 / 6)


@pytest.mark.parametrize("scorer", [score, utterance_scores])
def test_empty_normalised_reference_names_the_id_but_not_the_text(
    scorer: Callable[[list[Result]], list[GroupScore] | list[UtteranceScore]],
) -> None:
    secret = "গোপন নম্বর"
    with pytest.raises(ValueError) as error:
        scorer(
            [
                result(secret, secret, id="ok-1"),
                result("।", "কিছু", id="empty-1"),
                result(" ? ", "", id="empty-2"),
            ]
        )
    message = str(error.value)
    assert message.endswith("ids: empty-1, empty-2")
    assert secret not in message and "কিছু" not in message and "।" not in message
