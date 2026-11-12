import csv
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from kotha.asr.whisper import DECODING_OPTIONS, Transcript
from kotha.audio.convert import SAMPLE_RATE_HZ
from kotha.text.normalize import normalize
from wer import (
    LEXICON_PATH,
    REPO_ROOT,
    GroupScore,
    ManifestRow,
    Result,
    UtteranceScore,
    apply_lexicon,
    main,
    read_lexicon,
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
NO_LEXICON: dict[str, str] = {}
LEXICON = {"মোবাইল": "mobile", "ব্যালেন্স": "balance", "চেক": "check", "ডাটা": "data"}


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
        ],
        NO_LEXICON,
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
    [lang_group, _] = score([result(text, hypothesis)], NO_LEXICON)
    assert lang_group.cer == cer


def test_score_ignores_punctuation_digits_and_case() -> None:
    scores = score(
        [
            result("আমার ৩০ জিবি প্যাকেজ।", "আমার 30 জিবি প্যাকেজ", id="bn-1", lang="bn"),
            result("Check my balance.", "check my BALANCE", id="en-1", lang="en"),
        ],
        NO_LEXICON,
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
        ],
        NO_LEXICON,
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
    assert score(results, NO_LEXICON) == [
        GroupScore("16k", "bn", 5, 0.0, 0.0, 0.0, 0.0, 3.0, pytest.approx(4.8)),
        GroupScore("16k", "all", 5, 0.0, 0.0, 0.0, 0.0, 3.0, pytest.approx(4.8)),
    ]


def test_score_latency_of_a_single_result() -> None:
    [lang_group, all_group] = score([result("আমার নম্বর", "আমার নম্বর", latency_s=2.5)], NO_LEXICON)
    assert (lang_group.latency_p50_s, lang_group.latency_p95_s) == (2.5, 2.5)
    assert (all_group.latency_p50_s, all_group.latency_p95_s) == (2.5, 2.5)


def test_utterance_scores_keep_input_order_and_normalise_both_sides() -> None:
    scores = utterance_scores(
        [
            result("আমার নম্বর", "আমার নাম্বার।", id="bn-1", lang="bn", latency_s=1.5),
            result("Check my balance.", "check my BALANCE", id="en-1", lang="en", latency_s=0.5),
        ],
        NO_LEXICON,
    )
    assert scores == [
        UtteranceScore(
            "bn-1", "bn", "clean", "16k", "আমার নম্বর", "আমার নাম্বার", 0.5, 0.2, 0.5, 0.2, 1.5
        ),
        UtteranceScore(
            "en-1",
            "en",
            "clean",
            "16k",
            "check my balance",
            "check my balance",
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
        ),
    ]
    assert all(type(item.wer) is float and type(item.cer) is float for item in scores)


def test_empty_hypothesis_is_scored_as_all_deletions() -> None:
    [utterance] = utterance_scores([result("আমার নম্বর কত", "")], NO_LEXICON)
    assert (utterance.hypothesis, utterance.wer, utterance.cer) == ("", 1.0, 1.0)
    [lang_group, _] = score(
        [result("আমার নম্বর কত", ""), result("check my balance", "check")], NO_LEXICON
    )
    assert lang_group.wer == pytest.approx(5 / 6)


@pytest.mark.parametrize("scorer", [score, utterance_scores])
def test_empty_normalised_reference_names_the_id_but_not_the_text(
    scorer: Callable[[list[Result], Mapping[str, str]], list[GroupScore] | list[UtteranceScore]],
) -> None:
    secret = "গোপন নম্বর"
    with pytest.raises(ValueError) as error:
        scorer(
            [
                result(secret, secret, id="ok-1"),
                result("।", "কিছু", id="empty-1"),
                result(" ? ", "", id="empty-2"),
            ],
            NO_LEXICON,
        )
    message = str(error.value)
    assert message.endswith("ids: empty-1, empty-2")
    assert secret not in message and "কিছু" not in message and "।" not in message


CLI_ROWS = [
    ["bn-1", "bn-1.wav", "s01", "bn", "clean", "dev", "আমার নম্বর কত?"],
    ["en-1", "en-1.wav", "s02", "en", "street", "test", "Check my balance."],
    ["mix-1", "mix-1.wav", "s01", "mix", "fan", "dev", "আমার 30GB প্যাকেজ activate করো"],
]
CLIP_SAMPLES = {"bn-1": 1_600, "en-1": 3_200, "mix-1": 4_800}
HYPOTHESES = {
    1_600: "আমার নাম্বার কত",
    3_200: "check my balance",
    4_800: "আমার 30 GB প্যাকেজ activate করো",
}
TABLE_HEADER = (
    "condition\tlang\tn\twer\tcer\tlexicon_wer\tlexicon_cer\tlatency_p50_s\tlatency_p95_s"
)
CLOSING_LINES = [
    "CER counts Unicode code points including spaces",
    "wer/cer are the primary scores; lexicon_wer/lexicon_cer map the lexicon's "
    "Bengali-script loanword spellings to Latin on both sides",
]


class FakeAsr:
    instances: list["FakeAsr"] = []

    def __init__(self, model: str, *, device: str, compute_type: str) -> None:
        self.settings = (model, device, compute_type)
        self.calls: list[tuple[np.ndarray, str | None, int]] = []
        FakeAsr.instances.append(self)

    def transcribe(
        self, audio_16k: np.ndarray, *, language: str | None, beam_size: int
    ) -> Transcript:
        self.calls.append((audio_16k, language, beam_size))
        return Transcript(HYPOTHESES[audio_16k.size], "bn", -0.3, 0.1, 1.2, 5, ())


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    FakeAsr.instances = []
    monkeypatch.setattr("wer.WhisperAsr", FakeAsr)
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    for clip_id, samples in CLIP_SAMPLES.items():
        sf.write(data_dir / f"{clip_id}.wav", np.zeros(samples, dtype=np.float32), SAMPLE_RATE_HZ)
    return SimpleNamespace(
        data_dir=data_dir,
        manifest=write_manifest(tmp_path / "manifest.csv", HEADER, CLI_ROWS),
        output=tmp_path / "results" / "nested" / "utterances.csv",
        lexicon=write_manifest(
            tmp_path / "loanwords.csv",
            ["latin", "bangla"],
            [["mobile", "মোবাইল"], ["balance", "ব্যালেন্স"]],
        ),
    )


def install_clock(monkeypatch: pytest.MonkeyPatch, latencies: list[float]) -> Iterator[float]:
    ticks = iter(
        [
            tick
            for index, latency in enumerate(latencies)
            for tick in (10.0 * index, 10.0 * index + latency)
        ]
    )
    monkeypatch.setattr("wer.perf_counter", lambda: next(ticks))
    return ticks


def run_cli(cli: SimpleNamespace, *options: str) -> None:
    main(
        [
            "--manifest",
            str(cli.manifest),
            "--data-dir",
            str(cli.data_dir),
            "--output",
            str(cli.output),
            "--lexicon",
            str(cli.lexicon),
            *options,
        ]
    )


def summary_rows(output: str) -> list[list[str]]:
    lines = output.splitlines()
    start = lines.index(TABLE_HEADER) + 1
    return [line.split("\t") for line in lines[start : lines.index("", start)]]


def test_main_scores_a_manifest_end_to_end(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_clock(monkeypatch, [1.0, 2.0, 3.0])
    run_cli(cli, "--split", "all")
    output = capsys.readouterr().out
    assert summary_rows(output) == [
        ["16k", "bn", "1", "0.3333", "0.1538", "0.3333", "0.1538", "1.00", "1.00"],
        ["16k", "en", "1", "0.0000", "0.0000", "0.0000", "0.0000", "2.00", "2.00"],
        ["16k", "mix", "1", "0.4000", "0.0333", "0.4000", "0.0333", "3.00", "3.00"],
        ["16k", "all", "3", "0.2727", "0.0508", "0.2727", "0.0508", "2.00", "2.90"],
    ]
    assert output.splitlines()[-2:] == CLOSING_LINES
    with cli.output.open(newline="", encoding="utf-8") as csv_file:
        written = list(csv.DictReader(csv_file))
    assert [
        (
            row["id"],
            row["lang"],
            row["noise"],
            row["condition"],
            row["reference"],
            row["hypothesis"],
        )
        for row in written
    ] == [
        ("bn-1", "bn", "clean", "16k", "আমার নম্বর কত", "আমার নাম্বার কত"),
        ("en-1", "en", "street", "16k", "check my balance", "check my balance"),
        (
            "mix-1",
            "mix",
            "fan",
            "16k",
            "আমার 30gb প্যাকেজ activate করো",
            "আমার 30 gb প্যাকেজ activate করো",
        ),
    ]
    assert [(float(row["wer"]), float(row["cer"]), float(row["latency_s"])) for row in written] == [
        (pytest.approx(1 / 3), pytest.approx(2 / 13), 1.0),
        (0.0, 0.0, 2.0),
        (pytest.approx(2 / 5), pytest.approx(1 / 30), 3.0),
    ]


def test_warmup_is_one_extra_call_that_is_neither_timed_nor_scored(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ticks = install_clock(monkeypatch, [1.0, 2.0, 3.0])
    run_cli(cli, "--split", "all")
    [asr] = FakeAsr.instances
    assert [audio.size for audio, _, _ in asr.calls] == [1_600, 1_600, 3_200, 4_800]
    assert next(ticks, None) is None
    assert summary_rows(capsys.readouterr().out)[-1][:3] == ["16k", "all", "3"]


def test_8k_condition_goes_through_telephony(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    telephony_outputs: list[np.ndarray] = []

    def fake_telephony(audio_16k: np.ndarray) -> np.ndarray:
        telephony_outputs.append(audio_16k.copy())
        return telephony_outputs[-1]

    monkeypatch.setattr("wer.simulate_telephony", fake_telephony)
    install_clock(monkeypatch, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    run_cli(cli, "--split", "all", "--condition", "16k", "8k")
    assert [row[:2] for row in summary_rows(capsys.readouterr().out)] == [
        ["16k", "bn"],
        ["16k", "en"],
        ["16k", "mix"],
        ["16k", "all"],
        ["8k", "bn"],
        ["8k", "en"],
        ["8k", "mix"],
        ["8k", "all"],
    ]
    [asr] = FakeAsr.instances
    assert len(telephony_outputs) == 3
    assert all(
        audio is sent for (audio, _, _), sent in zip(asr.calls[4:], telephony_outputs, strict=True)
    )
    with cli.output.open(newline="", encoding="utf-8") as csv_file:
        assert [row["condition"] for row in csv.DictReader(csv_file)] == ["16k"] * 3 + ["8k"] * 3


@pytest.mark.parametrize(("option", "language"), [("auto", None), ("bn", "bn")])
def test_language_option(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, option: str, language: str | None
) -> None:
    install_clock(monkeypatch, [1.0, 2.0, 3.0])
    run_cli(cli, "--split", "all", "--language", option)
    [asr] = FakeAsr.instances
    assert [(call_language, beam_size) for _, call_language, beam_size in asr.calls] == [
        (language, 1)
    ] * 4


def test_split_filters_the_rows(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_clock(monkeypatch, [2.0])
    run_cli(cli, "--split", "test")
    assert summary_rows(capsys.readouterr().out) == [
        ["16k", "en", "1", "0.0000", "0.0000", "0.0000", "0.0000", "2.00", "2.00"],
        ["16k", "all", "1", "0.0000", "0.0000", "0.0000", "0.0000", "2.00", "2.00"],
    ]


def test_an_empty_split_is_an_error(
    cli: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    write_manifest(cli.manifest, HEADER, [CLI_ROWS[0], CLI_ROWS[2]])
    with pytest.raises(SystemExit):
        run_cli(cli, "--split", "test")
    assert "no rows in split 'test'" in capsys.readouterr().err
    assert FakeAsr.instances == []


def test_output_inside_the_repository_is_refused(
    cli: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.output = REPO_ROOT / "eval" / "out.csv"
    with pytest.raises(SystemExit):
        run_cli(cli)
    assert "must be written outside it" in capsys.readouterr().err
    assert FakeAsr.instances == []
    assert not cli.output.exists()


def test_missing_output_directories_are_created(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_clock(monkeypatch, [1.0, 3.0])
    assert not cli.output.parent.exists()
    run_cli(cli)
    assert cli.output.is_file()


def test_header_shows_the_run_settings_and_decoding_options(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_clock(monkeypatch, [1.0, 2.0, 3.0])
    run_cli(cli, "--split", "all")
    decoding = ", ".join(f"{key}={value}" for key, value in DECODING_OPTIONS.items())
    assert capsys.readouterr().out.splitlines()[:9] == [
        "model: small",
        "device: cpu",
        "compute_type: int8",
        "language: bn",
        "split: all",
        "rows: 3",
        f"decoding: beam_size=1, {decoding}",
        "normaliser: kotha.text.normalize.normalize",
        f"lexicon: {cli.lexicon} (2 spellings)",
    ]
    assert FakeAsr.instances[0].settings == ("small", "cpu", "int8")


def test_read_lexicon_loads_the_committed_starter_list() -> None:
    lexicon = read_lexicon(LEXICON_PATH)
    assert (lexicon["মোবাইল"], lexicon["অ্যাক্টিভ"], lexicon["জিবি"]) == ("mobile", "activate", "gb")
    assert not any(0x0980 <= ord(char) <= 0x09FF for latin in lexicon.values() for char in latin)


def test_read_lexicon_normalises_both_sides(tmp_path: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "মোবাইল")
    assert decomposed != "মোবাইল"
    path = write_manifest(
        tmp_path / "loanwords.csv",
        ["latin", "bangla"],
        [["Mobile", decomposed], ["data", "ডাটা"], ["data", "ডাটা"]],
    )
    lexicon = read_lexicon(path)
    assert lexicon == {"মোবাইল": "mobile", "ডাটা": "data"}
    assert apply_lexicon(normalize("আমার মোবাইল"), lexicon) == "আমার mobile"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([["mobile", "মোবাইল ফোন"]], "line 2: bangla must be one word"),
        ([["mobile", "।"]], "line 2: bangla must be one word"),
        ([["", "মোবাইল"]], "line 2: latin must be one word"),
        ([["মোবাইল", "মোবাইল"]], "line 2: latin must not contain Bengali letters"),
        (
            [["mobile", "মোবাইল"], ["phone", "মোবাইল"]],
            "line 3: 'মোবাইল' maps to 'phone' here and to 'mobile' on line 2",
        ),
    ],
)
def test_read_lexicon_rejects_bad_rows(tmp_path: Path, rows: list[list[str]], message: str) -> None:
    path = write_manifest(tmp_path / "loanwords.csv", ["latin", "bangla"], rows)
    with pytest.raises(ValueError, match=message):
        read_lexicon(path)


def test_read_lexicon_names_missing_columns(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "loanwords.csv", ["latin", "bengali"], [["sim", "সিম"]])
    with pytest.raises(ValueError, match="missing columns: bangla$"):
        read_lexicon(path)


@pytest.mark.parametrize(
    ("text", "mapped"),
    [
        ("মোবাইলটা আমার", "মোবাইলটা আমার"),
        ("আমার মোবাইল ব্যালেন্স চেক করো", "আমার mobile balance check করো"),
        ("", ""),
    ],
)
def test_apply_lexicon_replaces_whole_tokens_only(text: str, mapped: str) -> None:
    assert apply_lexicon(text, LEXICON) == mapped


def test_lexicon_score_maps_loanwords_to_latin() -> None:
    [lang_group, _] = score([result("আমার mobile balance কত", "আমার মোবাইল ব্যালেন্স কত")], LEXICON)
    assert (lang_group.wer, lang_group.cer) == (0.5, pytest.approx(15 / 22))
    assert (lang_group.lexicon_wer, lang_group.lexicon_cer) == (0.0, 0.0)


def test_lexicon_score_equals_the_strict_score_with_an_empty_lexicon() -> None:
    scores = score(
        [
            result("আমার mobile balance কত", "আমার মোবাইল ব্যালেন্স কথ", id="bn-1"),
            result("check my data", "চেক my ডাটা balance", id="en-1", lang="en"),
        ],
        NO_LEXICON,
    )
    assert all((group.lexicon_wer, group.lexicon_cer) == (group.wer, group.cer) for group in scores)


def test_lexicon_score_is_corpus_level_in_the_all_row() -> None:
    scores = score(
        [
            result("আমার mobile balance কত", "আমার মোবাইল ব্যালেন্স কথ", id="bn-1"),
            result("check my data", "চেক my ডাটা balance", id="en-1", lang="en"),
        ],
        LEXICON,
    )
    assert [(group.lang, group.wer, group.lexicon_wer) for group in scores] == [
        ("bn", 0.75, 0.25),
        ("en", 1.0, pytest.approx(1 / 3)),
        ("all", pytest.approx(6 / 7), pytest.approx(2 / 7)),
    ]


def test_utterance_scores_map_the_reference_side_too() -> None:
    [utterance] = utterance_scores([result("আমার মোবাইল", "আমার mobile")], LEXICON)
    assert (utterance.reference, utterance.hypothesis) == ("আমার মোবাইল", "আমার mobile")
    assert (utterance.wer, utterance.lexicon_wer, utterance.lexicon_cer) == (0.5, 0.0, 0.0)


def test_main_reports_the_lexicon_scores(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(HYPOTHESES, 6_400, "আমার মোবাইল ব্যালেন্স কত")
    sf.write(cli.data_dir / "k8.wav", np.zeros(6_400, dtype=np.float32), SAMPLE_RATE_HZ)
    write_manifest(
        cli.manifest,
        HEADER,
        [["k8", "k8.wav", "s01", "mix", "clean", "dev", "আমার mobile balance কত"]],
    )
    install_clock(monkeypatch, [1.0])
    run_cli(cli)
    output = capsys.readouterr().out
    assert summary_rows(output) == [
        ["16k", "mix", "1", "0.5000", "0.6818", "0.0000", "0.0000", "1.00", "1.00"],
        ["16k", "all", "1", "0.5000", "0.6818", "0.0000", "0.0000", "1.00", "1.00"],
    ]
    assert output.splitlines()[-2:] == CLOSING_LINES
    with cli.output.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        [row] = list(reader)
    assert reader.fieldnames == [
        "id",
        "lang",
        "noise",
        "condition",
        "reference",
        "hypothesis",
        "wer",
        "cer",
        "lexicon_wer",
        "lexicon_cer",
        "latency_s",
    ]
    assert (float(row["wer"]), float(row["cer"])) == (0.5, pytest.approx(15 / 22))
    assert (float(row["lexicon_wer"]), float(row["lexicon_cer"])) == (0.0, 0.0)


def test_main_uses_the_committed_lexicon_by_default(
    cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_clock(monkeypatch, [1.0, 3.0])
    main(
        [
            "--manifest",
            str(cli.manifest),
            "--data-dir",
            str(cli.data_dir),
            "--output",
            str(cli.output),
        ]
    )
    spellings = len(read_lexicon(LEXICON_PATH))
    assert (
        f"lexicon: {LEXICON_PATH} ({spellings} spellings)" in capsys.readouterr().out.splitlines()
    )
