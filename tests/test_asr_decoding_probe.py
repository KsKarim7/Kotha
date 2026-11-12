from collections.abc import Iterator
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asr_decoding_probe import (
    COLUMNS,
    CONFIGS,
    FIXED_OPTIONS,
    RunStats,
    bengali_share,
    main,
    run_once,
    summarize,
)
from kotha.audio.convert import SAMPLE_RATE_HZ


def word(start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=" w", start=start, end=end, probability=0.9)


def segment(
    text: str,
    tokens: int = 5,
    compression_ratio: float = 1.2,
    temperature: float = 0.0,
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.1,
    words: tuple[SimpleNamespace, ...] | None = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        tokens=list(range(tokens)),
        compression_ratio=compression_ratio,
        temperature=temperature,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        words=None if words is None else list(words),
    )


def stats(
    seconds: float,
    text: str = "a",
    temperatures: tuple[float, ...] = (0.0,),
    worst_avg_logprob: float = -0.3,
    max_no_speech_prob: float = 0.1,
) -> RunStats:
    return RunStats(
        seconds,
        text,
        5,
        1.2,
        temperatures,
        worst_avg_logprob,
        max_no_speech_prob,
        not text,
        words=0,
        segments=1,
        first_word_s=None,
        last_word_end_s=None,
    )


class FakeModel:
    instances: list["FakeModel"] = []
    segments = [segment(" hello ")]

    def __init__(self, model: str, *, device: str, compute_type: str) -> None:
        self.settings = (model, device, compute_type)
        self.calls: list[dict] = []
        FakeModel.instances.append(self)

    def transcribe(
        self, audio_16k: np.ndarray, **options: object
    ) -> tuple[Iterator[SimpleNamespace], None]:
        self.calls.append(options)
        return (item for item in self.segments), None


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeModel]:
    FakeModel.instances = []
    monkeypatch.setattr("asr_decoding_probe.WhisperModel", FakeModel)
    return FakeModel


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(SAMPLE_RATE_HZ, dtype=np.float32), SAMPLE_RATE_HZ)
    return path


def rows(output: str) -> list[list[str]]:
    header, *lines = output.splitlines()
    assert header.split("\t") == list(COLUMNS)
    return [line.split("\t") for line in lines]


def test_run_once_consumes_segments_and_takes_the_extremes() -> None:
    model = FakeModel("small", device="cpu", compute_type="int8")
    model.segments = [
        segment(" আমার ", tokens=12, compression_ratio=1.4, temperature=0.0, avg_logprob=-0.4),
        segment("  ", tokens=3, compression_ratio=0.9, temperature=0.2, no_speech_prob=0.6),
        segment(" নম্বর ", tokens=30, compression_ratio=2.1, temperature=0.0, avg_logprob=-1.1),
    ]
    run = run_once(model, np.zeros(10, dtype=np.float32), {"beam_size": 1})
    assert (run.text, run.max_tokens, run.max_compression_ratio) == ("আমার নম্বর", 30, 2.1)
    assert (run.temperatures, run.worst_avg_logprob, run.max_no_speech_prob) == (
        (0.0, 0.2),
        -1.1,
        0.6,
    )
    assert run.empty is False and run.seconds >= 0
    assert model.calls == [
        {
            "language": "bn",
            "condition_on_previous_text": False,
            "word_timestamps": True,
            "vad_filter": False,
            "beam_size": 1,
        }
    ]


def test_run_once_with_no_segments_uses_the_wrapper_sentinels() -> None:
    model = FakeModel("small", device="cpu", compute_type="int8")
    model.segments = []
    run = run_once(model, np.zeros(10, dtype=np.float32), {})
    assert (run.text, run.max_tokens, run.temperatures, run.empty) == ("", 0, (), True)
    assert (run.worst_avg_logprob, run.max_no_speech_prob) == (float("-inf"), 1.0)


def test_summarize_repeatable_runs() -> None:
    row = summarize(
        "clip.wav",
        "cuda",
        "t0",
        2.0,
        [
            stats(3.0, temperatures=(0.0,), worst_avg_logprob=-0.2, max_no_speech_prob=0.1),
            stats(1.0, temperatures=(0.0, 0.4), worst_avg_logprob=-0.9, max_no_speech_prob=0.3),
            stats(2.0, temperatures=(0.2,), worst_avg_logprob=-0.5, max_no_speech_prob=0.2),
        ],
    )
    assert (row.runs, row.median_s, row.max_s, row.repeatable) == (3, 2.0, 3.0, True)
    assert (row.temperatures, row.worst_avg_logprob, row.max_no_speech_prob) == (
        (0.0, 0.2, 0.4),
        -0.9,
        0.3,
    )
    assert (row.empty, row.text) == (False, "a")


def test_summarize_flags_differing_and_empty_runs() -> None:
    row = summarize("clip.wav", "cpu", "current", 2.0, [stats(1.0, "a  b\tc"), stats(1.0, "")])
    assert (row.repeatable, row.empty, row.text) == (False, True, "a b c")


def test_summarize_leaves_repeatability_open_for_a_single_run() -> None:
    row = summarize("clip.wav", "cpu", "current", 2.0, [stats(4.0, "a")])
    assert (row.runs, row.median_s, row.max_s, row.repeatable) == (1, 4.0, 4.0, None)


@pytest.mark.parametrize(
    ("text", "share"),
    [
        ("আমার নম্বর", 1.0),
        ("आमार मुबायल", 0.0),
        ("আমার mobile", 0.4),
        ("আমার ৩০ 30, ৳২৯৯।", 1.0),
        ("", None),
        ("30 ৩০ !", None),
    ],
)
def test_bengali_share_counts_letters_only(text: str, share: float | None) -> None:
    assert bengali_share(text) == share


@pytest.mark.parametrize(
    ("segments", "printed_share"),
    [([segment(" আমার ")], "1.00"), ([segment(" hello ")], "0.00"), ([], "-")],
)
def test_main_prints_the_bengali_share_of_the_first_run(
    fake_model: type[FakeModel],
    clip: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    segments: list[SimpleNamespace],
    printed_share: str,
) -> None:
    monkeypatch.setattr(FakeModel, "segments", segments)
    main(["--clips", str(clip), "--configs", "t0", "--runs", "1"])
    [row] = rows(capsys.readouterr().out)
    assert row[COLUMNS.index("bengali_share")] == printed_share


def test_main_prints_one_row_per_config(
    fake_model: type[FakeModel], clip: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--clips", str(clip), "--device", "cpu", "--max-new-tokens", "50", "--runs", "2"])
    printed = rows(capsys.readouterr().out)
    assert [row[2] for row in printed] == list(CONFIGS)
    assert [row[3] for row in printed] == ["1"] + ["2"] * 10
    assert [row[6] for row in printed] == ["-"] + ["yes"] * 10
    model = fake_model.instances[0]
    assert model.settings == ("small", "cpu", "int8_float16")
    assert [call.get("max_new_tokens") for call in model.calls if "max_new_tokens" in call] == [
        50
    ] * 4


@pytest.mark.parametrize(("config", "beam_size"), [("t0_nots", 1), ("beam5_nots", 5)])
def test_nots_configs_decode_without_timestamps(
    fake_model: type[FakeModel], clip: Path, config: str, beam_size: int
) -> None:
    main(["--clips", str(clip), "--configs", config, "--runs", "1"])
    assert fake_model.instances[0].calls == [
        {**FIXED_OPTIONS, "beam_size": beam_size, "temperature": 0.0, "without_timestamps": True}
    ]


@pytest.mark.parametrize(
    ("config", "timestamp_options"),
    [
        ("t0_nw", {"word_timestamps": False}),
        ("t0_nots_nw", {"without_timestamps": True, "word_timestamps": False}),
    ],
)
def test_configs_override_fixed_options(
    fake_model: type[FakeModel], clip: Path, config: str, timestamp_options: dict
) -> None:
    main(["--clips", str(clip), "--configs", config, "--runs", "1"])
    assert fake_model.instances[0].calls == [
        {**FIXED_OPTIONS, "beam_size": 1, "temperature": 0.0, **timestamp_options}
    ]


def test_warmup_makes_one_untimed_call_with_the_first_config(
    fake_model: type[FakeModel],
    clip: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = count(step=10.0)
    monkeypatch.setattr("asr_decoding_probe.perf_counter", lambda: next(clock))
    main(
        [
            "--clips",
            str(clip),
            "--configs",
            "t0",
            "beam5_t0",
            "--runs",
            "1",
            "--warmup",
            "--budget-s",
            "15",
        ]
    )
    printed = rows(capsys.readouterr().out)
    assert [(row[2], row[3]) for row in printed] == [("t0", "1"), ("beam5_t0", "skipped (budget)")]
    assert printed[0][COLUMNS.index("median_s")] == "10.00"
    warmup_call, t0_call = fake_model.instances[0].calls
    assert warmup_call == t0_call


@pytest.mark.parametrize(
    ("segments", "word_columns"),
    [
        (
            [
                segment(" a b ", words=(word(0.12, 0.5), word(0.6, 1.0))),
                segment(" c ", words=(word(1.8, 2.5),)),
            ],
            ["1.00", "3", "2", "0.12", "2.50"],
        ),
        ([segment(" a ")], ["1.00", "0", "1", "-", "-"]),
        ([segment(" a ", words=None)], ["1.00", "0", "1", "-", "-"]),
        ([], ["1.00", "0", "0", "-", "-"]),
    ],
)
def test_main_prints_the_word_timings_of_the_first_run(
    fake_model: type[FakeModel],
    clip: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    segments: list[SimpleNamespace],
    word_columns: list[str],
) -> None:
    monkeypatch.setattr(FakeModel, "segments", segments)
    main(["--clips", str(clip), "--configs", "t0", "--runs", "1"])
    [row] = rows(capsys.readouterr().out)
    first = COLUMNS.index("audio_s")
    assert row[first : first + 5] == word_columns


def test_word_columns_sit_between_bengali_share_and_text() -> None:
    assert COLUMNS[COLUMNS.index("bengali_share") :] == (
        "bengali_share",
        "audio_s",
        "words",
        "segments",
        "first_word_s",
        "last_word_end_s",
        "text",
    )


def test_main_runs_only_the_selected_configs(
    fake_model: type[FakeModel], clip: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--clips", str(clip), "--configs", "t0", "beam5_t0", "--runs", "1"])
    assert [row[2] for row in rows(capsys.readouterr().out)] == ["t0", "beam5_t0"]


def test_main_skips_configs_once_the_budget_is_spent(
    fake_model: type[FakeModel],
    clip: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = count(step=10.0)
    monkeypatch.setattr("asr_decoding_probe.perf_counter", lambda: next(clock))
    main(["--clips", str(clip), "--configs", "t0", "beam5_t0", "t0_rep", "--runs", "1"])
    main_rows = rows(capsys.readouterr().out)
    assert [row[3] for row in main_rows] == ["1", "1", "1"]
    clock = count(step=10.0)
    monkeypatch.setattr("asr_decoding_probe.perf_counter", lambda: next(clock))
    main(
        [
            "--clips",
            str(clip),
            "--configs",
            "t0",
            "beam5_t0",
            "t0_rep",
            "--runs",
            "1",
            "--budget-s",
            "25",
        ]
    )
    budget_rows = rows(capsys.readouterr().out)
    assert [row[3] for row in budget_rows] == ["1", "skipped (budget)", "skipped (budget)"]
    assert budget_rows[1][4:] == ["-"] * (len(COLUMNS) - 4)


def test_main_requires_a_cap_for_capped_configs(fake_model: type[FakeModel], clip: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--clips", str(clip), "--configs", "t0_cap"])
