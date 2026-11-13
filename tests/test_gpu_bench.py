from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_bench import (
    COLUMNS,
    GpuSample,
    GpuSummary,
    StageRun,
    decode_ms,
    decode_ms_per_token,
    format_row,
    main,
    parse_sample,
    samples_between,
    summarise_samples,
    transcribe_options,
)
from kotha.asr.whisper import DECODING_OPTIONS

NOT_ACTIVE = "Not Active"


def sample(
    at: float,
    pstate: str = "P0",
    sm_mhz: float | None = 1785.0,
    power_w: float | None = 30.0,
    util_pct: float | None = 90.0,
    temp_c: float | None = 60.0,
    reasons: tuple[str, ...] = (),
    *,
    mem_mhz: float | None = 6000.0,
) -> GpuSample:
    return GpuSample(
        at, pstate, sm_mhz, 1785.0, mem_mhz, 6001.0, power_w, util_pct, temp_c, reasons
    )


def test_parse_sample_reads_values_and_active_reasons() -> None:
    line = (
        f"P0, 1785, 1785, 6000, 6001, 30.50, 97, 66, {NOT_ACTIVE}, Active, {NOT_ACTIVE}, "
        f"{NOT_ACTIVE}, Active, {NOT_ACTIVE}\n"
    )
    assert parse_sample(line, 1.5) == GpuSample(
        1.5,
        "P0",
        1785.0,
        1785.0,
        6000.0,
        6001.0,
        30.5,
        97.0,
        66.0,
        ("sw_power_cap", "sw_thermal_slowdown"),
    )


def test_parse_sample_turns_not_available_into_none() -> None:
    line = f"P8, 375, 1785, [N/A], [N/A], [N/A], 0, 53, Active, {', '.join([NOT_ACTIVE] * 5)}"
    parsed = parse_sample(line, 0.0)
    assert (
        parsed.pstate,
        parsed.sm_mhz,
        parsed.mem_mhz,
        parsed.mem_mhz_limit,
        parsed.power_w,
        parsed.reasons,
    ) == ("P8", 375.0, None, None, None, ("gpu_idle",))


def test_summarise_samples_takes_medians_maxima_and_any_active_reason() -> None:
    samples = [
        sample(0.0, "P0", 1785.0, 30.0, 90.0, 60.0, mem_mhz=6000.0),
        sample(0.1, "P2", 1500.0, 20.0, 70.0, 65.0, ("sw_power_cap",), mem_mhz=810.0),
        sample(0.2, "P0", 1600.0, None, 80.0, 62.0, mem_mhz=5000.0),
    ]
    assert summarise_samples(samples) == GpuSummary(
        samples=3,
        pstates=("P0", "P2"),
        sm_mhz_median=1600.0,
        sm_mhz_max=1785.0,
        sm_mhz_limit=1785.0,
        mem_mhz_median=5000.0,
        mem_mhz_max=6000.0,
        mem_mhz_limit=6001.0,
        power_w_median=25.0,
        power_w_max=30.0,
        util_pct_median=80.0,
        temp_c_max=65.0,
        reasons=("sw_power_cap",),
    )


def test_summarise_samples_of_nothing_is_none() -> None:
    assert summarise_samples([]) is None


def test_samples_between_keeps_only_the_stage_window() -> None:
    samples = [sample(0.9), sample(1.0), sample(1.5), sample(2.0), sample(2.1)]
    assert [s.at for s in samples_between(samples, 1.0, 2.0)] == [1.0, 1.5, 2.0]


def test_decode_time_per_token() -> None:
    assert decode_ms(1000.0, 300.0, 20.0) == 680.0
    assert decode_ms_per_token(680.0, 170) == 4.0
    assert decode_ms_per_token(680.0, 0) is None


def test_format_row_for_a_sampled_cuda_stage() -> None:
    runs = [StageRun(0.0, 1.0, 1000.0, 1100.0), StageRun(2.0, 3.0, 1200.0, 1300.0)]
    samples = [
        sample(0.5, "P0", 1785.0, 30.0, 90.0, 60.0, mem_mhz=6000.0),
        sample(2.5, "P2", 1505.0, 20.0, 70.0, 65.0, ("sw_power_cap",), mem_mhz=810.0),
        sample(5.0, "P8", 375.0, 5.0, 0.0, 50.0, ("gpu_idle",), mem_mhz=405.0),
    ]
    row = format_row(
        "int8_float16", "whole", runs, samples, "1238/4096", [170, 170], [680.0, 850.0]
    )
    assert row.split("\t") == [
        "int8_float16",
        "whole",
        "2",
        "1100.0",
        "1200.0",
        "1000,1200",
        "1200.0",
        "1.09",
        "170",
        "765.0",
        "4.50",
        "1238/4096",
        "2",
        "P0,P2",
        "1645",
        "1785",
        "1785",
        "3405",
        "6000",
        "6001",
        "25.0",
        "30.0",
        "80",
        "65",
        "sw_power_cap",
        "1785,1505",
        "6000,810",
        "30.0,20.0",
        "60,65",
        "none,sw_power_cap",
    ]
    assert len(row.split("\t")) == len(COLUMNS) == 30


def test_memory_clock_columns_follow_the_sm_columns() -> None:
    limit = COLUMNS.index("sm_mhz_limit")
    assert COLUMNS[limit : limit + 5] == (
        "sm_mhz_limit",
        "mem_mhz_median",
        "mem_mhz_max",
        "mem_mhz_limit",
        "power_w_median",
    )
    runs = COLUMNS.index("sm_mhz_runs")
    assert COLUMNS[runs : runs + 3] == ("sm_mhz_runs", "mem_mhz_runs", "power_w_runs")


def test_format_row_without_samples_prints_dashes_for_the_gpu() -> None:
    row = format_row("int8", "encoder", [StageRun(0.0, 1.0, 400.0, 1600.0)], [], None)
    cells = row.split("\t")
    assert cells[:8] == ["int8", "encoder", "1", "400.0", "400.0", "400", "1600.0", "4.00"]
    assert cells[8:] == ["-"] * (len(COLUMNS) - 8)


class FakeFeatureExtractor:
    nb_max_frames = 3_000

    def __call__(self, audio_16k: np.ndarray) -> np.ndarray:
        return np.zeros((80, 100), dtype=np.float32)


class FakeWhisperModel:
    instances: list["FakeWhisperModel"] = []

    def __init__(self, model: str, *, device: str, compute_type: str) -> None:
        self.settings = (model, device, compute_type)
        self.feature_extractor = FakeFeatureExtractor()
        self.encoded: list[tuple[int, ...]] = []
        self.transcribe_calls: list[dict] = []
        FakeWhisperModel.instances.append(self)

    def encode(self, window: np.ndarray) -> None:
        self.encoded.append(window.shape)

    def transcribe(
        self, audio_16k: np.ndarray, **options: object
    ) -> tuple[Iterator[SimpleNamespace], None]:
        self.transcribe_calls.append(options)
        return iter([SimpleNamespace(text="secret words", tokens=[1, 2, 3])]), None


@pytest.fixture
def fake_cpu_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_subprocess(*args: object, **kwargs: object) -> None:
        raise AssertionError("nvidia-smi must not run on cpu")

    FakeWhisperModel.instances = []
    monkeypatch.setattr("gpu_bench.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr("gpu_bench.load_mono_16k", lambda path: np.zeros(16_000, np.float32))
    monkeypatch.setattr("gpu_bench.subprocess.Popen", no_subprocess)
    monkeypatch.setattr("gpu_bench.subprocess.run", no_subprocess)


def test_main_on_cpu_runs_every_stage_without_the_gpu(
    fake_cpu_run: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--device", "cpu", "--compute-types", "int8", "--clip", "k8.wav", "--runs", "2"])
    output = capsys.readouterr().out
    header, *rows = output.splitlines()
    assert header.split("\t") == list(COLUMNS)
    assert [row.split("\t")[:3] for row in rows] == [
        ["int8", "features", "2"],
        ["int8", "encoder", "2"],
        ["int8", "whole", "2"],
    ]
    assert all(row.split("\t")[11:] == ["-"] * (len(COLUMNS) - 11) for row in rows)
    assert rows[2].split("\t")[8] == "3"
    assert "secret" not in output
    [model] = FakeWhisperModel.instances
    assert model.settings == ("small", "cpu", "int8")
    assert model.encoded == [(80, 3_000)] * 2
    assert model.transcribe_calls == [transcribe_options()] * 3


def test_transcribe_options_match_the_wrapper() -> None:
    assert transcribe_options() == {
        "language": "bn",
        "beam_size": 1,
        "initial_prompt": None,
        "word_timestamps": False,
        **DECODING_OPTIONS,
    }


def test_main_runs_only_the_selected_stages(
    fake_cpu_run: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--device", "cpu", "--compute-types", "int8", "--clip", "k8.wav", "--stages", "encoder"])
    [_, row] = capsys.readouterr().out.splitlines()
    cells = row.split("\t")
    assert (cells[1], cells[2], cells[8], cells[9], cells[10]) == ("encoder", "5", "-", "-", "-")
    assert FakeWhisperModel.instances[0].transcribe_calls == [transcribe_options()]
