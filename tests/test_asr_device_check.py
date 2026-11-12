from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_device_check import ComputeTypeResult, check_compute_type, main
from kotha.asr.whisper import Transcript
from kotha.audio.convert import SAMPLE_RATE_HZ

SECRET_TEXT = "this transcript must never be printed"


def transcript(text: str) -> Transcript:
    return Transcript(text, "bn", -0.3, 0.1, 1.2, 5, ())


class FakeWhisperAsr:
    created: list[tuple[str, str, str]] = []
    load_error: Exception | None = None
    transcribe_error: Exception | None = None
    texts: list[str] = [SECRET_TEXT]
    languages: list[str | None] = []

    def __init__(self, model: str, *, device: str, compute_type: str) -> None:
        if FakeWhisperAsr.load_error is not None:
            raise FakeWhisperAsr.load_error
        FakeWhisperAsr.created.append((model, device, compute_type))

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None) -> Transcript:
        if FakeWhisperAsr.transcribe_error is not None:
            raise FakeWhisperAsr.transcribe_error
        FakeWhisperAsr.languages.append(language)
        texts = FakeWhisperAsr.texts
        return transcript(texts[(len(FakeWhisperAsr.languages) - 1) % len(texts)])


@pytest.fixture
def fake_asr(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperAsr]:
    FakeWhisperAsr.created = []
    FakeWhisperAsr.load_error = None
    FakeWhisperAsr.transcribe_error = None
    FakeWhisperAsr.texts = [SECRET_TEXT]
    FakeWhisperAsr.languages = []
    monkeypatch.setattr("asr_device_check.WhisperAsr", FakeWhisperAsr)
    return FakeWhisperAsr


def audio() -> np.ndarray:
    return np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)


def check(compute_type: str = "int8", runs: int = 3) -> ComputeTypeResult:
    return check_compute_type(audio(), "small", "cuda", compute_type, runs, lambda: 2048)


def test_successful_compute_type(fake_asr: type[FakeWhisperAsr]) -> None:
    result = check("int8_float16", runs=3)
    assert fake_asr.created == [("small", "cuda", "int8_float16")]
    assert fake_asr.languages == ["bn"] * 4
    assert (result.compute_type, result.ok, result.repeatable) == ("int8_float16", True, True)
    assert (result.gpu_mem_mb, result.error) == (2048, "")
    assert result.load_s is not None and result.load_s >= 0
    assert result.median_s is not None and result.median_s >= 0


def test_load_error_becomes_an_error_row(fake_asr: type[FakeWhisperAsr]) -> None:
    fake_asr.load_error = ValueError(
        "Requested float16 compute type, but the target device does not support it"
    )
    assert check("float16") == ComputeTypeResult(
        "float16",
        False,
        None,
        None,
        None,
        None,
        "ValueError: Requested float16 compute type, but the target device does not support it",
    )


def test_transcribe_error_becomes_an_error_row(fake_asr: type[FakeWhisperAsr]) -> None:
    fake_asr.transcribe_error = RuntimeError("CUDA failed with error\nout of memory")
    result = check()
    assert (result.ok, result.error) == (
        False,
        "RuntimeError: CUDA failed with error out of memory",
    )


def test_different_transcripts_are_not_repeatable(fake_asr: type[FakeWhisperAsr]) -> None:
    fake_asr.texts = ["warm-up", "first", "second", "third"]
    result = check(runs=3)
    assert (result.ok, result.repeatable) == (True, False)


def test_empty_text_is_not_ok(fake_asr: type[FakeWhisperAsr]) -> None:
    fake_asr.texts = [""]
    result = check()
    assert (result.ok, result.repeatable, result.error) == (False, True, "")


def test_main_prints_one_row_per_compute_type(
    fake_asr: type[FakeWhisperAsr],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("asr_device_check.gpu_memory_used_mb", lambda: 1500)
    wav = tmp_path / "speech.wav"
    sf.write(wav, audio(), SAMPLE_RATE_HZ)
    main([str(wav), "--model", "tiny", "--compute-types", "int8,float16", "--runs", "2"])
    header, *rows = capsys.readouterr().out.splitlines()
    assert header == "compute_type\tok\tload_s\tmedian_s\trepeatable\tgpu_mem_mb\terror"
    assert [row.split("\t")[0] for row in rows] == ["int8", "float16"]
    assert all(row.split("\t")[5] == "1500" for row in rows)
    assert fake_asr.created == [("tiny", "cuda", "int8"), ("tiny", "cuda", "float16")]
    assert SECRET_TEXT not in "\n".join(rows)


def test_main_on_cpu_does_not_read_gpu_memory(
    fake_asr: type[FakeWhisperAsr],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> int:
        raise AssertionError("nvidia-smi must not be read for --device cpu")

    monkeypatch.setattr("asr_device_check.gpu_memory_used_mb", fail)
    wav = tmp_path / "speech.wav"
    sf.write(wav, audio(), SAMPLE_RATE_HZ)
    main([str(wav), "--device", "cpu", "--compute-types", "int8", "--runs", "1"])
    row = capsys.readouterr().out.splitlines()[1].split("\t")
    assert (row[0], row[1], row[5]) == ("int8", "yes", "-")
