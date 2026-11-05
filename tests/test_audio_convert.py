from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from kotha.audio.convert import (
    SAMPLE_RATE_HZ,
    float_to_pcm16,
    load_mono_16k,
    pcm16_to_float,
    resample,
)


def sine(frequency_hz: float, sample_rate: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def test_all_int16_values_round_trip_exactly() -> None:
    pcm = np.arange(-32768, 32768, dtype="<i2").tobytes()
    assert float_to_pcm16(pcm16_to_float(pcm)) == pcm


def test_pcm16_to_float_is_little_endian() -> None:
    np.testing.assert_array_equal(pcm16_to_float(b"\x01\x00"), [1 / 32768])


def test_float_to_pcm16_is_little_endian() -> None:
    assert float_to_pcm16(np.array([1 / 32768], dtype=np.float32)) == b"\x01\x00"


def test_float_to_pcm16_clips_to_int16_range() -> None:
    pcm = float_to_pcm16(np.array([1.5, -1.5, 1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(np.frombuffer(pcm, dtype="<i2"), [32767, -32768, 32767, -32768])


def test_pcm16_to_float_rejects_odd_length() -> None:
    with pytest.raises(ValueError):
        pcm16_to_float(b"\x01\x00\x02")


def test_pcm16_wav_loads_losslessly(tmp_path: Path) -> None:
    samples = np.random.default_rng(0).integers(-32768, 32768, size=16_000, dtype=np.int16)
    path = tmp_path / "mono.wav"
    sf.write(path, samples, SAMPLE_RATE_HZ, subtype="PCM_16")
    assert float_to_pcm16(load_mono_16k(path)) == samples.astype("<i2").tobytes()


def test_stereo_file_loads_as_channel_mean(tmp_path: Path) -> None:
    left = sine(440, SAMPLE_RATE_HZ)
    right = sine(1_000, SAMPLE_RATE_HZ)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([left, right]), SAMPLE_RATE_HZ, subtype="FLOAT")
    np.testing.assert_allclose(load_mono_16k(path), (left + right) / 2, atol=1e-7)


@pytest.mark.parametrize("sample_rate", [48_000, 44_100])
def test_resampled_file_has_16000_samples(tmp_path: Path, sample_rate: int) -> None:
    path = tmp_path / "input.wav"
    sf.write(path, sine(1_000, sample_rate), sample_rate, subtype="FLOAT")
    assert load_mono_16k(path).shape == (16_000,)


@pytest.mark.parametrize("sample_rate", [48_000, 44_100])
def test_resampled_sine_keeps_frequency_and_rms(tmp_path: Path, sample_rate: int) -> None:
    tone = sine(1_000, sample_rate)
    path = tmp_path / "input.wav"
    sf.write(path, tone, sample_rate, subtype="FLOAT")
    loaded = load_mono_16k(path)

    frequencies = np.fft.rfftfreq(loaded.size, d=1 / SAMPLE_RATE_HZ)
    peak_hz = frequencies[np.argmax(np.abs(np.fft.rfft(loaded)))]
    assert peak_hz == 1_000
    assert rms(loaded[100:-100]) == pytest.approx(rms(tone), rel=0.05)


def test_mono_16k_file_loads_unchanged(tmp_path: Path) -> None:
    tone = sine(1_000, SAMPLE_RATE_HZ)
    path = tmp_path / "mono.wav"
    sf.write(path, tone, SAMPLE_RATE_HZ, subtype="FLOAT")
    assert np.array_equal(load_mono_16k(path), tone)


def test_outputs_are_float32(tmp_path: Path) -> None:
    path = tmp_path / "input.wav"
    sf.write(path, sine(1_000, 48_000), 48_000, subtype="PCM_16")
    assert pcm16_to_float(b"\x01\x00").dtype == np.float32
    assert resample(np.zeros(480, dtype=np.float64), 48_000, 16_000).dtype == np.float32
    assert load_mono_16k(path).dtype == np.float32


def test_resample_with_equal_rates_returns_input() -> None:
    tone = sine(1_000, SAMPLE_RATE_HZ)
    resampled = resample(tone, SAMPLE_RATE_HZ, SAMPLE_RATE_HZ)
    assert resampled.dtype == np.float32
    assert np.array_equal(resampled, tone)
