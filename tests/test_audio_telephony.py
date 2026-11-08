import numpy as np
import pytest

from kotha.audio.convert import SAMPLE_RATE_HZ
from kotha.audio.telephony import TELEPHONY_SAMPLE_RATE_HZ, mulaw_roundtrip, simulate_telephony


def sine(frequency_hz: float, amplitude: float, sample_rate: int, size: int) -> np.ndarray:
    t = np.arange(size) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def test_mulaw_maps_zeros_to_zeros() -> None:
    assert np.array_equal(mulaw_roundtrip(np.zeros(100, dtype=np.float32)), np.zeros(100))


def test_mulaw_is_odd_symmetric() -> None:
    audio = np.random.default_rng(0).uniform(-1, 1, size=10_000).astype(np.float32)
    assert np.array_equal(mulaw_roundtrip(-audio), -mulaw_roundtrip(audio))


def test_mulaw_ramp_has_at_most_255_levels() -> None:
    ramp = np.linspace(-1, 1, 200_001, dtype=np.float32)
    assert np.unique(mulaw_roundtrip(ramp)).size <= 255


def test_mulaw_is_idempotent() -> None:
    audio = np.random.default_rng(1).uniform(-1, 1, size=10_000).astype(np.float32)
    once = mulaw_roundtrip(audio)
    assert np.allclose(mulaw_roundtrip(once), once, atol=1e-6)


def test_mulaw_clips_out_of_range_input() -> None:
    audio = np.linspace(-3, 3, 10_001, dtype=np.float32)
    companded = mulaw_roundtrip(audio)
    assert np.array_equal(companded, mulaw_roundtrip(np.clip(audio, -1, 1)))
    assert companded.min() >= -1 and companded.max() <= 1


@pytest.mark.parametrize("amplitude", [0.01, 0.1, 0.9])
def test_mulaw_snr_of_sine_is_at_least_30_db(amplitude: float) -> None:
    tone = sine(1_000, amplitude, TELEPHONY_SAMPLE_RATE_HZ, TELEPHONY_SAMPLE_RATE_HZ)
    noise = mulaw_roundtrip(tone) - tone
    assert 20 * np.log10(rms(tone) / rms(noise)) >= 30


def test_telephony_keeps_1khz_rms() -> None:
    tone = sine(1_000, 0.5, SAMPLE_RATE_HZ, SAMPLE_RATE_HZ)
    phone = simulate_telephony(tone)
    assert rms(phone[200:-200]) == pytest.approx(rms(tone[200:-200]), rel=0.05)


def test_telephony_removes_6khz() -> None:
    tone = sine(6_000, 0.5, SAMPLE_RATE_HZ, SAMPLE_RATE_HZ)
    phone = simulate_telephony(tone)
    assert rms(phone[200:-200]) < 0.01 * rms(tone[200:-200])


@pytest.mark.parametrize("size", [16_000, 16_001])
def test_telephony_keeps_length_and_dtype(size: int) -> None:
    phone = simulate_telephony(sine(1_000, 0.5, SAMPLE_RATE_HZ, size))
    assert phone.shape == (size,)
    assert phone.dtype == np.float32


def test_telephony_clips_out_of_range_input() -> None:
    audio = sine(1_000, 3.0, SAMPLE_RATE_HZ, SAMPLE_RATE_HZ)
    assert np.array_equal(simulate_telephony(audio), simulate_telephony(np.clip(audio, -1, 1)))
