from pathlib import Path

import numpy as np
import pytest

from kotha.audio.convert import SAMPLE_RATE_HZ, load_mono_16k
from kotha.vad import WINDOW_SAMPLES
from kotha.vad.silero import SileroVad

SPEECH_PATH = Path(__file__).parent / "fixtures" / "speech_16k.wav"


@pytest.fixture(scope="module")
def vad() -> SileroVad:
    return SileroVad()


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    return load_mono_16k(SPEECH_PATH)


def windows(audio: np.ndarray) -> np.ndarray:
    count = audio.size // WINDOW_SAMPLES
    return audio[: count * WINDOW_SAMPLES].reshape(count, WINDOW_SAMPLES)


def probabilities(vad: SileroVad, audio: np.ndarray) -> np.ndarray:
    return np.array([vad.probability(window) for window in windows(audio)])


def test_silence_is_not_speech(vad: SileroVad) -> None:
    vad.reset()
    assert probabilities(vad, np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)).max() < 0.1


def test_noise_is_not_speech(vad: SileroVad) -> None:
    rms = 10 ** (-30 / 20)
    noise = (rms * np.random.default_rng(0).standard_normal(SAMPLE_RATE_HZ)).astype(np.float32)
    vad.reset()
    assert probabilities(vad, noise).max() < 0.5


def test_speech_is_detected(vad: SileroVad, speech: np.ndarray) -> None:
    vad.reset()
    assert probabilities(vad, speech).max() >= 0.5


def test_reset_restores_initial_state(vad: SileroVad, speech: np.ndarray) -> None:
    vad.reset()
    first_pass = probabilities(vad, speech)
    vad.reset()
    assert np.allclose(probabilities(vad, speech), first_pass)


def test_instances_do_not_share_state(vad: SileroVad, speech: np.ndarray) -> None:
    silence = np.zeros(speech.size, dtype=np.float32)
    talker, listener = SileroVad(), SileroVad()
    listener_probabilities = []
    for speech_window, silence_window in zip(windows(speech), windows(silence), strict=True):
        talker.probability(speech_window)
        listener_probabilities.append(listener.probability(silence_window))
    vad.reset()
    assert np.allclose(listener_probabilities, probabilities(vad, silence))


@pytest.mark.parametrize(
    "window",
    [
        np.zeros(WINDOW_SAMPLES - 1, dtype=np.float32),
        np.zeros(WINDOW_SAMPLES + 1, dtype=np.float32),
        np.zeros(WINDOW_SAMPLES, dtype=np.float64),
        np.zeros((1, WINDOW_SAMPLES), dtype=np.float32),
    ],
    ids=["511-samples", "513-samples", "float64", "2-d"],
)
def test_probability_rejects_wrong_input(vad: SileroVad, window: np.ndarray) -> None:
    with pytest.raises(ValueError):
        vad.probability(window)


def test_probability_is_a_float_between_0_and_1(vad: SileroVad, speech: np.ndarray) -> None:
    vad.reset()
    for window in windows(speech):
        probability = vad.probability(window)
        assert type(probability) is float
        assert 0.0 <= probability <= 1.0
