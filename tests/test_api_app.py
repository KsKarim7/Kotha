from collections.abc import Callable
from contextlib import ExitStack

import numpy as np
import pytest
from starlette.testclient import TestClient
from test_api_session import FakeAsr, FakePipeline, FakeTts, loud_probability

from kotha.api.app import create_app, warm_up
from kotha.api.session import CallServices
from kotha.asr.whisper import Transcript
from kotha.tts.service import TtsService
from kotha.vad import WINDOW_SAMPLES


class RecordingAsr:
    def __init__(self, steps: list[str]) -> None:
        self.steps = steps
        self.audio_lengths: list[int] = []

    def transcribe(self, audio_16k: np.ndarray) -> Transcript:
        self.steps.append("asr")
        self.audio_lengths.append(len(audio_16k))
        return Transcript("", "bn", 0.0, 0.0, 1.0, 1, ())


class RecordingPipeline:
    def __init__(self, steps: list[str]) -> None:
        self.steps = steps

    def warm_up(self) -> None:
        self.steps.append("search")


class ToneVoice:
    sampling_rate = 16_000

    def __init__(self, steps: list[str]) -> None:
        self.steps = steps
        self.texts: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.steps.append("tts")
        self.texts.append(text)
        return np.full(800, 0.3, dtype=np.float32)


def test_warm_up_runs_vad_then_asr_on_synthetic_speech_then_one_search() -> None:
    steps: list[str] = []
    voice = ToneVoice(steps)
    asr = RecordingAsr(steps)

    def probability(window: np.ndarray) -> float:
        steps.append(f"vad {len(window)}")
        return 0.0

    def new_vad() -> Callable[[np.ndarray], float]:
        steps.append("load vad")
        return probability

    warm_up(new_vad, asr, RecordingPipeline(steps), TtsService({"bn": voice}))
    assert steps == ["load vad", f"vad {WINDOW_SAMPLES}", "tts", "asr", "search"]
    assert voice.texts == ["প্যাকের মেয়াদ ত্রিশ দিন"]
    assert asr.audio_lengths == [800]


def test_ready_answers_only_after_the_services_are_loaded() -> None:
    loads: list[str] = []

    def load(resources: ExitStack) -> CallServices:
        loads.append("loaded")
        return CallServices(lambda: loud_probability, FakeAsr(), FakePipeline(), FakeTts())

    with TestClient(create_app(load)) as client:
        assert loads == ["loaded"]
        response = client.get("/v1/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert isinstance(body["startup_seconds"], float) and body["startup_seconds"] >= 0


def test_a_failed_warm_up_stops_startup_and_releases_resources() -> None:
    released: list[str] = []

    def failing_load(resources: ExitStack) -> CallServices:
        resources.callback(released.append, "database")
        raise RuntimeError("warm-up failed")

    with pytest.raises(RuntimeError, match="warm-up failed"):
        with TestClient(create_app(failing_load)):
            pass
    assert released == ["database"]
