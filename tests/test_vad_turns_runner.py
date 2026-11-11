from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from kotha.audio.convert import SAMPLE_RATE_HZ, load_mono_16k
from kotha.vad import WINDOW_SAMPLES
from kotha.vad.silero import SileroVad
from kotha.vad.turns import TurnDetector
from vad_turns import find_turns, main

SPEECH_PATH = Path(__file__).parent / "fixtures" / "speech_16k.wav"


class ScriptedProbability:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.calls = 0

    def __call__(self, window: np.ndarray) -> float:
        self.calls += 1
        return self.probabilities[self.calls - 1]


def test_find_turns_flushes_a_turn_open_at_the_end() -> None:
    audio = np.repeat(np.arange(35, dtype=np.float32), WINDOW_SAMPLES)
    detector = TurnDetector(ScriptedProbability([0.0] * 15 + [0.9] * 20))
    turns = find_turns(audio, detector)
    assert [(turn.start_sample, turn.end_sample) for turn in turns] == [
        (15 * WINDOW_SAMPLES, 35 * WINDOW_SAMPLES)
    ]
    assert np.array_equal(turns[0].audio, audio[5 * WINDOW_SAMPLES :])


def test_find_turns_drops_a_trailing_partial_window() -> None:
    probability = ScriptedProbability([0.0] * 21)
    turns = find_turns(
        np.zeros(20 * WINDOW_SAMPLES + 100, dtype=np.float32), TurnDetector(probability)
    )
    assert turns == []
    assert probability.calls == 20


def test_find_turns_on_real_speech() -> None:
    speech = load_mono_16k(SPEECH_PATH)
    first_run = find_turns(speech, TurnDetector(SileroVad().probability))
    second_run = find_turns(speech, TurnDetector(SileroVad().probability))
    assert len(first_run) >= 1
    for turn in first_run:
        assert 0 <= turn.start_sample < turn.end_sample <= speech.size
    assert [(turn.start_sample, turn.end_sample) for turn in first_run] == [
        (turn.start_sample, turn.end_sample) for turn in second_run
    ]
    for first, second in zip(first_run, second_run, strict=True):
        assert np.array_equal(first.audio, second.audio)


def test_main_prints_one_line_per_turn(capsys: pytest.CaptureFixture[str]) -> None:
    main([str(SPEECH_PATH)])
    header, *rows = capsys.readouterr().out.splitlines()
    assert header == "start_s\tend_s\tduration_s"
    assert rows
    for row in rows:
        start_s, end_s, duration_s = (float(field) for field in row.split("\t"))
        assert end_s > start_s
        assert duration_s == pytest.approx(end_s - start_s, abs=0.01)


def test_main_reports_no_turns_in_silence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    silence_path = tmp_path / "silence.wav"
    sf.write(silence_path, np.zeros(SAMPLE_RATE_HZ, dtype=np.float32), SAMPLE_RATE_HZ)
    main([str(silence_path)])
    assert capsys.readouterr().out == "no turns found\n"
