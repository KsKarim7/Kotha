import numpy as np
import pytest

from kotha.vad import WINDOW_SAMPLES
from kotha.vad.turns import SpeechEnded, SpeechStarted, TurnDetector


class FakeProbability:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.windows: list[np.ndarray] = []

    def __call__(self, window: np.ndarray) -> float:
        self.windows.append(window)
        return self.probabilities[len(self.windows) - 1]


def window(index: int) -> np.ndarray:
    return np.full(WINDOW_SAMPLES, index, dtype=np.float32)


def ramp(first: int, last: int) -> np.ndarray:
    return np.repeat(np.arange(first, last + 1, dtype=np.float32), WINDOW_SAMPLES)


def run(probabilities: list[float], **settings: int) -> list[tuple]:
    detector = TurnDetector(FakeProbability(probabilities), **settings)
    events = []
    for index in range(len(probabilities)):
        event = detector.push(window(index))
        if isinstance(event, SpeechStarted):
            events.append(("start", index, event.start_sample))
        elif isinstance(event, SpeechEnded):
            first, last = int(event.audio[0]), int(event.audio[-1])
            assert event.audio.dtype == np.float32
            assert np.array_equal(event.audio, ramp(first, last))
            events.append(("end", index, event.start_sample, event.end_sample, first, last))
    return events


def test_silence_produces_no_events() -> None:
    assert run([0.0] * 50) == []


def test_one_utterance() -> None:
    probabilities = [0.0] * 15 + [0.9] * 20 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 22, 15 * WINDOW_SAMPLES),
        ("end", 56, 15 * WINDOW_SAMPLES, 35 * WINDOW_SAMPLES, 5, 56),
    ]


def test_preroll_is_clipped_at_stream_start() -> None:
    probabilities = [0.0] * 3 + [0.9] * 20 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 10, 3 * WINDOW_SAMPLES),
        ("end", 44, 3 * WINDOW_SAMPLES, 23 * WINDOW_SAMPLES, 0, 44),
    ]


def test_minimum_speech_of_eight_windows_makes_a_turn() -> None:
    probabilities = [0.0] * 15 + [0.9] * 8 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 22, 15 * WINDOW_SAMPLES),
        ("end", 44, 15 * WINDOW_SAMPLES, 23 * WINDOW_SAMPLES, 5, 44),
    ]


def test_seven_speech_windows_produce_no_events() -> None:
    assert run([0.0] * 15 + [0.9] * 7 + [0.0] * 30) == []


def test_speech_windows_need_not_be_consecutive() -> None:
    probabilities = [0.0] * 15 + [0.9] * 4 + [0.0] * 3 + [0.9] * 4 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 25, 15 * WINDOW_SAMPLES),
        ("end", 47, 15 * WINDOW_SAMPLES, 26 * WINDOW_SAMPLES, 5, 47),
    ]


def test_short_burst_is_ignored_before_a_real_utterance() -> None:
    probabilities = [0.0] * 10 + [0.9] * 3 + [0.0] * 25 + [0.9] * 20 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 45, 38 * WINDOW_SAMPLES),
        ("end", 79, 38 * WINDOW_SAMPLES, 58 * WINDOW_SAMPLES, 28, 79),
    ]


def test_probability_between_thresholds_never_starts_speech() -> None:
    assert run([0.0] * 5 + [0.4] * 30 + [0.0] * 30) == []


def test_probability_between_thresholds_keeps_speech_going() -> None:
    probabilities = [0.0] * 15 + [0.9] * 8 + [0.4] * 30 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 22, 15 * WINDOW_SAMPLES),
        ("end", 74, 15 * WINDOW_SAMPLES, 53 * WINDOW_SAMPLES, 5, 74),
    ]


def test_hangover_bridges_a_gap_of_21_silent_windows() -> None:
    probabilities = [0.0] * 15 + [0.9] * 10 + [0.0] * 21 + [0.9] * 10 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 22, 15 * WINDOW_SAMPLES),
        ("end", 77, 15 * WINDOW_SAMPLES, 56 * WINDOW_SAMPLES, 5, 77),
    ]


def test_gap_of_22_silent_windows_splits_the_turn() -> None:
    probabilities = [0.0] * 15 + [0.9] * 10 + [0.0] * 22 + [0.9] * 10 + [0.0] * 30
    assert run(probabilities) == [
        ("start", 22, 15 * WINDOW_SAMPLES),
        ("end", 46, 15 * WINDOW_SAMPLES, 25 * WINDOW_SAMPLES, 5, 46),
        ("start", 54, 47 * WINDOW_SAMPLES),
        ("end", 78, 47 * WINDOW_SAMPLES, 57 * WINDOW_SAMPLES, 37, 78),
    ]


@pytest.mark.parametrize(("hangover_ms", "end_push"), [(500, 40), (700, 46), (900, 53)])
def test_hangover_ends_on_the_last_silent_window(hangover_ms: int, end_push: int) -> None:
    probabilities = [0.0] * 15 + [0.9] * 10 + [0.0] * 40
    assert run(probabilities, hangover_ms=hangover_ms) == [
        ("start", 22, 15 * WINDOW_SAMPLES),
        ("end", end_push, 15 * WINDOW_SAMPLES, 25 * WINDOW_SAMPLES, 5, end_push),
    ]


def test_two_utterances_have_separate_audio() -> None:
    probabilities = [0.0] * 15 + [0.9] * 10 + [0.0] * 30 + [0.9] * 12 + [0.0] * 30
    detector = TurnDetector(FakeProbability(probabilities))
    events = [detector.push(window(index)) for index in range(len(probabilities))]
    turns = [event for event in events if isinstance(event, SpeechEnded)]
    assert [index for index, event in enumerate(events) if event is not None] == [22, 46, 62, 88]
    assert [(turn.start_sample, turn.end_sample) for turn in turns] == [
        (15 * WINDOW_SAMPLES, 25 * WINDOW_SAMPLES),
        (55 * WINDOW_SAMPLES, 67 * WINDOW_SAMPLES),
    ]
    assert np.array_equal(turns[0].audio, ramp(5, 46))
    assert np.array_equal(turns[1].audio, ramp(45, 88))
    assert not np.shares_memory(turns[0].audio, turns[1].audio)


def test_probability_is_called_once_per_window() -> None:
    probabilities = [0.0] * 15 + [0.9] * 20 + [0.0] * 30
    fake = FakeProbability(probabilities)
    detector = TurnDetector(fake)
    pushed = [window(index) for index in range(len(probabilities))]
    for pushed_window in pushed:
        detector.push(pushed_window)
    assert len(fake.windows) == len(pushed)
    assert all(seen is sent for seen, sent in zip(fake.windows, pushed, strict=True))


def test_turn_audio_survives_a_reused_window_buffer() -> None:
    probabilities = [0.0] * 15 + [0.9] * 20 + [0.0] * 30
    detector = TurnDetector(FakeProbability(probabilities))
    buffer = np.empty(WINDOW_SAMPLES, dtype=np.float32)
    turns = []
    for index in range(len(probabilities)):
        buffer[:] = index
        event = detector.push(buffer)
        if isinstance(event, SpeechEnded):
            turns.append(event)
    assert len(turns) == 1
    assert np.array_equal(turns[0].audio, ramp(5, 56))


@pytest.mark.parametrize("size", [WINDOW_SAMPLES - 1, WINDOW_SAMPLES + 1])
def test_push_rejects_wrong_window_size(size: int) -> None:
    detector = TurnDetector(FakeProbability([0.0]))
    with pytest.raises(ValueError):
        detector.push(np.zeros(size, dtype=np.float32))


def test_end_threshold_must_be_below_start_threshold() -> None:
    with pytest.raises(ValueError):
        TurnDetector(FakeProbability([]), start_threshold=0.5, end_threshold=0.5)


@pytest.mark.parametrize("setting", ["min_speech_ms", "hangover_ms", "preroll_ms"])
def test_durations_must_be_positive(setting: str) -> None:
    with pytest.raises(ValueError):
        TurnDetector(FakeProbability([]), **{setting: 0})


def push_all(detector: TurnDetector, first: int, last: int) -> list[tuple]:
    events = []
    for index in range(first, last + 1):
        event = detector.push(window(index))
        if event is not None:
            events.append((type(event).__name__, index))
    return events


def test_flush_closes_a_turn_when_the_stream_ends_mid_speech() -> None:
    probabilities = [0.0] * 15 + [0.9] * 20
    assert run(probabilities) == [("start", 22, 15 * WINDOW_SAMPLES)]
    detector = TurnDetector(FakeProbability(probabilities))
    push_all(detector, 0, 34)
    turn = detector.flush()
    assert isinstance(turn, SpeechEnded)
    assert (turn.start_sample, turn.end_sample) == (15 * WINDOW_SAMPLES, 35 * WINDOW_SAMPLES)
    assert np.array_equal(turn.audio, ramp(5, 34))


def test_flush_keeps_trailing_silence_but_ends_at_the_last_speech() -> None:
    probabilities = [0.0] * 15 + [0.9] * 20 + [0.0] * 10
    detector = TurnDetector(FakeProbability(probabilities))
    assert push_all(detector, 0, 44) == [("SpeechStarted", 22)]
    turn = detector.flush()
    assert isinstance(turn, SpeechEnded)
    assert (turn.start_sample, turn.end_sample) == (15 * WINDOW_SAMPLES, 35 * WINDOW_SAMPLES)
    assert np.array_equal(turn.audio, ramp(5, 44))


def test_flush_drops_an_unconfirmed_candidate() -> None:
    detector = TurnDetector(FakeProbability([0.0] * 15 + [0.9] * 5))
    assert push_all(detector, 0, 19) == []
    assert detector.flush() is None


def test_flush_when_idle_returns_none() -> None:
    assert TurnDetector(FakeProbability([])).flush() is None
    detector = TurnDetector(FakeProbability([0.0] * 20))
    push_all(detector, 0, 19)
    assert detector.flush() is None


def test_second_flush_returns_none() -> None:
    detector = TurnDetector(FakeProbability([0.0] * 15 + [0.9] * 20))
    push_all(detector, 0, 34)
    assert isinstance(detector.flush(), SpeechEnded)
    assert detector.flush() is None


def test_pushing_after_a_flush_continues_the_stream() -> None:
    probabilities = [0.0] * 15 + [0.9] * 20 + [0.0] * 5 + [0.9] * 10 + [0.0] * 30
    detector = TurnDetector(FakeProbability(probabilities))
    push_all(detector, 0, 34)
    assert isinstance(detector.flush(), SpeechEnded)
    events = []
    for index in range(35, 80):
        event = detector.push(window(index))
        if event is not None:
            events.append((index, event))
    assert [index for index, _ in events] == [47, 71]
    started, ended = events[0][1], events[1][1]
    assert isinstance(started, SpeechStarted) and isinstance(ended, SpeechEnded)
    assert started.start_sample == 40 * WINDOW_SAMPLES
    assert (ended.start_sample, ended.end_sample) == (40 * WINDOW_SAMPLES, 50 * WINDOW_SAMPLES)
    assert np.array_equal(ended.audio, ramp(30, 71))
