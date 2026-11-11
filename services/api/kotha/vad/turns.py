from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kotha.audio.convert import SAMPLE_RATE_HZ
from kotha.vad import WINDOW_SAMPLES


@dataclass(frozen=True)
class SpeechStarted:
    start_sample: int


@dataclass(frozen=True)
class SpeechEnded:
    start_sample: int
    end_sample: int
    audio: np.ndarray


class TurnDetector:
    def __init__(
        self,
        probability: Callable[[np.ndarray], float],
        *,
        start_threshold: float = 0.5,
        end_threshold: float = 0.35,
        min_speech_ms: int = 250,
        hangover_ms: int = 700,
        preroll_ms: int = 300,
    ) -> None:
        if end_threshold >= start_threshold:
            raise ValueError("end_threshold must be below start_threshold")
        if min(min_speech_ms, hangover_ms, preroll_ms) <= 0:
            raise ValueError("min_speech_ms, hangover_ms and preroll_ms must be positive")
        self._probability = probability
        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._min_speech_windows = self._to_windows(min_speech_ms)
        self._hangover_windows = self._to_windows(hangover_ms)
        self._preroll: deque[np.ndarray] = deque(maxlen=self._to_windows(preroll_ms))
        self._samples_pushed = 0
        self._turn: list[np.ndarray] | None = None
        self._start_sample = 0
        self._end_sample = 0
        self._speech_windows = 0
        self._silence_windows = 0
        self._confirmed = False

    @staticmethod
    def _to_windows(ms: int) -> int:
        return -(-ms * SAMPLE_RATE_HZ // (1000 * WINDOW_SAMPLES))

    def push(self, window: np.ndarray) -> SpeechStarted | SpeechEnded | None:
        if window.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"expected shape ({WINDOW_SAMPLES},), got {window.shape}")
        speech_probability = self._probability(window)
        stored_window = np.array(window, dtype=np.float32)
        position = self._samples_pushed
        self._samples_pushed += WINDOW_SAMPLES
        event = self._advance(stored_window, speech_probability, position)
        self._preroll.append(stored_window)
        return event

    def _advance(
        self, window: np.ndarray, speech_probability: float, position: int
    ) -> SpeechStarted | SpeechEnded | None:
        if self._turn is None:
            if speech_probability < self._start_threshold:
                return None
            self._turn = list(self._preroll)
            self._start_sample = position
            self._speech_windows = 0
            self._silence_windows = 0
            self._confirmed = False
        self._turn.append(window)

        if speech_probability >= self._end_threshold:
            self._speech_windows += 1
            self._silence_windows = 0
            self._end_sample = position + WINDOW_SAMPLES
            if not self._confirmed and self._speech_windows >= self._min_speech_windows:
                self._confirmed = True
                return SpeechStarted(self._start_sample)
            return None

        self._silence_windows += 1
        if self._silence_windows < self._hangover_windows:
            return None
        turn_windows, self._turn = self._turn, None
        if not self._confirmed:
            return None
        return SpeechEnded(self._start_sample, self._end_sample, np.concatenate(turn_windows))
