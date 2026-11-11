import numpy as np
import torch
from silero_vad import load_silero_vad

from kotha.audio.convert import SAMPLE_RATE_HZ

WINDOW_SAMPLES = 512


class SileroVad:
    def __init__(self) -> None:
        self._model = load_silero_vad()

    def probability(self, window: np.ndarray) -> float:
        if window.dtype != np.float32 or window.shape != (WINDOW_SAMPLES,):
            raise ValueError(
                f"expected a float32 array of shape ({WINDOW_SAMPLES},), "
                f"got {window.dtype} {window.shape}"
            )
        with torch.no_grad():
            return float(self._model(torch.from_numpy(window), SAMPLE_RATE_HZ).item())

    def reset(self) -> None:
        self._model.reset_states()
