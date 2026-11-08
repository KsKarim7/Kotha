import numpy as np

from kotha.audio.convert import SAMPLE_RATE_HZ, resample

MU = 255
TELEPHONY_SAMPLE_RATE_HZ = 8_000


def mulaw_roundtrip(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1, 1)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    quantised = np.sign(compressed) * np.round(np.abs(compressed) * 127) / 127
    decoded = np.sign(quantised) * np.expm1(np.abs(quantised) * np.log1p(MU)) / MU
    return decoded.astype(np.float32)


def simulate_telephony(audio_16k: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio_16k, -1, 1)
    narrowband = resample(clipped, SAMPLE_RATE_HZ, TELEPHONY_SAMPLE_RATE_HZ)
    companded = mulaw_roundtrip(narrowband)
    wideband = resample(companded, TELEPHONY_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ)
    return wideband[: audio_16k.size]
