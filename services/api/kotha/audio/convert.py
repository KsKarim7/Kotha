from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE_HZ = 16_000


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype="<i2")
    return (samples / 32768).astype(np.float32)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    scaled = np.round(audio * 32768)
    return np.clip(scaled, -32768, 32767).astype("<i2").tobytes()


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return audio.astype(np.float32, copy=False)
    divisor = gcd(from_rate, to_rate)
    resampled = resample_poly(audio, to_rate // divisor, from_rate // divisor)
    return resampled.astype(np.float32)


def load_mono_16k(path: str | Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1, dtype=np.float32)
    return resample(mono, sample_rate, SAMPLE_RATE_HZ)
