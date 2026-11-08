from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import chirp

from kotha.audio.convert import SAMPLE_RATE_HZ
from plot_spectrograms import main, plot_comparison


@pytest.fixture
def chirp_16k() -> np.ndarray:
    t = np.arange(2 * SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
    return (0.5 * chirp(t, f0=100, t1=2, f1=7_900, method="linear")).astype(np.float32)


def test_plot_comparison_writes_small_png(tmp_path: Path, chirp_16k: np.ndarray) -> None:
    png = tmp_path / "spectrogram.png"
    plot_comparison(chirp_16k, png)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.stat().st_size < 500_000


def test_main_writes_png_into_new_directory(tmp_path: Path, chirp_16k: np.ndarray) -> None:
    wav = tmp_path / "chirp.wav"
    sf.write(wav, chirp_16k, SAMPLE_RATE_HZ, subtype="FLOAT")
    png = tmp_path / "figures" / "spectrogram.png"
    main([str(wav), "--output", str(png)])
    assert png.exists()
