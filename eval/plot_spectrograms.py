import argparse
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from kotha.audio.convert import SAMPLE_RATE_HZ, load_mono_16k
from kotha.audio.telephony import simulate_telephony


def plot_comparison(audio_16k: np.ndarray, output_path: Path) -> None:
    phone_16k = simulate_telephony(audio_16k)
    fig = Figure(figsize=(8, 6), layout="constrained")
    original_ax, phone_ax = fig.subplots(2, 1, sharex=True, sharey=True)
    viridis = colormaps["viridis"]
    window = {
        "NFFT": 512,
        "Fs": SAMPLE_RATE_HZ,
        "noverlap": 384,
        "cmap": viridis.with_extremes(bad=viridis(0)),
    }

    spectrum, _, _, original_image = original_ax.specgram(audio_16k, **window)
    vmax = 10 * np.log10(spectrum.max())
    vmin = vmax - 80
    original_image.set_clim(vmin, vmax)
    # Mu-law maps near-silence to exact zero, which specgram draws as -inf dB.
    with np.errstate(divide="ignore"):
        phone_ax.specgram(phone_16k, vmin=vmin, vmax=vmax, **window)

    original_ax.set_title("Original (16 kHz)")
    phone_ax.set_title("After simulate_telephony (8 kHz mu-law, back to 16 kHz)")
    phone_ax.set_xlabel("Time (s)")
    for ax in (original_ax, phone_ax):
        ax.axhline(4_000, color="white", linestyle="--", linewidth=1)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, SAMPLE_RATE_HZ / 2)
    fig.colorbar(original_image, ax=[original_ax, phone_ax], label="Power (dB)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=72)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plot_comparison(load_mono_16k(args.input), args.output)


if __name__ == "__main__":
    main()
