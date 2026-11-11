import argparse
from pathlib import Path

import numpy as np

from kotha.audio.convert import SAMPLE_RATE_HZ, load_mono_16k
from kotha.vad import WINDOW_SAMPLES
from kotha.vad.silero import SileroVad
from kotha.vad.turns import SpeechEnded, TurnDetector


def find_turns(audio_16k: np.ndarray, detector: TurnDetector) -> list[SpeechEnded]:
    window_count = audio_16k.size // WINDOW_SAMPLES
    windows = audio_16k[: window_count * WINDOW_SAMPLES].reshape(window_count, WINDOW_SAMPLES)
    turns = []
    for window in windows:
        event = detector.push(window)
        if isinstance(event, SpeechEnded):
            turns.append(event)
    final_turn = detector.flush()
    if final_turn is not None:
        turns.append(final_turn)
    return turns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--hangover-ms", type=int, default=700)
    args = parser.parse_args(argv)
    detector = TurnDetector(SileroVad().probability, hangover_ms=args.hangover_ms)
    turns = find_turns(load_mono_16k(args.input), detector)
    if not turns:
        print("no turns found")
        return
    print("start_s\tend_s\tduration_s")
    for turn in turns:
        start_s = turn.start_sample / SAMPLE_RATE_HZ
        end_s = turn.end_sample / SAMPLE_RATE_HZ
        print(f"{start_s:.2f}\t{end_s:.2f}\t{end_s - start_s:.2f}")


if __name__ == "__main__":
    main()
