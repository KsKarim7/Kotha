from itertools import islice

import numpy as np
import pytest

from kotha.audio.convert import SAMPLE_RATE_HZ
from latency import (
    FRAME_SAMPLES,
    LEAD_SILENCE_SECONDS,
    REPO_ROOT,
    STAGES,
    TurnResult,
    frame_clock,
    main,
    speech_frames,
    summarise,
    turn_stages,
    without_trailing_silence,
)


def test_speech_frames_are_20_ms_of_pcm16_after_a_short_lead_silence() -> None:
    clip = np.full(SAMPLE_RATE_HZ, 0.5, dtype=np.float32)
    frames = speech_frames(clip)
    lead_samples = int(LEAD_SILENCE_SECONDS * SAMPLE_RATE_HZ)
    assert FRAME_SAMPLES == 320
    assert sum(len(frame) for frame in frames) == 2 * (lead_samples + SAMPLE_RATE_HZ)
    assert all(len(frame) == 2 * FRAME_SAMPLES for frame in frames[:-1])
    assert frames[0] == b"\x00\x00" * FRAME_SAMPLES


def test_trailing_silence_is_cut_so_the_last_frame_ends_with_speech() -> None:
    clip = np.concatenate(
        [np.full(1_000, 0.5, dtype=np.float32), np.full(4_000, 0.001, dtype=np.float32)]
    )
    assert len(without_trailing_silence(clip)) == 1_000
    assert len(without_trailing_silence(np.zeros(500, dtype=np.float32))) == 500
    frames = speech_frames(clip)
    lead_samples = int(LEAD_SILENCE_SECONDS * SAMPLE_RATE_HZ)
    assert sum(len(frame) for frame in frames) == 2 * (lead_samples + 1_000)


def test_turn_stages_measure_from_the_true_end_of_speech_and_from_the_vad_decision() -> None:
    timing = {
        "asr_seconds": 0.8,
        "retrieval_seconds": 0.2,
        "llm_seconds": 1.4,
        "tts_first_audio_seconds": 0.5,
    }
    stages = turn_stages(speech_end=10.0, vad_end=10.75, first_audio=14.0, timing=timing)
    assert stages == pytest.approx(
        {
            "vad": 0.75,
            "asr": 0.8,
            "retrieval": 0.2,
            "llm": 1.4,
            "tts_first_audio": 0.5,
            "ttfa": 4.0,
            "ttfa_from_vad": 3.25,
        }
    )


def test_server_stages_that_did_not_run_are_left_out() -> None:
    timing = {"asr_seconds": 0.8, "retrieval_seconds": None, "llm_seconds": None}
    stages = turn_stages(1.0, 1.7, 3.0, timing)
    assert set(stages) == {"vad", "asr", "ttfa", "ttfa_from_vad"}


def test_summary_gives_count_p50_and_p95_per_stage_without_split_turns() -> None:
    results = [
        TurnResult(f"q{index}", "bn", "answer", stages={"ttfa": float(index), "asr": 1.0})
        for index in range(1, 21)
    ]
    results.append(TurnResult("split", "bn", "answer", split=True, stages={"ttfa": 99.0}))
    summary = summarise(results)
    assert set(summary) == {"ttfa", "asr"}
    count, p50, p95 = summary["ttfa"]
    assert count == 20
    assert p50 == pytest.approx(10.5)
    assert p95 == pytest.approx(19.05)
    assert summary["asr"] == (20, 1.0, 1.0)
    assert set(summary) <= set(STAGES)


def test_frame_clock_paces_one_frame_every_20_ms() -> None:
    assert list(islice(frame_clock(5.0), 3)) == pytest.approx([5.0, 5.02, 5.04])


def test_main_refuses_an_output_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    output = REPO_ROOT / "eval" / "latency.csv"
    monkeypatch.setattr("sys.argv", ["latency.py", "--output", str(output)])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    assert not output.exists()
