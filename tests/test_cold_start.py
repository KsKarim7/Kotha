import logging

import pytest

from cold_start import EXPECTED_STEPS, breakdown, startup_steps
from kotha.logs import logged_duration


def logged_steps(caplog: pytest.LogCaptureFixture, steps: tuple[str, ...]) -> str:
    caplog.set_level(logging.INFO, logger="kotha")
    for step in steps:
        with logged_duration(step):
            pass
    lines = [f"2026-11-23 10:00:00,000 INFO kotha {message}" for message in caplog.messages]
    return "\n".join(["INFO:     Started server process [1]", *lines, ""])


def test_startup_steps_are_read_back_from_the_apps_own_log_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    steps = startup_steps(logged_steps(caplog, (*EXPECTED_STEPS, "tts_load_x.y-z")))
    assert list(steps) == [*EXPECTED_STEPS, "tts_load_x.y-z"]
    assert all(0 <= seconds < 1 for seconds in steps.values())


def test_a_step_missing_from_the_log_is_an_error_not_unexplained_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match="tts_prompts_en"):
        startup_steps(logged_steps(caplog, EXPECTED_STEPS[:-5] + EXPECTED_STEPS[-4:]))


def test_the_breakdown_names_what_the_steps_do_not_cover() -> None:
    parts = breakdown({"asr_load": 10.0, "tts_load_bn": 5.0}, 20.0, 26.5)
    assert parts["other_startup"] == pytest.approx(5.0)
    assert parts["before_startup"] == pytest.approx(6.5)
    assert (parts["startup_seconds"], parts["cold_start_seconds"]) == (20.0, 26.5)
