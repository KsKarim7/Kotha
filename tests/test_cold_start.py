import logging

import pytest

from cold_start import breakdown, startup_steps
from kotha.logs import logged_duration


def test_startup_steps_are_read_back_from_the_apps_own_log_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="kotha")
    for step in ("asr_load", "tts_load_bn"):
        with logged_duration(step):
            pass
    log = "\n".join(f"2026-11-23 10:00:00,000 INFO kotha {line}" for line in caplog.messages)
    steps = startup_steps(f"INFO:     Started server process [1]\n{log}\n")
    assert list(steps) == ["asr_load", "tts_load_bn"]
    assert all(0 <= seconds < 1 for seconds in steps.values())


def test_the_breakdown_names_what_the_steps_do_not_cover() -> None:
    parts = breakdown({"asr_load": 10.0, "tts_load_bn": 5.0}, 20.0, 26.5)
    assert parts["other_startup"] == pytest.approx(5.0)
    assert parts["before_startup"] == pytest.approx(6.5)
    assert (parts["startup_seconds"], parts["cold_start_seconds"]) == (20.0, 26.5)
