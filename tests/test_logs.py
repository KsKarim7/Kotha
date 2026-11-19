import logging

import pytest

from kotha.logs import log_event


def test_allowed_fields_are_logged_in_a_stable_order(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="kotha")
    log_event("reply", lang="bn", call_id="call-1a2b3c4d", kind="answer", seconds=0.42)
    assert caplog.messages == ["event=reply call_id=call-1a2b3c4d kind=answer lang=bn seconds=0.42"]


@pytest.mark.parametrize("field", ["text", "transcript", "phone", "question"])
def test_fields_off_the_allow_list_are_refused(field: str) -> None:
    with pytest.raises(ValueError, match="not on the allow-list"):
        log_event("reply", **{field: "x"})


@pytest.mark.parametrize(
    "value",
    ["01700000009", "1234", "how do I replace my SIM", "আমার ব্যালেন্স", "x" * 65, ""],
)
def test_string_values_must_be_short_tokens_that_start_with_a_letter(value: str) -> None:
    with pytest.raises(ValueError, match="short token"):
        log_event("reply", reason=value)


def test_event_names_must_be_short_tokens() -> None:
    with pytest.raises(ValueError, match="short token"):
        log_event("caller said hello")
