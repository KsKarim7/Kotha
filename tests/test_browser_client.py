import re
from collections.abc import Iterator
from contextlib import ExitStack

import pytest
from starlette.testclient import TestClient
from test_api_session import FakeAsr, FakePipeline, FakeTts, loud_probability

from kotha.api.app import STATIC_DIR, create_app
from kotha.api.session import CallServices
from kotha.dialogue.answer import HandoffReason

CLIENT_FILES = ("index.html", "call.js", "pcm-worklet.js", "call.css")


def fake_services(resources: ExitStack) -> CallServices:
    return CallServices(lambda: loud_probability, FakeAsr(), FakePipeline(), FakeTts())


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(fake_services)) as test_client:
        yield test_client


def test_the_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    for element in ('id="consent"', 'id="start"', 'id="stop"', 'id="transcript"'):
        assert element in response.text
    assert '<script type="module" src="/static/call.js">' in response.text


@pytest.mark.parametrize(
    ("name", "content_type"),
    [
        ("call.js", "text/javascript"),
        ("pcm-worklet.js", "text/javascript"),
        ("call.css", "text/css"),
    ],
)
def test_client_files_are_served(client: TestClient, name: str, content_type: str) -> None:
    response = client.get(f"/static/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)


def test_the_client_loads_nothing_from_another_origin() -> None:
    for name in CLIENT_FILES:
        text = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]", text), name


def test_the_client_follows_the_call_protocol() -> None:
    script = (STATIC_DIR / "call.js").read_text(encoding="utf-8")
    worklet = (STATIC_DIR / "pcm-worklet.js").read_text(encoding="utf-8")
    assert '"/v1/ready"' in script and "/v1/calls/stream" in script
    assert 'type: "start", sample_rate: 16000, consent: true' in script
    assert 'type: "stop"' in script
    assert "echoCancellation: true" in script
    assert "AbortSignal.timeout(READY_REQUEST_TIMEOUT_MS)" in script
    assert "numberOfOutputs: 0" in script
    assert script.index('type: "start"') < script.index("pending.sending = true")
    for event in ("vad", "transcript", "bot_text", "handoff", "reply_audio_end", "error"):
        assert f'event.type === "{event}"' in script
    assert "TARGET_RATE = 16000" in worklet and "FRAME_SAMPLES = 320" in worklet
    assert "Int16Array" in worklet


def test_the_page_explains_every_handoff_reason() -> None:
    script = (STATIC_DIR / "call.js").read_text(encoding="utf-8")
    for reason in HandoffReason:
        assert re.search(rf'^\s*{reason.value}:\s*"[^"]+",?$', script, re.MULTILINE), reason
    assert "HANDOFF_REASONS[event.reason]" in script


def test_the_page_waits_for_readiness_before_the_microphone() -> None:
    script = (STATIC_DIR / "call.js").read_text(encoding="utf-8")
    start = script[script.index("async function startCall") :]
    assert start.index("await waitUntilReady(pending)") < start.index("getUserMedia")


def test_server_text_is_inserted_as_text_only() -> None:
    script = (STATIC_DIR / "call.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script
    assert "textContent = text" in script
