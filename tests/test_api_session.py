import json
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient, WebSocketTestSession

from kotha.api.app import create_app
from kotha.api.session import (
    MAX_FRAME_BYTES,
    CallServices,
    frame_error,
    unusable_reason,
)
from kotha.asr.whisper import Transcript
from kotha.audio.convert import SAMPLE_RATE_HZ, float_to_pcm16
from kotha.dialogue.answer import (
    FIXED_REPLIES,
    GREETING,
    BotReply,
    HandoffReason,
    ReplyKind,
    Source,
)
from kotha.dialogue.routing import Language

FRAME_SAMPLES = SAMPLE_RATE_HZ // 10
QUESTION = "how do I replace my SIM"


def loud_probability(window: np.ndarray) -> float:
    return 0.9 if float(np.abs(window).mean()) > 0.1 else 0.0


def transcript(text: str = QUESTION, ratio: float = 1.2, tokens: int = 12) -> Transcript:
    return Transcript(
        text=text,
        language="en",
        worst_avg_logprob=-0.2,
        max_no_speech_prob=0.0,
        max_compression_ratio=ratio,
        max_segment_tokens=tokens,
        words=(),
    )


class FakeAsr:
    def __init__(self, result: Transcript | None = None) -> None:
        self.result = result or transcript()
        self.turn_seconds: list[float] = []

    def transcribe(self, audio_16k: np.ndarray) -> Transcript:
        self.turn_seconds.append(len(audio_16k) / SAMPLE_RATE_HZ)
        return self.result


class FakePipeline:
    def __init__(self, release: threading.Event | None = None) -> None:
        self.questions: list[str] = []
        self.release = release

    def reply(self, question: str) -> BotReply:
        self.questions.append(question)
        if self.release is not None:
            self.release.wait(timeout=5)
        sources = (
            Source("sim-replacement.en.0", "sim-replacement", "Replacing a SIM"),
            Source("sim-replacement.en.1", "sim-replacement", "Replacing a SIM"),
            Source("service-hours.en.0", "service-hours", "Service hours"),
        )
        return BotReply(ReplyKind.ANSWER, "en", "Visit a service centre.", sources)


class FakeTts:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, Language]] = []

    async def speak(self, text: str, lang: Language) -> AsyncIterator[np.ndarray]:
        self.spoken.append((text, lang))
        for _ in range(2):
            yield np.full(160, 0.25, dtype=np.float32)


def services(
    asr: FakeAsr | None = None, pipeline: FakePipeline | None = None
) -> tuple[CallServices, FakeAsr, FakePipeline, FakeTts]:
    asr, pipeline, tts = asr or FakeAsr(), pipeline or FakePipeline(), FakeTts()
    return CallServices(lambda: loud_probability, asr, pipeline, tts), asr, pipeline, tts


def speech_then_silence(speech_seconds: float = 1.0, silence_seconds: float = 1.2) -> list[bytes]:
    audio = np.concatenate(
        [
            np.full(int(speech_seconds * SAMPLE_RATE_HZ), 0.5, dtype=np.float32),
            np.zeros(int(silence_seconds * SAMPLE_RATE_HZ), dtype=np.float32),
        ]
    )
    pcm = float_to_pcm16(audio)
    step = FRAME_SAMPLES * 2
    return [pcm[start : start + step] for start in range(0, len(pcm), step)]


def next_event(websocket: WebSocketTestSession) -> dict[str, Any] | bytes:
    message = websocket.receive()
    if message.get("bytes") is not None:
        return message["bytes"]
    return json.loads(message["text"])


def events_until(websocket: WebSocketTestSession, last_type: str) -> list[dict[str, Any] | bytes]:
    events: list[dict[str, Any] | bytes] = []
    while True:
        event = next_event(websocket)
        events.append(event)
        if isinstance(event, dict) and event["type"] == last_type:
            return events


def types(events: list[dict[str, Any] | bytes]) -> list[str]:
    return ["audio" if isinstance(event, bytes) else event["type"] for event in events]


def connect_and_skip_greeting(client: TestClient) -> Any:
    websocket = client.websocket_connect("/ws/call")
    session = websocket.__enter__()
    events_until(session, "reply_audio_end")
    return websocket, session


def test_a_call_starts_with_ready_and_the_bangla_greeting() -> None:
    call_services, _, _, tts = services()
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events = events_until(websocket, "reply_audio_end")
    assert types(events) == ["ready", "reply", "audio", "audio", "reply_audio_end"]
    ready, greeting = events[0], events[1]
    assert isinstance(ready, dict) and ready == {
        "type": "ready",
        "sample_rate": 16000,
        "audio_format": "pcm16le_mono",
    }
    assert isinstance(greeting, dict) and greeting["kind"] == "greeting"
    assert greeting["text"] == GREETING["bn"]
    assert tts.spoken == [(GREETING["bn"], "bn")]
    assert events[2] == float_to_pcm16(np.full(160, 0.25, dtype=np.float32))


def test_a_spoken_turn_gets_a_transcript_a_reply_with_sources_and_audio() -> None:
    call_services, asr, pipeline, tts = services()
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events = events_until(websocket, "reply_audio_end")
    assert types(events) == [
        "speech_started",
        "transcript",
        "reply",
        "audio",
        "audio",
        "reply_audio_end",
    ]
    assert events[1] == {"type": "transcript", "text": QUESTION}
    reply = events[2]
    assert isinstance(reply, dict)
    assert reply["kind"] == "answer" and reply["lang"] == "en"
    assert reply["sources"] == [
        {"doc_id": "sim-replacement", "title": "Replacing a SIM"},
        {"doc_id": "service-hours", "title": "Service hours"},
    ]
    assert pipeline.questions == [QUESTION]
    assert len(asr.turn_seconds) == 1 and 1.0 <= asr.turn_seconds[0] < 2.2
    assert tts.spoken[-1] == ("Visit a service centre.", "en")


def test_silence_and_quiet_noise_never_reach_asr() -> None:
    call_services, asr, _, _ = services()
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(3 * SAMPLE_RATE_HZ) * 0.02).astype(np.float32)
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            websocket.send_bytes(float_to_pcm16(np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)))
            pcm = float_to_pcm16(noise)
            for start in range(0, len(pcm), FRAME_SAMPLES * 2):
                websocket.send_bytes(pcm[start : start + FRAME_SAMPLES * 2])
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events_until(websocket, "reply_audio_end")
    assert len(asr.turn_seconds) == 1


@pytest.mark.parametrize(
    ("bad_transcript", "reason"),
    [
        (transcript(text="  "), "empty_transcript"),
        (transcript(ratio=3.1), "compression_ratio"),
        (transcript(tokens=224), "token_ceiling"),
    ],
)
def test_an_unusable_transcript_gets_the_repeat_prompt_without_the_pipeline(
    bad_transcript: Transcript, reason: str
) -> None:
    assert unusable_reason(bad_transcript) == reason
    call_services, _, pipeline, _ = services(asr=FakeAsr(bad_transcript))
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events = events_until(websocket, "reply_audio_end")
    assert "transcript" not in types(events)
    reply = next(event for event in events if isinstance(event, dict) and event["type"] == "reply")
    assert reply["kind"] == "repeat"
    assert reply["text"] == FIXED_REPLIES[ReplyKind.REPEAT]["bn"]
    assert pipeline.questions == []


def test_a_usable_transcript_passes_the_gate() -> None:
    assert unusable_reason(transcript(ratio=2.4, tokens=223)) is None


@pytest.mark.parametrize(
    ("frame", "code"),
    [(b"\x00" * 3, "odd_length_frame"), (b"\x00" * (MAX_FRAME_BYTES + 2), "frame_too_large")],
)
def test_frame_errors(frame: bytes, code: str) -> None:
    assert frame_error(frame) == code
    assert frame_error(b"\x00" * MAX_FRAME_BYTES) is None
    assert frame_error(b"") is None


def test_malformed_frames_get_error_events_and_the_call_goes_on() -> None:
    call_services, asr, _, _ = services()
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            websocket.send_text("hello")
            assert next_event(websocket) == {"type": "error", "code": "text_frame_ignored"}
            websocket.send_bytes(b"\x01\x02\x03")
            assert next_event(websocket) == {"type": "error", "code": "odd_length_frame"}
            websocket.send_bytes(b"\x00" * (MAX_FRAME_BYTES + 2))
            assert next_event(websocket) == {"type": "error", "code": "frame_too_large"}
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events = events_until(websocket, "reply_audio_end")
    assert "reply" in types(events)
    assert len(asr.turn_seconds) == 1


def test_the_repeat_prompt_follows_the_language_of_the_last_reply() -> None:
    asr = FakeAsr()
    call_services, _, _, _ = services(asr=asr)
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events_until(websocket, "reply_audio_end")
            asr.result = transcript(ratio=5.0)
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events = events_until(websocket, "reply_audio_end")
    reply = next(event for event in events if isinstance(event, dict) and event["type"] == "reply")
    assert reply["kind"] == "repeat" and reply["lang"] == "en"


def test_a_disconnect_while_the_pipeline_runs_ends_the_call_cleanly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    release = threading.Event()
    call_services, _, pipeline, tts = services(pipeline=FakePipeline(release))
    caplog.set_level(logging.INFO, logger="kotha")
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            assert next_event(websocket) == {"type": "speech_started"}
            assert next_event(websocket) == {"type": "transcript", "text": QUESTION}
        release.set()
    assert pipeline.questions == [QUESTION]
    assert [text for text, _ in tts.spoken] == [GREETING["bn"]]
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("event=call_ended") for message in messages)


def test_logs_name_events_but_never_carry_the_caller_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call_services, _, _, _ = services()
    caplog.set_level(logging.INFO, logger="kotha")
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events_until(websocket, "reply_audio_end")
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "event=call_started" in messages and "event=reply" in messages
    assert "kind=answer" in messages
    assert "SIM" not in messages and "service centre" not in messages


def test_handoff_reason_reaches_the_client() -> None:
    class HandoffPipeline(FakePipeline):
        def reply(self, question: str) -> BotReply:
            return BotReply(ReplyKind.HANDOFF, "bn", "এজেন্ট", (), HandoffReason.NO_ANSWER)

    call_services = CallServices(lambda: loud_probability, FakeAsr(), HandoffPipeline(), FakeTts())
    with TestClient(create_app(lambda _: call_services)) as client:
        with client.websocket_connect("/ws/call") as websocket:
            events_until(websocket, "reply_audio_end")
            for frame in speech_then_silence():
                websocket.send_bytes(frame)
            events = events_until(websocket, "reply_audio_end")
    reply = next(event for event in events if isinstance(event, dict) and event["type"] == "reply")
    assert reply["kind"] == "handoff" and reply["handoff_reason"] == "no_answer"
