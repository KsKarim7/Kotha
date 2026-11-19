import json
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient, WebSocketTestSession

from kotha.api.app import create_app
from kotha.api.session import (
    MAX_BAD_MESSAGES,
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
from kotha.vad import WINDOW_SAMPLES

QUESTION = "how do I replace my SIM"
START = {"type": "start", "sample_rate": 16000, "consent": True}
Event = dict[str, Any] | bytes


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
    def __init__(self, result: Transcript | Exception | None = None) -> None:
        self.result = result or transcript()
        self.turn_samples: list[int] = []

    def transcribe(self, audio_16k: np.ndarray) -> Transcript:
        self.turn_samples.append(len(audio_16k))
        if isinstance(self.result, Exception):
            raise self.result
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
    def __init__(self, fail_on: str | None = None) -> None:
        self.spoken: list[tuple[str, Language]] = []
        self.fail_on = fail_on

    async def speak(self, text: str, lang: Language) -> AsyncIterator[np.ndarray]:
        self.spoken.append((text, lang))
        yield np.full(160, 0.25, dtype=np.float32)
        if text == self.fail_on:
            raise RuntimeError("synthesis failed")
        yield np.full(160, 0.25, dtype=np.float32)


@contextmanager
def call(
    asr: FakeAsr | None = None,
    pipeline: FakePipeline | None = None,
    tts: FakeTts | None = None,
) -> Iterator[WebSocketTestSession]:
    services = CallServices(
        lambda: loud_probability, asr or FakeAsr(), pipeline or FakePipeline(), tts or FakeTts()
    )
    with TestClient(create_app(lambda _: services)) as client:
        with client.websocket_connect("/v1/calls/stream") as websocket:
            assert next_event(websocket) == {
                "type": "ready",
                "sample_rate": 16000,
                "audio_format": "pcm16le_mono",
            }
            yield websocket


def started(websocket: WebSocketTestSession) -> list[Event]:
    websocket.send_text(json.dumps(START))
    return events_until(websocket, "reply_audio_end")


def speech_then_silence(
    speech_seconds: float = 1.0, silence_seconds: float = 1.2, frame_samples: int = 1_600
) -> list[bytes]:
    audio = np.concatenate(
        [
            np.full(int(speech_seconds * SAMPLE_RATE_HZ), 0.5, dtype=np.float32),
            np.zeros(int(silence_seconds * SAMPLE_RATE_HZ), dtype=np.float32),
        ]
    )
    pcm = float_to_pcm16(audio)
    step = frame_samples * 2
    return [pcm[start : start + step] for start in range(0, len(pcm), step)]


def say(websocket: WebSocketTestSession, frames: list[bytes] | None = None) -> None:
    for frame in frames or speech_then_silence():
        websocket.send_bytes(frame)


def next_event(websocket: WebSocketTestSession) -> Event:
    message = websocket.receive()
    if message.get("bytes") is not None:
        return message["bytes"]
    if message["type"] == "websocket.close":
        return {"type": "closed", "code": message.get("code")}
    return json.loads(message["text"])


def events_until(websocket: WebSocketTestSession, last_type: str) -> list[Event]:
    events: list[Event] = []
    while True:
        event = next_event(websocket)
        events.append(event)
        if isinstance(event, dict) and event["type"] == last_type:
            return events


def types(events: list[Event]) -> list[str]:
    return ["audio" if isinstance(event, bytes) else event["type"] for event in events]


def first(events: list[Event], event_type: str) -> dict[str, Any]:
    return next(e for e in events if isinstance(e, dict) and e["type"] == event_type)


def finishes_within(seconds: float, body: Callable[[], None]) -> bool:
    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    return not worker.is_alive()


def test_the_greeting_waits_for_start_with_consent() -> None:
    tts = FakeTts()
    with call(tts=tts) as websocket:
        events = started(websocket)
    assert types(events) == ["bot_text", "audio", "audio", "reply_audio_end"]
    greeting = first(events, "bot_text")
    assert greeting == {
        "type": "bot_text",
        "kind": "greeting",
        "lang": "bn",
        "text": GREETING["bn"],
        "citations": [],
    }
    assert tts.spoken == [(GREETING["bn"], "bn")]
    assert events[1] == float_to_pcm16(np.full(160, 0.25, dtype=np.float32))


def test_a_spoken_turn_gets_vad_events_a_transcript_and_a_cited_reply() -> None:
    asr, pipeline, tts = FakeAsr(), FakePipeline(), FakeTts()
    with call(asr, pipeline, tts) as websocket:
        started(websocket)
        say(websocket)
        events = events_until(websocket, "reply_audio_end")
    assert types(events) == [
        "vad",
        "vad",
        "transcript",
        "bot_text",
        "audio",
        "audio",
        "reply_audio_end",
    ]
    assert [event["event"] for event in events[:2] if isinstance(event, dict)] == [
        "speech_start",
        "speech_end",
    ]
    assert events[2] == {"type": "transcript", "text": QUESTION, "final": True}
    reply = first(events, "bot_text")
    assert reply["kind"] == "answer" and reply["lang"] == "en"
    assert reply["citations"] == [
        {"doc_id": "sim-replacement", "title": "Replacing a SIM"},
        {"doc_id": "service-hours", "title": "Service hours"},
    ]
    assert pipeline.questions == [QUESTION]
    assert tts.spoken[-1] == ("Visit a service centre.", "en")


def test_frames_that_split_windows_lose_no_samples() -> None:
    asr = FakeAsr()
    with call(asr=asr) as websocket:
        started(websocket)
        say(websocket, speech_then_silence(frame_samples=1_000))
        events_until(websocket, "reply_audio_end")
    speech_windows = -(-SAMPLE_RATE_HZ // WINDOW_SAMPLES)
    hangover_windows = -(-700 * SAMPLE_RATE_HZ // (1000 * WINDOW_SAMPLES))
    assert asr.turn_samples == [(speech_windows + hangover_windows) * WINDOW_SAMPLES]


def test_silence_and_quiet_noise_never_reach_asr() -> None:
    asr = FakeAsr()
    noise = (np.random.default_rng(0).standard_normal(3 * SAMPLE_RATE_HZ) * 0.02).astype(np.float32)
    with call(asr=asr) as websocket:
        started(websocket)
        say(websocket, [float_to_pcm16(np.zeros(SAMPLE_RATE_HZ, dtype=np.float32))])
        pcm = float_to_pcm16(noise)
        say(websocket, [pcm[start : start + 3_200] for start in range(0, len(pcm), 3_200)])
        say(websocket)
        events_until(websocket, "reply_audio_end")
    assert len(asr.turn_samples) == 1


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
    pipeline = FakePipeline()
    with call(asr=FakeAsr(bad_transcript), pipeline=pipeline) as websocket:
        started(websocket)
        say(websocket)
        events = events_until(websocket, "reply_audio_end")
    assert "transcript" not in types(events)
    reply = first(events, "bot_text")
    assert reply["kind"] == "repeat"
    assert reply["text"] == FIXED_REPLIES[ReplyKind.REPEAT]["bn"]
    assert pipeline.questions == []


def test_a_usable_transcript_passes_the_gate() -> None:
    assert unusable_reason(transcript(ratio=2.4, tokens=223)) is None


def test_an_asr_error_is_answered_and_two_failed_turns_hand_off() -> None:
    asr = FakeAsr(RuntimeError("decoder crashed"))
    with call(asr=asr) as websocket:
        started(websocket)
        say(websocket)
        first_turn = events_until(websocket, "reply_audio_end")
        say(websocket)
        second_turn = events_until(websocket, "handoff")
    assert first(first_turn, "bot_text")["kind"] == "repeat"
    assert first(second_turn, "bot_text")["kind"] == "handoff"
    assert first(second_turn, "bot_text")["text"] == FIXED_REPLIES[ReplyKind.HANDOFF]["bn"]
    assert first(second_turn, "handoff") == {"type": "handoff", "reason": "failed_turns"}
    assert types(second_turn)[-2:] == ["reply_audio_end", "handoff"]


def test_the_failed_turn_count_starts_again_after_a_handoff() -> None:
    with call(asr=FakeAsr(transcript(ratio=5.0))) as websocket:
        started(websocket)
        kinds = []
        for _ in range(3):
            say(websocket)
            kinds.append(first(events_until(websocket, "reply_audio_end"), "bot_text")["kind"])
    assert kinds == ["repeat", "handoff", "repeat"]


def test_a_good_turn_resets_the_failed_turn_count() -> None:
    asr = FakeAsr(transcript(ratio=5.0))
    with call(asr=asr) as websocket:
        started(websocket)
        say(websocket)
        events_until(websocket, "reply_audio_end")
        asr.result = transcript()
        say(websocket)
        events_until(websocket, "reply_audio_end")
        asr.result = transcript(ratio=5.0)
        say(websocket)
        events = events_until(websocket, "reply_audio_end")
    reply = first(events, "bot_text")
    assert reply["kind"] == "repeat" and reply["lang"] == "en"


def test_a_tts_error_sends_an_error_and_still_ends_the_reply_audio() -> None:
    with call(tts=FakeTts(fail_on="Visit a service centre.")) as websocket:
        started(websocket)
        say(websocket)
        events = events_until(websocket, "reply_audio_end")
        say(websocket)
        next_turn = events_until(websocket, "reply_audio_end")
    assert types(events)[-4:] == ["bot_text", "audio", "error", "reply_audio_end"]
    assert first(events, "error") == {"type": "error", "code": "tts_failed"}
    assert first(next_turn, "bot_text")["kind"] == "answer"


def test_the_pipeline_handoff_reason_reaches_the_client() -> None:
    class HandoffPipeline(FakePipeline):
        def reply(self, question: str) -> BotReply:
            return BotReply(ReplyKind.HANDOFF, "bn", "এজেন্ট", (), HandoffReason.NO_ANSWER)

    with call(pipeline=HandoffPipeline()) as websocket:
        started(websocket)
        say(websocket)
        events = events_until(websocket, "handoff")
    assert first(events, "bot_text")["kind"] == "handoff"
    assert first(events, "handoff") == {"type": "handoff", "reason": "no_answer"}


def test_turns_that_arrive_while_the_bot_is_busy_keep_only_the_latest_waiting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    release = threading.Event()
    pipeline = FakePipeline(release)
    caplog.set_level(logging.INFO, logger="kotha")
    with call(pipeline=pipeline) as websocket:
        started(websocket)
        for _ in range(3):
            say(websocket)
        speech_ends = 0
        while speech_ends < 3:
            event = next_event(websocket)
            if isinstance(event, dict) and event.get("event") == "speech_end":
                speech_ends += 1
        release.set()
        events_until(websocket, "reply_audio_end")
        events_until(websocket, "reply_audio_end")
    assert pipeline.questions == [QUESTION, QUESTION]
    assert any("event=turn_dropped" in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("hello", "bad_control_message"),
        (
            json.dumps({"type": "start", "sample_rate": 8000, "consent": True}),
            "bad_control_message",
        ),
        (json.dumps({"type": "dance"}), "bad_control_message"),
        (json.dumps(START), "already_started"),
    ],
)
def test_bad_control_messages_get_error_events_and_the_call_goes_on(
    message: str, code: str
) -> None:
    asr = FakeAsr()
    with call(asr=asr) as websocket:
        started(websocket)
        websocket.send_text(message)
        assert next_event(websocket) == {"type": "error", "code": code}
        say(websocket)
        events = events_until(websocket, "reply_audio_end")
    assert first(events, "bot_text")["kind"] == "answer"


def test_audio_before_start_is_refused_and_never_reaches_asr() -> None:
    asr = FakeAsr()
    with call(asr=asr) as websocket:
        websocket.send_bytes(b"\x00\x00" * 160)
        assert next_event(websocket) == {"type": "error", "code": "not_started"}
        started(websocket)
    assert asr.turn_samples == []


def test_start_without_consent_ends_the_call() -> None:
    tts = FakeTts()
    with call(tts=tts) as websocket:
        websocket.send_text(json.dumps({**START, "consent": False}))
        assert next_event(websocket) == {"type": "error", "code": "consent_required"}
        assert next_event(websocket) == {"type": "closed", "code": 1008}
    assert tts.spoken == []


def test_stop_ends_the_call() -> None:
    with call() as websocket:
        started(websocket)
        websocket.send_text(json.dumps({"type": "stop"}))
        assert next_event(websocket) == {"type": "closed", "code": 1000}


@pytest.mark.parametrize(
    ("frame", "code"),
    [(b"\x00" * 3, "odd_length_frame"), (b"\x00" * (MAX_FRAME_BYTES + 2), "frame_too_large")],
)
def test_frame_errors(frame: bytes, code: str) -> None:
    assert frame_error(frame) == code
    assert frame_error(b"\x00" * MAX_FRAME_BYTES) is None
    assert frame_error(b"") is None


def test_malformed_frames_get_error_events_and_the_call_goes_on() -> None:
    asr = FakeAsr()
    with call(asr=asr) as websocket:
        started(websocket)
        websocket.send_bytes(b"\x01\x02\x03")
        assert next_event(websocket) == {"type": "error", "code": "odd_length_frame"}
        websocket.send_bytes(b"\x00" * (MAX_FRAME_BYTES + 2))
        assert next_event(websocket) == {"type": "error", "code": "frame_too_large"}
        say(websocket)
        events_until(websocket, "reply_audio_end")
    assert len(asr.turn_samples) == 1


def test_too_many_bad_messages_close_the_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="kotha")
    with call() as websocket:
        started(websocket)
        for _ in range(MAX_BAD_MESSAGES + 5):
            websocket.send_bytes(b"\x01")
        events = events_until(websocket, "closed")
    assert events[-1] == {"type": "closed", "code": 1008}
    errors = [event for event in events if isinstance(event, dict) and event["type"] == "error"]
    assert len(errors) == MAX_BAD_MESSAGES
    assert errors[-1] == {"type": "error", "code": "too_many_bad_frames"}
    rejected = [message for message in caplog.messages if "event=message_rejected" in message]
    assert len(rejected) == MAX_BAD_MESSAGES


def test_a_disconnect_while_the_pipeline_runs_ends_the_call_without_hanging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    release = threading.Event()
    pipeline, tts = FakePipeline(release), FakeTts()
    caplog.set_level(logging.INFO, logger="kotha")

    def hang_up_mid_turn() -> None:
        with call(pipeline=pipeline, tts=tts) as websocket:
            started(websocket)
            say(websocket)
            events_until(websocket, "transcript")

    assert finishes_within(10, hang_up_mid_turn)
    release.set()
    assert pipeline.questions == [QUESTION]
    assert [text for text, _ in tts.spoken] == [GREETING["bn"]]
    assert any(message.startswith("event=call_ended") for message in caplog.messages)


def test_logs_name_events_but_never_carry_the_caller_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="kotha")
    with call() as websocket:
        started(websocket)
        say(websocket)
        events_until(websocket, "reply_audio_end")
    messages = " ".join(caplog.messages)
    assert "event=call_started" in messages and "kind=answer" in messages
    assert "SIM" not in messages and "service centre" not in messages
