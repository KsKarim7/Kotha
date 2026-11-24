import asyncio
import secrets
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

import anyio
import numpy as np
from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from kotha.api.protocol import (
    CLIENT_MESSAGE,
    BotText,
    Citation,
    ErrorCode,
    ErrorEvent,
    Handoff,
    Ready,
    ReplyAudioEnd,
    ServerEvent,
    StageSeconds,
    Start,
    Timing,
    TranscriptEvent,
    Vad,
)
from kotha.asr.whisper import Transcript
from kotha.audio.convert import SAMPLE_RATE_HZ, float_to_pcm16, pcm16_to_float
from kotha.dialogue.answer import (
    GREETING,
    BotReply,
    HandoffReason,
    ReplyKind,
    fixed_reply,
    handoff,
)
from kotha.dialogue.routing import Language
from kotha.logs import log_event
from kotha.vad import WINDOW_SAMPLES
from kotha.vad import turns as vad_turns

# ADR 0005: a looping decode shows a high compression ratio or hits Whisper's 224-token ceiling
MAX_COMPRESSION_RATIO = 2.4
SEGMENT_TOKEN_CEILING = 224
MAX_FRAME_BYTES = SAMPLE_RATE_HZ * 2
MAX_BAD_MESSAGES = 20
FAILED_TURNS_BEFORE_HANDOFF = 2
OUTGOING_LIMIT = 64
GREETING_LANG: Language = "bn"
MODEL_ERRORS = (RuntimeError, ValueError)
POLICY_VIOLATION = 1008
GOING_AWAY = 1001
# a browser sends a 20 ms frame all call long, so this much silence means the client is gone
CLIENT_SILENCE_TIMEOUT_S = 30.0
# a client that stops reading lets the socket buffer fill, and every send then waits forever
SEND_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Close:
    code: int = 1000


Outgoing = ServerEvent | bytes | Close


class Transcriber(Protocol):
    def transcribe(self, audio_16k: np.ndarray) -> Transcript: ...


class Replier(Protocol):
    def reply(self, question: str) -> BotReply: ...


class Speaker(Protocol):
    def speak(self, text: str, lang: Language) -> AsyncIterator[np.ndarray]: ...


@dataclass(frozen=True)
class CallServices:
    new_vad: Callable[[], Callable[[np.ndarray], float]]
    asr: Transcriber
    pipeline: Replier
    tts: Speaker


class CallEnded(Exception):
    pass


def unusable_reason(transcript: Transcript) -> str | None:
    if not transcript.text.strip():
        return "empty_transcript"
    if transcript.max_compression_ratio > MAX_COMPRESSION_RATIO:
        return "compression_ratio"
    if transcript.max_segment_tokens >= SEGMENT_TOKEN_CEILING:
        return "token_ceiling"
    return None


def frame_error(frame: bytes) -> ErrorCode | None:
    if len(frame) > MAX_FRAME_BYTES:
        return "frame_too_large"
    if len(frame) % 2:
        return "odd_length_frame"
    return None


def bot_text(reply: BotReply) -> BotText:
    titles = {source.doc_id: source.title for source in reply.sources}
    citations = [Citation(doc_id=doc_id, title=title) for doc_id, title in titles.items()]
    return BotText(kind=reply.kind, lang=reply.lang, text=reply.text, citations=citations)


class CallSession:
    def __init__(self, websocket: WebSocket, services: CallServices) -> None:
        self._websocket = websocket
        self._services = services
        self._unwindowed = np.zeros(0, dtype=np.float32)
        self._started = asyncio.Event()
        # one waiting turn at most: without barge-in, older speech would be answered too late
        self._turns: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
        self._outgoing: asyncio.Queue[Outgoing] = asyncio.Queue(maxsize=OUTGOING_LIMIT)
        self._bad_messages = 0
        self._failed_turns = 0
        self._lang: Language = GREETING_LANG
        self.call_id = f"call-{secrets.token_hex(4)}"

    async def run(self) -> None:
        await self._websocket.accept()
        log_event("call_started", call_id=self.call_id)
        await self._outgoing.put(Ready())
        # anyio, not asyncio.TaskGroup, so Starlette's cancel scopes can end the call cleanly
        try:
            detector = await asyncio.to_thread(self._new_detector)
            async with anyio.create_task_group() as group:
                group.start_soon(self._receive, detector)
                group.start_soon(self._handle_turns)
                group.start_soon(self._send)
        except* CallEnded:
            pass
        finally:
            log_event("call_ended", call_id=self.call_id)

    def _new_detector(self) -> vad_turns.TurnDetector:
        return vad_turns.TurnDetector(self._services.new_vad())

    async def _receive(self, detector: vad_turns.TurnDetector) -> None:
        while True:
            try:
                with anyio.fail_after(CLIENT_SILENCE_TIMEOUT_S):
                    message = await self._websocket.receive()
            except TimeoutError:
                log_event("call_timeout", call_id=self.call_id, reason="client_silent")
                await self._outgoing.put(Close(GOING_AWAY))
                return
            if message["type"] == "websocket.disconnect":
                raise CallEnded
            if message.get("text") is not None:
                await self._control(message["text"])
                continue
            frame: bytes = message.get("bytes") or b""
            if not self._started.is_set():
                await self._reject("not_started")
            elif (error := frame_error(frame)) is not None:
                await self._reject(error)
            else:
                await self._listen(detector, pcm16_to_float(frame))

    async def _control(self, text: str) -> None:
        try:
            message = CLIENT_MESSAGE.validate_json(text)
        except ValidationError:
            await self._reject("bad_control_message")
            return
        if not isinstance(message, Start):
            await self._outgoing.put(Close())
        elif not message.consent:
            await self._outgoing.put(ErrorEvent(code="consent_required"))
            await self._outgoing.put(Close(POLICY_VIOLATION))
        elif self._started.is_set():
            await self._reject("already_started")
        else:
            self._started.set()

    async def _listen(self, detector: vad_turns.TurnDetector, samples: np.ndarray) -> None:
        for event in await asyncio.to_thread(self._detect, detector, samples):
            if isinstance(event, vad_turns.SpeechStarted):
                await self._outgoing.put(Vad(event="speech_start"))
                continue
            await self._outgoing.put(Vad(event="speech_end"))
            if self._turns.full():
                log_event("turn_dropped", call_id=self.call_id, reason="bot_busy")
            else:
                self._turns.put_nowait(event.audio)

    async def _reject(self, error: ErrorCode) -> None:
        self._bad_messages += 1
        if self._bad_messages > MAX_BAD_MESSAGES:
            return
        log_event("message_rejected", call_id=self.call_id, error=error)
        if self._bad_messages == MAX_BAD_MESSAGES:
            await self._outgoing.put(ErrorEvent(code="too_many_bad_frames"))
            await self._outgoing.put(Close(POLICY_VIOLATION))
        else:
            await self._outgoing.put(ErrorEvent(code=error))

    def _detect(
        self, detector: vad_turns.TurnDetector, samples: np.ndarray
    ) -> list[vad_turns.SpeechStarted | vad_turns.SpeechEnded]:
        audio = np.concatenate([self._unwindowed, samples])
        whole = len(audio) - len(audio) % WINDOW_SAMPLES
        self._unwindowed = audio[whole:]
        events = (
            detector.push(audio[start : start + WINDOW_SAMPLES])
            for start in range(0, whole, WINDOW_SAMPLES)
        )
        return [event for event in events if event is not None]

    async def _handle_turns(self) -> None:
        await self._started.wait()
        greeting = BotText(kind="greeting", lang=GREETING_LANG, text=GREETING[GREETING_LANG])
        await self._speak(greeting)
        while True:
            await self._answer(await self._turns.get())

    async def _transcribe(self, turn_audio: np.ndarray) -> Transcript | None:
        try:
            return await asyncio.to_thread(self._services.asr.transcribe, turn_audio)
        except MODEL_ERRORS as error:
            log_event("asr_failed", call_id=self.call_id, error=type(error).__name__)
            return None

    async def _answer(self, turn_audio: np.ndarray) -> None:
        asr_started = time.perf_counter()
        transcript = await self._transcribe(turn_audio)
        asr_seconds = time.perf_counter() - asr_started
        failure = "asr_error" if transcript is None else unusable_reason(transcript)
        if failure is not None or transcript is None:
            log_event("turn_failed", call_id=self.call_id, reason=failure or "asr_error")
            reply = self._failed_turn_reply()
        else:
            self._failed_turns = 0
            await self._outgoing.put(TranscriptEvent(text=transcript.text))
            reply = await asyncio.to_thread(self._services.pipeline.reply, transcript.text)
            self._lang = reply.lang
        details = {"reason": reply.handoff_reason.value} if reply.handoff_reason else {}
        if reply.model:
            details["model"] = reply.model
        log_event("reply", call_id=self.call_id, kind=reply.kind.value, lang=reply.lang, **details)
        for rejection in reply.rejections:
            log_event("answer_rejected", call_id=self.call_id, reason=rejection.value)
        tts_first_audio_seconds = await self._speak(bot_text(reply))
        stages = StageSeconds(
            asr_seconds=asr_seconds,
            retrieval_seconds=reply.stage_seconds.get("retrieval"),
            llm_seconds=reply.stage_seconds.get("llm"),
            tts_first_audio_seconds=tts_first_audio_seconds,
        )
        for stage, seconds in stages.model_dump(exclude_none=True).items():
            log_event("stage", call_id=self.call_id, stage=stage, seconds=round(seconds, 4))
        timing = Timing(
            **stages.model_dump(),
            kind=reply.kind,
            handoff_reason=reply.handoff_reason,
            rejections=list(reply.rejections),
            model=reply.model,
        )
        # timing is the last event of every turn, so a client can stop reading there
        if reply.handoff_reason is not None:
            await self._outgoing.put(Handoff(reason=reply.handoff_reason))
        await self._outgoing.put(timing)

    def _failed_turn_reply(self) -> BotReply:
        self._failed_turns += 1
        if self._failed_turns >= FAILED_TURNS_BEFORE_HANDOFF:
            self._failed_turns = 0
            return handoff(self._lang, HandoffReason.FAILED_TURNS)
        return fixed_reply(ReplyKind.REPEAT, self._lang)

    async def _speak(self, text: BotText) -> float | None:
        await self._outgoing.put(text)
        started = time.perf_counter()
        first_audio_seconds = None
        try:
            async for sentence_audio in self._services.tts.speak(text.text, text.lang):
                if first_audio_seconds is None:
                    first_audio_seconds = time.perf_counter() - started
                await self._outgoing.put(float_to_pcm16(sentence_audio))
        except MODEL_ERRORS as error:
            log_event("tts_failed", call_id=self.call_id, error=type(error).__name__)
            await self._outgoing.put(ErrorEvent(code="tts_failed"))
        await self._outgoing.put(ReplyAudioEnd())
        return first_audio_seconds

    async def _send(self) -> None:
        while True:
            item = await self._outgoing.get()
            try:
                with anyio.fail_after(SEND_TIMEOUT_S):
                    await self._deliver(item)
            except WebSocketDisconnect as disconnect:
                raise CallEnded from disconnect
            except TimeoutError:
                log_event("call_timeout", call_id=self.call_id, reason="client_not_reading")
                raise CallEnded from None

    async def _deliver(self, item: Outgoing) -> None:
        if isinstance(item, Close):
            await self._websocket.close(item.code)
            raise CallEnded
        if isinstance(item, bytes):
            await self._websocket.send_bytes(item)
        else:
            await self._websocket.send_text(item.model_dump_json())
