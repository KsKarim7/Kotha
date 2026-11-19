import asyncio
import secrets
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

import anyio
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

from kotha.api.protocol import (
    ErrorCode,
    ErrorEvent,
    Ready,
    ReplyAudioEnd,
    ReplyEvent,
    ReplyEventKind,
    ServerEvent,
    SourceEvent,
    SpeechStarted,
    TranscriptEvent,
)
from kotha.asr.whisper import Transcript
from kotha.audio.convert import SAMPLE_RATE_HZ, float_to_pcm16, pcm16_to_float
from kotha.dialogue.answer import GREETING, BotReply, ReplyKind, fixed_reply
from kotha.dialogue.routing import Language
from kotha.logs import log_event
from kotha.vad import WINDOW_SAMPLES
from kotha.vad import turns as vad_turns

# ADR 0005: a looping decode shows a high compression ratio or hits Whisper's 224-token ceiling
MAX_COMPRESSION_RATIO = 2.4
SEGMENT_TOKEN_CEILING = 224
MAX_FRAME_BYTES = SAMPLE_RATE_HZ * 2
GREETING_LANG: Language = "bn"


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


def reply_event(reply: BotReply) -> ReplyEvent:
    titles = {source.doc_id: source.title for source in reply.sources}
    return ReplyEvent(
        kind=reply.kind.value,
        lang=reply.lang,
        text=reply.text,
        sources=[SourceEvent(doc_id=doc_id, title=title) for doc_id, title in titles.items()],
        handoff_reason=reply.handoff_reason,
    )


class CallSession:
    def __init__(self, websocket: WebSocket, services: CallServices) -> None:
        self._websocket = websocket
        self._services = services
        self._detector = vad_turns.TurnDetector(services.new_vad())
        self._unwindowed = np.zeros(0, dtype=np.float32)
        self._turns: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._outgoing: asyncio.Queue[ServerEvent | bytes] = asyncio.Queue()
        self._lang: Language = GREETING_LANG
        self.call_id = f"call-{secrets.token_hex(4)}"

    async def run(self) -> None:
        await self._websocket.accept()
        log_event("call_started", call_id=self.call_id)
        await self._outgoing.put(Ready())
        # anyio, not asyncio.TaskGroup, so Starlette's cancel scopes can end the call cleanly
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(self._receive)
                group.start_soon(self._handle_turns)
                group.start_soon(self._send)
        except* CallEnded:
            pass
        finally:
            log_event("call_ended", call_id=self.call_id)

    async def _receive(self) -> None:
        while True:
            message = await self._websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise CallEnded
            frame = message.get("bytes")
            if frame is None:
                await self._reject("text_frame_ignored")
                continue
            error = frame_error(frame)
            if error is not None:
                await self._reject(error)
                continue
            for event in await asyncio.to_thread(self._detect, pcm16_to_float(frame)):
                if isinstance(event, vad_turns.SpeechStarted):
                    await self._outgoing.put(SpeechStarted())
                else:
                    await self._turns.put(event.audio)

    async def _reject(self, error: ErrorCode) -> None:
        log_event("frame_rejected", call_id=self.call_id, error=error)
        await self._outgoing.put(ErrorEvent(code=error))

    def _detect(self, samples: np.ndarray) -> list[vad_turns.SpeechStarted | vad_turns.SpeechEnded]:
        audio = np.concatenate([self._unwindowed, samples])
        whole = len(audio) - len(audio) % WINDOW_SAMPLES
        self._unwindowed = audio[whole:]
        events = (
            self._detector.push(audio[start : start + WINDOW_SAMPLES])
            for start in range(0, whole, WINDOW_SAMPLES)
        )
        return [event for event in events if event is not None]

    async def _handle_turns(self) -> None:
        await self._say("greeting", GREETING_LANG, GREETING[GREETING_LANG])
        while True:
            turn_audio = await self._turns.get()
            await self._answer(turn_audio)

    async def _answer(self, turn_audio: np.ndarray) -> None:
        transcript = await asyncio.to_thread(self._services.asr.transcribe, turn_audio)
        reason = unusable_reason(transcript)
        if reason is not None:
            log_event("turn_rejected", call_id=self.call_id, reason=reason)
            reply = fixed_reply(ReplyKind.REPEAT, self._lang)
        else:
            await self._outgoing.put(TranscriptEvent(text=transcript.text))
            reply = await asyncio.to_thread(self._services.pipeline.reply, transcript.text)
            self._lang = reply.lang
        log_event("reply", call_id=self.call_id, kind=reply.kind.value, lang=reply.lang)
        await self._speak(reply_event(reply))

    async def _say(self, kind: ReplyEventKind, lang: Language, text: str) -> None:
        await self._speak(ReplyEvent(kind=kind, lang=lang, text=text))

    async def _speak(self, event: ReplyEvent) -> None:
        await self._outgoing.put(event)
        async for sentence_audio in self._services.tts.speak(event.text, event.lang):
            await self._outgoing.put(float_to_pcm16(sentence_audio))
        await self._outgoing.put(ReplyAudioEnd())

    async def _send(self) -> None:
        while True:
            item = await self._outgoing.get()
            try:
                if isinstance(item, bytes):
                    await self._websocket.send_bytes(item)
                else:
                    await self._websocket.send_text(item.model_dump_json())
            except WebSocketDisconnect as disconnect:
                raise CallEnded from disconnect
