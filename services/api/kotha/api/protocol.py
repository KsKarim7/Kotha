from typing import Literal

from pydantic import BaseModel

from kotha.audio.convert import SAMPLE_RATE_HZ
from kotha.dialogue.routing import Language

ReplyEventKind = Literal["greeting", "answer", "account_unavailable", "handoff", "repeat"]
ErrorCode = Literal["text_frame_ignored", "odd_length_frame", "frame_too_large"]


class Ready(BaseModel):
    type: Literal["ready"] = "ready"
    sample_rate: int = SAMPLE_RATE_HZ
    audio_format: Literal["pcm16le_mono"] = "pcm16le_mono"


class SpeechStarted(BaseModel):
    type: Literal["speech_started"] = "speech_started"


class TranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    text: str


class SourceEvent(BaseModel):
    doc_id: str
    title: str


class ReplyEvent(BaseModel):
    type: Literal["reply"] = "reply"
    kind: ReplyEventKind
    lang: Language
    text: str
    sources: list[SourceEvent] = []
    handoff_reason: str | None = None


class ReplyAudioEnd(BaseModel):
    type: Literal["reply_audio_end"] = "reply_audio_end"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: ErrorCode


ServerEvent = Ready | SpeechStarted | TranscriptEvent | ReplyEvent | ReplyAudioEnd | ErrorEvent
