from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from kotha.audio.convert import SAMPLE_RATE_HZ
from kotha.dialogue.answer import HandoffReason, ReplyKind
from kotha.dialogue.routing import Language

ErrorCode = Literal[
    "bad_control_message",
    "not_started",
    "already_started",
    "consent_required",
    "odd_length_frame",
    "frame_too_large",
    "too_many_bad_frames",
    "tts_failed",
]


class Start(BaseModel):
    type: Literal["start"]
    sample_rate: Literal[16000]
    consent: bool


class Stop(BaseModel):
    type: Literal["stop"]


CLIENT_MESSAGE = TypeAdapter(Annotated[Start | Stop, Field(discriminator="type")])


class Ready(BaseModel):
    type: Literal["ready"] = "ready"
    sample_rate: int = SAMPLE_RATE_HZ
    audio_format: Literal["pcm16le_mono"] = "pcm16le_mono"


class Vad(BaseModel):
    type: Literal["vad"] = "vad"
    event: Literal["speech_start", "speech_end"]


class TranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    text: str
    final: bool = True


class Citation(BaseModel):
    doc_id: str
    title: str


class BotText(BaseModel):
    type: Literal["bot_text"] = "bot_text"
    kind: ReplyKind | Literal["greeting"]
    lang: Language
    text: str
    citations: list[Citation] = []


class Handoff(BaseModel):
    type: Literal["handoff"] = "handoff"
    reason: HandoffReason


class ReplyAudioEnd(BaseModel):
    type: Literal["reply_audio_end"] = "reply_audio_end"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: ErrorCode


ServerEvent = Ready | Vad | TranscriptEvent | BotText | Handoff | ReplyAudioEnd | ErrorEvent
