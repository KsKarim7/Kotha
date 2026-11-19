import asyncio
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from kotha.audio.convert import SAMPLE_RATE_HZ, resample
from kotha.audio.telephony import TELEPHONY_SAMPLE_RATE_HZ
from kotha.dialogue.answer import FIXED_REPLIES, GREETING, ReplyKind
from kotha.dialogue.routing import Language
from kotha.tts.mms import VOICES, MmsVoice
from kotha.tts.speakable import speakable_sentences

SentenceAudio = tuple[np.ndarray, ...]
CACHED_REPLY_KINDS = (ReplyKind.HANDOFF, ReplyKind.REPEAT, ReplyKind.ACCOUNT_UNAVAILABLE)


class Voice(Protocol):
    sampling_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


def fixed_prompts(lang: Language) -> list[str]:
    return [GREETING[lang], *(FIXED_REPLIES[kind][lang] for kind in CACHED_REPLY_KINDS)]


def read_only(audio: np.ndarray) -> np.ndarray:
    audio.setflags(write=False)
    return audio


class TtsService:
    def __init__(self, voices: Mapping[Language, Voice]) -> None:
        self._voices = voices
        self._cache: dict[tuple[Language, str], SentenceAudio] = {}

    def warm_fixed_prompts(self) -> None:
        for lang in self._voices:
            for prompt in fixed_prompts(lang):
                audio = self.synthesize(prompt, lang)
                self._cache[(lang, prompt)] = tuple(read_only(sentence) for sentence in audio)

    def synthesize(self, text: str, lang: Language) -> SentenceAudio:
        cached = self._cache.get((lang, text))
        if cached is not None:
            return cached
        voice = self._voices[lang]
        return tuple(
            resample(voice.synthesize(sentence), voice.sampling_rate, SAMPLE_RATE_HZ)
            for sentence in speakable_sentences(text, lang)
        )

    async def speak(self, text: str, lang: Language) -> SentenceAudio:
        cached = self._cache.get((lang, text))
        if cached is not None:
            return cached
        return await asyncio.to_thread(self.synthesize, text, lang)


def load_tts_service() -> TtsService:
    service = TtsService({lang: MmsVoice(lang) for lang in VOICES})
    service.warm_fixed_prompts()
    return service


def to_telephony_rate(audio_16k: np.ndarray) -> np.ndarray:
    return resample(audio_16k, SAMPLE_RATE_HZ, TELEPHONY_SAMPLE_RATE_HZ)
