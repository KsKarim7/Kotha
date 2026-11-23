import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

import numpy as np

from kotha.audio.convert import SAMPLE_RATE_HZ, resample
from kotha.audio.telephony import TELEPHONY_SAMPLE_RATE_HZ
from kotha.dialogue.answer import FIXED_REPLIES, GREETING, ReplyKind
from kotha.dialogue.routing import Language
from kotha.logs import logged_duration
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

    def warm_fixed_prompts(self, lang: Language) -> None:
        for prompt in fixed_prompts(lang):
            audio = self.synthesize(prompt, lang)
            self._cache[(lang, prompt)] = tuple(read_only(sentence) for sentence in audio)

    def synthesize(self, text: str, lang: Language) -> SentenceAudio:
        cached = self._cache.get((lang, text))
        if cached is not None:
            return cached
        return tuple(
            self.sentence_audio(sentence, lang) for sentence in speakable_sentences(text, lang)
        )

    def sentence_audio(self, speakable_sentence: str, lang: Language) -> np.ndarray:
        voice = self._voices[lang]
        return resample(voice.synthesize(speakable_sentence), voice.sampling_rate, SAMPLE_RATE_HZ)

    async def speak(self, text: str, lang: Language) -> AsyncIterator[np.ndarray]:
        cached = self._cache.get((lang, text))
        if cached is not None:
            for audio in cached:
                yield audio
            return
        # one worker call per sentence, so a cancelled reply stops after the current sentence
        for sentence in speakable_sentences(text, lang):
            yield await asyncio.to_thread(self.sentence_audio, sentence, lang)


def load_tts_service() -> TtsService:
    voices: dict[Language, Voice] = {}
    for lang in VOICES:
        with logged_duration(f"tts_load_{lang}"):
            voices[lang] = MmsVoice(lang)
    service = TtsService(voices)
    for lang in voices:
        with logged_duration(f"tts_prompts_{lang}"):
            service.warm_fixed_prompts(lang)
    return service


def to_telephony_rate(audio_16k: np.ndarray) -> np.ndarray:
    return resample(audio_16k, SAMPLE_RATE_HZ, TELEPHONY_SAMPLE_RATE_HZ)
