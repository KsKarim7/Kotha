import asyncio
import os
import threading

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache

from kotha.audio.convert import SAMPLE_RATE_HZ
from kotha.dialogue.answer import FIXED_REPLIES, GREETING, ReplyKind
from kotha.dialogue.routing import Language
from kotha.tts.mms import VOICES, MmsVoice
from kotha.tts.service import TtsService, fixed_prompts, to_telephony_rate
from kotha.tts.speakable import VOICE_CHARACTERS, speakable_sentences

voices_are_cached = all(
    isinstance(try_to_load_from_cache(model_id, "model.safetensors", revision=revision), str)
    for model_id, revision in VOICES.values()
)


class FakeVoice:
    def __init__(self, sampling_rate: int = SAMPLE_RATE_HZ) -> None:
        self.sampling_rate = sampling_rate
        self.texts: list[str] = []
        self.threads: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        self.threads.append(threading.current_thread().name)
        return np.full(len(text) * 10, 0.5, dtype=np.float32)


def test_each_language_goes_to_its_own_voice_as_speakable_sentences() -> None:
    bangla, english = FakeVoice(), FakeVoice()
    service = TtsService({"bn": bangla, "en": english})
    audio = service.synthesize("প্যাকের মেয়াদ ৩০ দিন। *747# ডায়াল করুন।", "bn")
    assert bangla.texts == speakable_sentences("প্যাকের মেয়াদ ৩০ দিন। *747# ডায়াল করুন।", "bn")
    assert english.texts == []
    assert len(audio) == 2
    service.synthesize("Dial *747# now [a.en.0].", "en")
    assert english.texts == ["dial star seven four seven hash now"]


def test_voice_input_stays_inside_the_voice_vocabulary() -> None:
    bangla, english = FakeVoice(), FakeVoice()
    service = TtsService({"bn": bangla, "en": english})
    service.synthesize("NT-30GB-তে ৩০ GB, SIM আর NID লাগবে।", "bn")
    service.synthesize("Send BAL to 7470, 8 GB for 7 days.", "en")
    assert set("".join(bangla.texts)) <= VOICE_CHARACTERS["bn"]
    assert set("".join(english.texts)) <= VOICE_CHARACTERS["en"]


def test_audio_is_resampled_to_16_khz() -> None:
    voice = FakeVoice(sampling_rate=22_050)
    audio = TtsService({"en": voice}).synthesize("Hello there.", "en")
    expected_length = round(len("hello there") * 10 * SAMPLE_RATE_HZ / 22_050)
    assert abs(len(audio[0]) - expected_length) <= 1
    assert audio[0].dtype == np.float32


def test_text_with_nothing_to_say_gives_no_audio() -> None:
    voice = FakeVoice()
    assert TtsService({"en": voice}).synthesize(" ... [a.en.0] ", "en") == ()
    assert voice.texts == []


def test_fixed_prompts_are_the_greeting_and_the_fixed_replies() -> None:
    langs: tuple[Language, ...] = ("bn", "en")
    for lang in langs:
        assert fixed_prompts(lang) == [
            GREETING[lang],
            FIXED_REPLIES[ReplyKind.HANDOFF][lang],
            FIXED_REPLIES[ReplyKind.REPEAT][lang],
            FIXED_REPLIES[ReplyKind.ACCOUNT_UNAVAILABLE][lang],
        ]


def test_warmed_prompts_are_served_from_the_cache_without_synthesis() -> None:
    bangla, english = FakeVoice(), FakeVoice()
    service = TtsService({"bn": bangla, "en": english})
    service.warm_fixed_prompts()
    synthesised = len(bangla.texts) + len(english.texts)
    handoff = FIXED_REPLIES[ReplyKind.HANDOFF]["bn"]
    first = service.synthesize(handoff, "bn")
    second = asyncio.run(service.speak(handoff, "bn"))
    assert first is second
    assert len(bangla.texts) + len(english.texts) == synthesised
    with pytest.raises(ValueError, match="read-only"):
        first[0][0] = 1.0


def test_answers_are_not_cached() -> None:
    voice = FakeVoice()
    service = TtsService({"en": voice})
    service.synthesize("The pack lasts 30 days.", "en")
    service.synthesize("The pack lasts 30 days.", "en")
    assert len(voice.texts) == 2


def test_speak_synthesises_off_the_event_loop() -> None:
    voice = FakeVoice()
    service = TtsService({"en": voice})

    async def speak_on_loop() -> str:
        await service.speak("Hello there.", "en")
        return threading.current_thread().name

    loop_thread = asyncio.run(speak_on_loop())
    assert voice.threads and voice.threads[0] != loop_thread


def test_telephony_rate_halves_the_samples() -> None:
    audio_16k = np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)
    audio_8k = to_telephony_rate(audio_16k)
    assert len(audio_8k) == 8_000
    assert audio_8k.dtype == np.float32


def test_mms_voice_loads_offline_at_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[tuple[str, dict[str, object]]] = []

    class FakeLoaded:
        config = type("Config", (), {"sampling_rate": 16_000})()

        def eval(self) -> None:
            pass

    def fake_from_pretrained(model_id: str, **options: object) -> FakeLoaded:
        loads.append((model_id, options))
        return FakeLoaded()

    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("transformers.VitsModel.from_pretrained", fake_from_pretrained)
    voice = MmsVoice("bn")
    model_id, revision = VOICES["bn"]
    expected = {"revision": revision, "local_files_only": True}
    assert loads == [(model_id, expected), (model_id, expected)]
    assert voice.sampling_rate == 16_000
    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "true"


@pytest.mark.skipif(not voices_are_cached, reason="the MMS voices are not in the local cache")
def test_the_real_voices_give_the_same_16_khz_audio_twice() -> None:
    service = TtsService({lang: MmsVoice(lang) for lang in VOICES})
    for text, lang in (("প্যাকের মেয়াদ ৩০ দিন।", "bn"), ("The pack lasts 30 days.", "en")):
        first, second = service.synthesize(text, lang), service.synthesize(text, lang)
        assert len(first) == 1
        assert first[0].dtype == np.float32
        assert np.array_equal(first[0], second[0])
        assert 0.5 < len(first[0]) / SAMPLE_RATE_HZ < 10
