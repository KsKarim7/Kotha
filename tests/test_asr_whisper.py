from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kotha.asr.whisper import Transcript, WhisperAsr, Word, join_segment_texts
from kotha.audio.convert import SAMPLE_RATE_HZ, load_mono_16k

SPEECH_PATH = Path(__file__).parent / "fixtures" / "speech_16k.wav"


def segment(
    text: str,
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.1,
    compression_ratio: float = 1.2,
    tokens: int = 5,
    words: list[tuple[str, float, float]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
        tokens=list(range(tokens)),
        words=None
        if words is None
        else [SimpleNamespace(word=word, start=start, end=end) for word, start, end in words],
    )


class FakeWhisperModel:
    instances: list["FakeWhisperModel"] = []

    def __init__(self, model: str, **settings: str) -> None:
        self.model = model
        self.settings = settings
        self.calls: list[dict] = []
        self.segments: list[SimpleNamespace] = [segment("hello")]
        self.language = "bn"
        FakeWhisperModel.instances.append(self)

    def transcribe(
        self, audio: np.ndarray, **options: object
    ) -> tuple[Iterator[SimpleNamespace], SimpleNamespace]:
        self.calls.append(options)
        return (item for item in self.segments), SimpleNamespace(language=self.language)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    FakeWhisperModel.instances = []
    monkeypatch.setattr("kotha.asr.whisper.WhisperModel", FakeWhisperModel)
    return FakeWhisperModel


def silence(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE_HZ * seconds), dtype=np.float32)


def test_model_is_loaded_once(fake_model: type[FakeWhisperModel]) -> None:
    asr = WhisperAsr("tiny", device="cpu", compute_type="int8")
    asr.transcribe(silence())
    asr.transcribe(silence())
    assert len(fake_model.instances) == 1
    model = fake_model.instances[0]
    assert (model.model, model.settings) == ("tiny", {"device": "cpu", "compute_type": "int8"})
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    ("options", "word_timestamps"), [({}, False), ({"word_timestamps": True}, True)]
)
def test_transcribe_passes_the_decoding_config(
    fake_model: type[FakeWhisperModel], options: dict, word_timestamps: bool
) -> None:
    WhisperAsr().transcribe(silence(), **options)
    assert fake_model.instances[0].calls == [
        {
            "language": "bn",
            "beam_size": 1,
            "initial_prompt": None,
            "temperature": 0.0,
            "without_timestamps": True,
            "condition_on_previous_text": False,
            "word_timestamps": word_timestamps,
            "vad_filter": False,
        }
    ]


def test_transcribe_passes_given_options(fake_model: type[FakeWhisperModel]) -> None:
    WhisperAsr().transcribe(silence(), language="en", beam_size=5, initial_prompt="Nodi Telecom")
    call = fake_model.instances[0].calls[0]
    assert (call["language"], call["beam_size"], call["initial_prompt"]) == (
        "en",
        5,
        "Nodi Telecom",
    )


def test_segment_texts_are_stripped_and_joined(fake_model: type[FakeWhisperModel]) -> None:
    asr = WhisperAsr()
    fake_model.instances[0].segments = [segment(" আমার "), segment("  "), segment(" নম্বর  ")]
    assert asr.transcribe(silence()).text == "আমার নম্বর"


@pytest.mark.parametrize(
    ("texts", "joined"),
    [
        ([" আমার ", "  ", " নম্বর  "], "আমার নম্বর"),
        (["", " a", "b "], "a b"),
        (["   "], ""),
        ([], ""),
    ],
)
def test_join_segment_texts_strips_and_skips_empty_segments(texts: list[str], joined: str) -> None:
    assert join_segment_texts(segment(text) for text in texts) == joined


def test_loop_signals_take_the_largest_segment(fake_model: type[FakeWhisperModel]) -> None:
    asr = WhisperAsr()
    fake_model.instances[0].segments = [
        segment("a", compression_ratio=1.4, tokens=12),
        segment("b", compression_ratio=2.6, tokens=3),
        segment("c", compression_ratio=0.9, tokens=30),
    ]
    transcript = asr.transcribe(silence())
    assert (transcript.max_compression_ratio, transcript.max_segment_tokens) == (2.6, 30)


def test_confidence_signals_take_the_worst_segment(fake_model: type[FakeWhisperModel]) -> None:
    asr = WhisperAsr()
    fake_model.instances[0].segments = [
        segment("a", avg_logprob=-0.2, no_speech_prob=0.1),
        segment("b", avg_logprob=-0.9, no_speech_prob=0.05),
        segment("c", avg_logprob=-0.4, no_speech_prob=0.3),
    ]
    transcript = asr.transcribe(silence())
    assert (transcript.worst_avg_logprob, transcript.max_no_speech_prob) == (-0.9, 0.3)


def test_words_are_flattened_in_order(fake_model: type[FakeWhisperModel]) -> None:
    asr = WhisperAsr()
    fake_model.instances[0].segments = [
        segment("a b", words=[(" a", 0.0, 0.4), (" b", 0.4, 0.8)]),
        segment("", words=None),
        segment("c", words=[(" c ", 1.0, 1.3)]),
    ]
    assert asr.transcribe(silence()).words == (
        Word("a", 0.0, 0.4),
        Word("b", 0.4, 0.8),
        Word("c", 1.0, 1.3),
    )


def test_auto_detect_returns_the_detected_language(fake_model: type[FakeWhisperModel]) -> None:
    asr = WhisperAsr()
    fake_model.instances[0].language = "en"
    transcript = asr.transcribe(silence(), language=None)
    assert fake_model.instances[0].calls[0]["language"] is None
    assert transcript.language == "en"


def test_no_segments_gives_an_empty_transcript(fake_model: type[FakeWhisperModel]) -> None:
    asr = WhisperAsr()
    fake_model.instances[0].segments = []
    assert asr.transcribe(silence()) == Transcript(
        text="",
        language="bn",
        worst_avg_logprob=float("-inf"),
        max_no_speech_prob=1.0,
        max_compression_ratio=0.0,
        max_segment_tokens=0,
        words=(),
    )


@pytest.mark.parametrize(
    "audio",
    [np.zeros(SAMPLE_RATE_HZ, dtype=np.float64), np.zeros((1, SAMPLE_RATE_HZ), dtype=np.float32)],
    ids=["float64", "2-d"],
)
def test_transcribe_rejects_wrong_input(
    fake_model: type[FakeWhisperModel], audio: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        WhisperAsr().transcribe(audio)


@pytest.fixture(scope="module")
def tiny_asr() -> WhisperAsr:
    return WhisperAsr("tiny", device="cpu", compute_type="int8")


def test_tiny_model_transcribes_real_speech(tiny_asr: WhisperAsr) -> None:
    speech = load_mono_16k(SPEECH_PATH)
    transcript = tiny_asr.transcribe(speech, language="bn")
    assert transcript.text
    assert transcript.language == "bn"
    assert transcript.worst_avg_logprob <= 0
    assert 0 <= transcript.max_no_speech_prob <= 1
    assert transcript.words == ()


def test_tiny_model_returns_ordered_word_timings_on_request(tiny_asr: WhisperAsr) -> None:
    speech = load_mono_16k(SPEECH_PATH)
    transcript = tiny_asr.transcribe(speech, language="bn", word_timestamps=True)
    assert transcript.words
    duration_s = speech.size / SAMPLE_RATE_HZ
    for word in transcript.words:
        assert 0 <= word.start <= word.end <= duration_s
    starts = [word.start for word in transcript.words]
    assert starts == sorted(starts)
