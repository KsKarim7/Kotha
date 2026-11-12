from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

DECODING_OPTIONS: dict[str, float | bool] = {
    "temperature": 0.0,
    "without_timestamps": True,
    "condition_on_previous_text": False,
    "vad_filter": False,
}


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    worst_avg_logprob: float
    max_no_speech_prob: float
    max_compression_ratio: float
    max_segment_tokens: int
    words: tuple[Word, ...]


def join_segment_texts(segments: Iterable[Segment]) -> str:
    segment_texts = (segment.text.strip() for segment in segments)
    return " ".join(text for text in segment_texts if text)


class WhisperAsr:
    def __init__(
        self, model: str = "small", *, device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_16k: np.ndarray,
        *,
        language: str | None = "bn",
        beam_size: int = 1,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> Transcript:
        if audio_16k.dtype != np.float32 or audio_16k.ndim != 1:
            raise ValueError(
                f"expected a 1-D float32 array, got {audio_16k.dtype} {audio_16k.shape}"
            )
        segment_stream, info = self._model.transcribe(
            audio_16k,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            **DECODING_OPTIONS,
        )
        segments = list(segment_stream)
        if not segments:
            return Transcript(
                text="",
                language=info.language,
                worst_avg_logprob=float("-inf"),
                max_no_speech_prob=1.0,
                max_compression_ratio=0.0,
                max_segment_tokens=0,
                words=(),
            )
        return Transcript(
            text=join_segment_texts(segments),
            language=info.language,
            worst_avg_logprob=min(segment.avg_logprob for segment in segments),
            max_no_speech_prob=max(segment.no_speech_prob for segment in segments),
            max_compression_ratio=max(segment.compression_ratio for segment in segments),
            max_segment_tokens=max(len(segment.tokens) for segment in segments),
            words=tuple(
                Word(word.word.strip(), float(word.start), float(word.end))
                for segment in segments
                for word in segment.words or []
            ),
        )
