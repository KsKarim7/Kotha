from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel


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
    words: tuple[Word, ...]


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
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=False,
        )
        segments = list(segment_stream)
        if not segments:
            return Transcript("", info.language, float("-inf"), 1.0, ())
        segment_texts = (segment.text.strip() for segment in segments)
        return Transcript(
            text=" ".join(text for text in segment_texts if text),
            language=info.language,
            worst_avg_logprob=min(segment.avg_logprob for segment in segments),
            max_no_speech_prob=max(segment.no_speech_prob for segment in segments),
            words=tuple(
                Word(word.word.strip(), float(word.start), float(word.end))
                for segment in segments
                for word in segment.words or []
            ),
        )
