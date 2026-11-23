import argparse
import csv
import hashlib
import time
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from kotha.asr.whisper import WhisperAsr
from kotha.dialogue.routing import Language
from kotha.tts.mms import SEED, VOICES, MmsVoice
from kotha.tts.speakable import speakable_sentences
from listening_sheet import SENTENCES_PATH, read_sentences
from tracking import git_tags, log_run, open_experiment
from wer import latency_percentiles

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "kotha-tts"


@dataclass(frozen=True)
class ProbeSentence:
    id: str
    lang: Language
    text: str


SENTENCES = (
    ProbeSentence("bn-plain", "bn", "সিম বদলাতে নিকটের সেবাকেন্দ্রে যান।"),
    ProbeSentence("bn-latin-code", "bn", "প্যাক NT-30GB সাত দিন চলবে।"),
    ProbeSentence("bn-latin-words", "bn", "SIM বদলাতে NID কার্ড লাগবে।"),
    ProbeSentence("bn-ascii-digits", "bn", "প্যাকের মেয়াদ 30 দিন।"),
    ProbeSentence("bn-bangla-digits", "bn", "প্যাকের মেয়াদ ৩০ দিন।"),
    ProbeSentence("bn-number-words", "bn", "প্যাকের মেয়াদ ত্রিশ দিন।"),
    ProbeSentence("bn-precomposed-ya", "bn", "প্যাকের মেয়াদ ত্রিশ দিন।"),
    ProbeSentence("bn-short-code", "bn", "প্যাক দেখতে *747# ডায়াল করুন।"),
    ProbeSentence("bn-loanwords-bangla", "bn", "সিম বদলাতে এনআইডি কার্ড আর অ্যাপ লাগবে।"),
    ProbeSentence("en-plain", "en", "Please visit the nearest service centre to replace your SIM."),
    ProbeSentence("en-digits-in-vocab", "en", "The pack lasts 30 days."),
    ProbeSentence("en-digits-not-in-vocab", "en", "The pack gives 8 GB for 7 days."),
    ProbeSentence("en-number-words", "en", "The pack gives eight gigabytes for seven days."),
    ProbeSentence("en-short-code", "en", "Dial *747# to see the packs."),
    ProbeSentence("en-bangla-word", "en", "Ask at any নদী service centre."),
)


@dataclass(frozen=True)
class ProbeResult:
    sentence: ProbeSentence
    spoken_text: str
    dropped: tuple[str, ...]
    audio_s: float
    synthesis_s: float
    heard: str | None


def dropped_characters(text: str, vocab: Collection[str], blank: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(sorted({char for char in lowered if char not in vocab or char == blank}))


def character_label(char: str) -> str:
    return f"U+{ord(char):04X} {unicodedata.name(char, '?')}"


def probe(
    voice: MmsVoice, sentence: ProbeSentence, output_dir: Path, asr: WhisperAsr | None
) -> ProbeResult:
    spoken_text, _ = voice.tokenizer.prepare_for_tokenization(sentence.text)
    started = time.perf_counter()
    audio = voice.synthesize(sentence.text)
    synthesis_s = time.perf_counter() - started
    sf.write(output_dir / f"{sentence.id}.wav", audio, voice.sampling_rate, subtype="PCM_16")
    heard = asr.transcribe(audio, language=sentence.lang).text if asr else None
    return ProbeResult(
        sentence=sentence,
        spoken_text=spoken_text,
        dropped=dropped_characters(
            sentence.text, voice.tokenizer.get_vocab(), voice.tokenizer.pad_token
        ),
        audio_s=len(audio) / voice.sampling_rate,
        synthesis_s=synthesis_s,
        heard=heard,
    )


def print_result(result: ProbeResult) -> None:
    rtf = f"{result.synthesis_s / result.audio_s:.3f}" if result.audio_s else "n/a"
    print(
        f"{result.sentence.id}: audio {result.audio_s:.2f} s, synthesis "
        f"{result.synthesis_s:.3f} s, RTF {rtf}"
    )
    print(f"  input:  {result.sentence.text}")
    print(f"  spoken: {result.spoken_text}")
    for char in result.dropped:
        print(f"  dropped: {character_label(char)}")
    if result.heard is not None:
        print(f"  heard:  {result.heard}")


def listening_sentences() -> list[ProbeSentence]:
    # the app speaks the normalised text, so that is what the voice is timed on
    return [
        ProbeSentence(
            sentence.id, sentence.lang, " ".join(speakable_sentences(sentence.text, sentence.lang))
        )
        for sentence in read_sentences(SENTENCES_PATH)
    ]


def real_time_factors(results: Sequence[ProbeResult]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for lang in VOICES:
        factors = [
            result.synthesis_s / result.audio_s
            for result in results
            if result.sentence.lang == lang and result.audio_s
        ]
        if factors:
            p50, p95 = latency_percentiles(factors)
            metrics |= {f"rtf.{lang}.p50": p50, f"rtf.{lang}.p95": p95}
            metrics[f"rtf.{lang}.sentences"] = len(factors)
    return metrics


def write_timings(path: Path, results: Sequence[ProbeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("id", "lang", "audio_s", "synthesis_s", "rtf"))
        for result in results:
            rtf = result.synthesis_s / result.audio_s if result.audio_s else ""
            writer.writerow(
                (result.sentence.id, result.sentence.lang, result.audio_s, result.synthesis_s, rtf)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the MMS TTS voices on fixed sentences.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--asr-model", help="faster-whisper model that transcribes each clip back, as a hint"
    )
    parser.add_argument(
        "--sentences",
        choices=("probe", "listening"),
        default="probe",
        help="the probe's fixed sentences, or the listening sheet's as the app would speak them",
    )
    parser.add_argument("--mlflow-dir", type=Path)
    args = parser.parse_args()
    output_dir: Path = args.output_dir.resolve()
    if output_dir.is_relative_to(REPO_ROOT):
        parser.error(f"--output-dir {output_dir} must be outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = git_tags(REPO_ROOT)
    sentences = listening_sentences() if args.sentences == "listening" else list(SENTENCES)

    import torch

    asr = WhisperAsr(args.asr_model) if args.asr_model else None
    print(f"torch threads: {torch.get_num_threads()}, seed: {SEED}, asr: {args.asr_model}")
    results: list[ProbeResult] = []
    for lang, (model_id, revision) in VOICES.items():
        voice = MmsVoice(lang)
        print(f"voice {lang}: {model_id}@{revision[:8]}, {voice.sampling_rate} Hz")
        voice.synthesize("warm up" if lang == "en" else "শুরু")
        for sentence in sentences:
            if sentence.lang == lang:
                results.append(probe(voice, sentence, output_dir, asr))
                print_result(results[-1])
    metrics = real_time_factors(results)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    timings = output_dir / "timings.csv"
    write_timings(timings, results)
    if args.mlflow_dir:
        client, experiment = open_experiment(args.mlflow_dir, EXPERIMENT_NAME)
        params = {
            "sentences": args.sentences,
            "device": "cpu",
            "torch_threads": str(torch.get_num_threads()),
            "seed": str(SEED),
            **{
                f"voice_{lang}": f"{model}@{revision}" for lang, (model, revision) in VOICES.items()
            },
        }
        if args.sentences == "listening":
            params["sentences_sha256"] = hashlib.sha256(SENTENCES_PATH.read_bytes()).hexdigest()
        run_id = log_run(client, experiment, params, metrics, tags, [timings])
        print(f"mlflow run: {run_id}")


if __name__ == "__main__":
    main()
