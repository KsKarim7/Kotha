import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import soundfile as sf

from kotha.audio.convert import SAMPLE_RATE_HZ, resample
from kotha.dialogue.routing import Language
from kotha.tts.mms import VOICES, MmsVoice
from kotha.tts.speakable import speakable

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTENCES_PATH = REPO_ROOT / "data" / "listening" / "tts_sentences.csv"
SENTENCE_COLUMNS = ("id", "lang", "category", "text")
VERSIONS = ("before", "after")
LISTENERS = 3
SHUFFLE_SEED = 0


@dataclass(frozen=True)
class ListeningSentence:
    id: str
    lang: Language
    category: str
    text: str


@dataclass(frozen=True)
class Clip:
    id: str
    sentence: ListeningSentence
    version: str


def read_sentences(path: Path) -> list[ListeningSentence]:
    with path.open(newline="", encoding="utf-8") as sentences_file:
        reader = csv.DictReader(sentences_file)
        missing = [name for name in SENTENCE_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        sentences = [
            read_sentence(row, f"{path}, line {line}") for line, row in enumerate(reader, 2)
        ]
    ids = [sentence.id for sentence in sentences]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has repeated sentence ids")
    return sentences


def read_sentence(row: dict[str, str | None], location: str) -> ListeningSentence:
    fields = {name: (row.get(name) or "").strip() for name in SENTENCE_COLUMNS}
    if fields["lang"] not in VOICES:
        raise ValueError(f"{location}: unknown lang {fields['lang']!r}")
    empty = [name for name in ("id", "category", "text") if not fields[name]]
    if empty:
        raise ValueError(f"{location}: {', '.join(empty)} must not be empty")
    lang = cast(Language, fields["lang"])
    return ListeningSentence(fields["id"], lang, fields["category"], fields["text"])


def voice_input(clip: Clip) -> str:
    text = clip.sentence.text
    return text if clip.version == "before" else speakable(text, clip.sentence.lang)


def clip_plan(sentences: list[ListeningSentence]) -> list[Clip]:
    pairs = [(sentence, version) for sentence in sentences for version in VERSIONS]
    random.Random(SHUFFLE_SEED).shuffle(pairs)
    return [
        Clip(f"clip-{number:02d}", sentence, version)
        for number, (sentence, version) in enumerate(pairs, start=1)
    ]


def write_rating_sheet(path: Path, clips: list[Clip]) -> None:
    listeners = [f"listener_{number}" for number in range(1, LISTENERS + 1)]
    with path.open("w", newline="", encoding="utf-8") as sheet:
        writer = csv.writer(sheet)
        writer.writerow(["clip", "lang", "intended_text", *listeners])
        for clip in clips:
            writer.writerow([clip.id, clip.sentence.lang, clip.sentence.text, *[""] * LISTENERS])


def write_key(path: Path, clips: list[Clip]) -> None:
    with path.open("w", newline="", encoding="utf-8") as key:
        writer = csv.writer(key)
        writer.writerow(["clip", "sentence_id", "category", "version", "voice_input"])
        for clip in clips:
            sentence = clip.sentence
            writer.writerow(
                [clip.id, sentence.id, sentence.category, clip.version, voice_input(clip)]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the TTS listening test: each sentence before and after normalisation."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--key",
        type=Path,
        required=True,
        help="CSV naming each clip's version, kept from listeners",
    )
    parser.add_argument("--sentences", type=Path, default=SENTENCES_PATH)
    args = parser.parse_args()
    output_dir: Path = args.output_dir.resolve()
    key_path: Path = args.key.resolve()
    if output_dir.is_relative_to(REPO_ROOT) or key_path.is_relative_to(REPO_ROOT):
        parser.error("--output-dir and --key must be outside the repository")
    if key_path.is_relative_to(output_dir):
        parser.error("--key must be outside --output-dir, which is shared with listeners")
    clips = clip_plan(read_sentences(args.sentences))
    output_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    voices = {lang: MmsVoice(lang) for lang in VOICES}
    for clip in clips:
        voice = voices[clip.sentence.lang]
        audio = resample(voice.synthesize(voice_input(clip)), voice.sampling_rate, SAMPLE_RATE_HZ)
        sf.write(output_dir / f"{clip.id}.wav", audio, SAMPLE_RATE_HZ, subtype="PCM_16")
    write_rating_sheet(output_dir / "ratings.csv", clips)
    write_key(key_path, clips)
    print(f"{len(clips)} clips and the rating sheet written to {output_dir}, the key to {key_path}")


if __name__ == "__main__":
    main()
