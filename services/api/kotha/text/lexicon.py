import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from kotha.text.normalize import normalize

LEXICON_PATH = Path(__file__).resolve().parents[4] / "data" / "lexicon" / "loanwords.csv"
LEXICON_COLUMNS = ("latin", "bangla")
BENGALI_BLOCK = range(0x0980, 0x0A00)


def read_lexicon(path: Path) -> dict[str, str]:
    lexicon: dict[str, str] = {}
    first_lines: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as lexicon_file:
        reader = csv.DictReader(lexicon_file)
        missing = [column for column in LEXICON_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"lexicon {path} is missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            location = f"lexicon {path}, line {line}"
            latin = lexicon_word(row["latin"], "latin", location)
            bangla = lexicon_word(row["bangla"], "bangla", location)
            if any(ord(char) in BENGALI_BLOCK for char in latin):
                raise ValueError(
                    f"{location}: latin must not contain Bengali letters, got {latin!r}"
                )
            if lexicon.get(bangla, latin) != latin:
                raise ValueError(
                    f"{location}: {bangla!r} maps to {latin!r} here and to {lexicon[bangla]!r}"
                    f" on line {first_lines[bangla]}"
                )
            lexicon[bangla] = latin
            first_lines.setdefault(bangla, line)
    return lexicon


def lexicon_word(value: str | None, column: str, location: str) -> str:
    word = normalize(value or "")
    if len(word.split()) != 1:
        raise ValueError(f"{location}: {column} must be one word after normalising, got {word!r}")
    return word


def apply_lexicon(normalised_text: str, lexicon: Mapping[str, str]) -> str:
    return " ".join(lexicon.get(token, token) for token in normalised_text.split())


def spelling_groups(lexicon: Mapping[str, str]) -> dict[str, frozenset[str]]:
    spellings_by_latin: defaultdict[str, set[str]] = defaultdict(set)
    for bangla, latin in lexicon.items():
        spellings_by_latin[latin].update((latin, bangla))
    return {
        spelling: frozenset(group) for group in spellings_by_latin.values() for spelling in group
    }


def expand_tokens(tokens: Sequence[str], groups: Mapping[str, frozenset[str]]) -> list[str]:
    variants = {variant for token in tokens for variant in groups.get(token, ())}
    return [*tokens, *sorted(variants - set(tokens))]
