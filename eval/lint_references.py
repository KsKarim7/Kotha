import argparse
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

from kotha.text.lexicon import BENGALI_BLOCK, LEXICON_PATH, read_lexicon
from kotha.text.normalize import normalize
from wer import ManifestRow, read_manifest

TAKA_SIGN = chr(0x09F3)
BANGLA_DIGITS = frozenset(chr(code_point) for code_point in range(0x09E6, 0x09F0))
FILLER_WORDS = frozenset({"আ", "উম", "এ", "uh", "um", "hmm"})


def is_latin_letter(char: str) -> bool:
    return char.isalpha() and unicodedata.name(char, "").startswith("LATIN")


def is_bengali_letter(char: str) -> bool:
    return ord(char) in BENGALI_BLOCK and unicodedata.category(char).startswith("L")


def first_difference(reference: str, normalised: str) -> int:
    for index, (char, expected) in enumerate(zip(reference, normalised, strict=False)):
        if char != expected:
            return index
    return min(len(normalised), len(reference) - 1)


def difference_kind(char: str) -> str:
    category = unicodedata.category(char)
    if category == "Cf":
        return "invisible character"
    if category.startswith("P"):
        return "punctuation"
    if char in BANGLA_DIGITS:
        return "Bangla digit"
    if char != char.lower():
        return "uppercase letter"
    if char.isspace():
        return "extra or odd whitespace"
    return "not NFC"


def normalisation_error(reference: str, normalised: str) -> str:
    index = first_difference(reference, normalised)
    char = reference[index]
    name = unicodedata.name(char, "no name")
    return (
        f"not normalised at position {index + 1}: U+{ord(char):04X} {name} "
        f"({difference_kind(char)})"
    )


def lang_error(lang: str | None, reference: str) -> str | None:
    has_latin = any(is_latin_letter(char) for char in reference)
    has_bengali = any(is_bengali_letter(char) for char in reference)
    if lang == "bn" and has_latin:
        return "lang bn contains Latin letters"
    if lang == "en" and has_bengali:
        return "lang en contains Bengali letters"
    if lang == "mix" and not (has_latin and has_bengali):
        return "lang mix needs both Bengali and Latin letters"
    return None


def loanword_errors(words: Sequence[str], lexicon: Mapping[str, str]) -> list[str]:
    return [
        f"word {number}: loanword in Bengali script, write it as '{lexicon[word]}' (rule 2)"
        for number, word in enumerate(words, start=1)
        if word in lexicon
    ]


def word_warnings(words: Sequence[str]) -> list[str]:
    warnings = []
    for number, word in enumerate(words, start=1):
        if any(char.isdigit() for char in word) and any(char.isalpha() for char in word):
            warnings.append(f"word {number}: number and unit joined (rule 3)")
        if any(map(is_latin_letter, word)) and any(map(is_bengali_letter, word)):
            warnings.append(
                f"word {number}: Latin word with a Bangla suffix attached, "
                "write the suffix as a separate word (rule 4)"
            )
        if word in FILLER_WORDS:
            warnings.append(f"word {number}: filler word, leave it out (rule 5)")
    return warnings


def row_problems(row: ManifestRow, lexicon: Mapping[str, str]) -> tuple[list[str], list[str]]:
    if not row.text:
        return ["empty reference"], []
    errors = []
    normalised = normalize(row.text)
    if normalised != row.text:
        errors.append(normalisation_error(row.text, normalised))
    if TAKA_SIGN in row.text:
        errors.append(f"taka sign {TAKA_SIGN}, write the word টাকা (rule 3)")
    lang_problem = lang_error(row.lang, row.text)
    if lang_problem:
        errors.append(lang_problem)
    words = normalised.split()
    return errors + loanword_errors(words, lexicon), word_warnings(words)


def lint(manifest_paths: Sequence[Path], lexicon: Mapping[str, str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for manifest_path in manifest_paths:
        try:
            rows = read_manifest(manifest_path)
        except ValueError as error:
            errors.append(f"{manifest_path.name}: {error}")
            continue
        for line, row in enumerate(rows, start=2):
            location = f"{manifest_path.name}:{line} {row.id or '-'}"
            row_errors, row_warnings = row_problems(row, lexicon)
            errors += [f"{location}: {error}" for error in row_errors]
            warnings += [f"{location}: warning: {warning}" for warning in row_warnings]
    return errors, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check that manifest references follow the transcription guideline. "
            "Reference text is never printed."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help="manifest CSVs, for example the public one and an optional private one",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=LEXICON_PATH,
        help="loanword lexicon CSV (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors, warnings = lint(args.manifest, read_lexicon(args.lexicon))
    for line in [*errors, *warnings]:
        print(line)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
