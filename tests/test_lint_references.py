import csv
from pathlib import Path

import pytest

from lint_references import FILLER_WORDS, main
from wer import MANIFEST_COLUMNS, REPO_ROOT

MARK_BN = "গোপনকথা"
MARK_EN = "zqxsecret"
GUIDELINE_PATH = REPO_ROOT / "data" / "testsets" / "GUIDELINE.md"


def guideline_examples() -> list[tuple[str, str]]:
    text = GUIDELINE_PATH.read_text(encoding="utf-8")
    table = text.split("## Examples", 1)[1].strip().splitlines()[2:]
    examples = []
    for line in table:
        _, reference, lang = (cell.strip() for cell in line.strip("|").split("|"))
        examples.append((reference, lang))
    return examples


def write_manifest(path: Path, references: list[tuple[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for number, (text, lang) in enumerate(references, start=1):
            writer.writerow(
                {
                    "id": f"k{number}",
                    "path": f"k{number}.wav",
                    "speaker": "spk01",
                    "lang": lang,
                    "noise": "clean",
                    "split": "dev",
                    "text": text,
                }
            )
    return path


def run(
    capsys: pytest.CaptureFixture[str], manifests: list[Path], *options: str
) -> tuple[int, list[str]]:
    exit_code = main(["--manifest", *map(str, manifests), *options])
    output = capsys.readouterr().out
    assert MARK_BN not in output
    assert MARK_EN not in output
    return exit_code, output.splitlines()


def lint_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, lang: str
) -> tuple[int, list[str]]:
    manifest = write_manifest(tmp_path / "m.csv", [(text, lang)])
    exit_code, lines = run(capsys, [manifest])
    assert not text or text not in "\n".join(lines)
    return exit_code, lines


def test_the_guideline_examples_pass_in_two_manifests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = guideline_examples()
    assert len(examples) == 6
    public = write_manifest(tmp_path / "public.csv", examples[:3])
    private = write_manifest(tmp_path / "private.csv", examples[3:])

    exit_code, lines = run(capsys, [public, private])

    assert exit_code == 0
    assert lines == ["0 errors, 0 warnings"]


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        (f"{MARK_BN} 30 টাকা", "bn"),
        (f"{MARK_EN} 01700000000", "en"),
        (f"{MARK_BN} {MARK_EN} হয়নি", "mix"),
    ],
)
def test_digits_and_a_decomposed_yya_are_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, lang: str
) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, text, lang)

    assert exit_code == 0
    assert lines == ["0 errors, 0 warnings"]


@pytest.mark.parametrize(
    ("bad", "offset", "description"),
    [
        (chr(0x0964), 1, "U+0964 DEVANAGARI DANDA (punctuation)"),
        (",", 1, "U+002C COMMA (punctuation)"),
        (chr(0x09E9), 1, "U+09E9 BENGALI DIGIT THREE (Bangla digit)"),
        ("K", 1, "U+004B LATIN CAPITAL LETTER K (uppercase letter)"),
        (" ", 2, "U+0020 SPACE (extra or odd whitespace)"),
        (chr(0x00A0), 1, "U+00A0 NO-BREAK SPACE (extra or odd whitespace)"),
        (chr(0x200C), 1, "U+200C ZERO WIDTH NON-JOINER (invisible character)"),
        (chr(0x09DF), 1, "U+09DF BENGALI LETTER YYA (not NFC)"),
    ],
)
def test_a_reference_that_is_not_normalised_reports_the_first_difference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad: str,
    offset: int,
    description: str,
) -> None:
    prefix = f"{MARK_BN} {MARK_EN}"

    exit_code, lines = lint_one(tmp_path, capsys, f"{prefix}{bad} আর", "mix")

    assert exit_code == 1
    position = len(prefix) + offset
    assert lines[0] == f"m.csv:2 k1: not normalised at position {position}: {description}"


@pytest.mark.parametrize(
    ("text", "description"),
    [
        (f" {MARK_BN}", "position 1: U+0020 SPACE (extra or odd whitespace)"),
        (f"{MARK_BN} ", f"position {len(MARK_BN) + 1}: U+0020 SPACE (extra or odd whitespace)"),
        (
            f"{MARK_BN}\tআর",
            f"position {len(MARK_BN) + 1}: U+0009 no name (extra or odd whitespace)",
        ),
    ],
)
def test_leading_trailing_and_tab_whitespace_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, description: str
) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, text, "bn")

    assert exit_code == 1
    assert lines[0] == f"m.csv:2 k1: not normalised at {description}"


def test_an_empty_reference_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, "", "bn")

    assert exit_code == 1
    assert lines == ["m.csv:2 k1: empty reference", "1 errors, 0 warnings"]


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        (f"{MARK_BN} {MARK_EN}", "bn", "lang bn contains Latin letters"),
        (f"{MARK_BN} {MARK_EN}", "en", "lang en contains Bengali letters"),
        (f"{MARK_BN} আমার 30", "mix", "lang mix needs both Bengali and Latin letters"),
        (f"{MARK_EN} check 30", "mix", "lang mix needs both Bengali and Latin letters"),
    ],
)
def test_the_lang_column_must_match_the_letters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, lang: str, expected: str
) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, text, lang)

    assert exit_code == 1
    assert lines == [f"m.csv:2 k1: {expected}", "1 errors, 0 warnings"]


def test_a_loanword_in_bengali_script_names_its_latin_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, f"{MARK_BN} {MARK_EN} আমার মোবাইল", "mix")

    assert exit_code == 1
    assert lines == [
        "m.csv:2 k1: word 4: loanword in Bengali script, write it as 'mobile' (rule 2)",
        "1 errors, 0 warnings",
    ]


def test_the_lexicon_option_replaces_the_committed_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lexicon = tmp_path / "lexicon.csv"
    lexicon.write_text(f"latin,bangla\nhidden,{MARK_BN}\n", encoding="utf-8")
    manifest = write_manifest(tmp_path / "m.csv", [(f"{MARK_BN} {MARK_EN} মোবাইল", "mix")])

    exit_code, lines = run(capsys, [manifest], "--lexicon", str(lexicon))

    assert exit_code == 1
    assert lines == [
        "m.csv:2 k1: word 1: loanword in Bengali script, write it as 'hidden' (rule 2)",
        "1 errors, 0 warnings",
    ]


def test_the_taka_sign_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, f"{MARK_BN} {chr(0x09F3)} 30", "bn")

    assert exit_code == 1
    assert lines == [
        "m.csv:2 k1: taka sign ৳, write the word টাকা (rule 3)",
        "1 errors, 0 warnings",
    ]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("30gb", "word 3: number and unit joined (rule 3)"),
        (
            "packageটা",
            "word 3: Latin word with a Bangla suffix attached, write the suffix as a separate "
            "word (rule 4)",
        ),
    ],
)
def test_joined_words_are_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], word: str, expected: str
) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, f"{MARK_BN} {MARK_EN} {word}", "mix")

    assert exit_code == 0
    assert lines == [f"m.csv:2 k1: warning: {expected}", "0 errors, 1 warnings"]


@pytest.mark.parametrize("filler", sorted(FILLER_WORDS))
def test_a_filler_word_is_a_warning_by_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], filler: str
) -> None:
    exit_code, lines = lint_one(tmp_path, capsys, f"{MARK_BN} {MARK_EN} {filler} আর", "mix")

    assert exit_code == 0
    assert lines == [
        "m.csv:2 k1: warning: word 3: filler word, leave it out (rule 5)",
        "0 errors, 1 warnings",
    ]


def test_problems_name_the_manifest_line_and_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    public = write_manifest(tmp_path / "public.csv", [(f"{MARK_BN} আর", "bn")])
    private = write_manifest(
        tmp_path / "private.csv", [(f"{MARK_BN} আর", "bn"), (f"{MARK_BN} আর", "en")]
    )

    exit_code, lines = run(capsys, [public, private])

    assert exit_code == 1
    assert lines == ["private.csv:3 k2: lang en contains Bengali letters", "1 errors, 0 warnings"]


def test_a_manifest_without_the_required_columns_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "bad.csv"
    manifest.write_text(f"id,text\nk1,{MARK_BN}\n", encoding="utf-8")

    exit_code, lines = run(capsys, [manifest])

    assert exit_code == 1
    assert lines[0].startswith("bad.csv: manifest ")
    assert lines[0].endswith("is missing columns: path, speaker, lang, noise, split")
    assert lines[-1] == "1 errors, 0 warnings"
