import re
import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pytest

from kotha.kb.documents import LANGS, read_document, read_documents

KB_FOLDER = Path(__file__).resolve().parent.parent / "data" / "kb"
PACKAGE_CODES = frozenset(
    "NT-1GB NT-5GB NT-15GB NT-30GB NT-60GB NT-NIGHT-5GB NV-100 NV-300 NC-10GB NC-25GB".split()
)
USSD_CODES = frozenset({"*747#", "*747*1#", "*747*2#", "*747*3#", "*747*4#"}) | {
    "*747*5*<voucher number>#",
    "*747*5*<ভাউচার নম্বর>#",
}
SHORT_NUMBERS = frozenset({"747", "7470"})
EXAMPLE_NUMBERS = frozenset(f"0170000000{digit}" for digit in range(10))
EXAMPLE_TICKETS = frozenset(f"TKT-00000{digit}" for digit in range(1, 10))
BLOCKED_LATIN = (
    "grameenphone robi airtel banglalink teletalk citycell skitto mygp mybl bkash nagad upay"
).split()
BLOCKED_BENGALI = ("গ্রামীণফোন", "এয়ারটেল", "বাংলালিংক", "টেলিটক", "সিটিসেল")
BENGALI_BLOCK = range(0x0980, 0x0A00)
TO_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
MONEY = re.compile(
    r"৳|[0-9০-৯]\s*(?:টাকা|(?:taka|tk|bdt)\b)|(?:টাকা|\b(?:taka|tk|bdt))\s*[0-9০-৯]", re.I
)


def file_problems(folder: Path, pattern: re.Pattern[str], allowed: frozenset[str]) -> list[str]:
    return [
        f"{document.path.name}: {match}"
        for document in read_documents(folder)
        for match in pattern.findall(document.body)
        if match not in allowed
    ]


def unreadable_files(folder: Path) -> list[str]:
    problems = []
    for path in sorted(folder.glob("*.*.md")):
        try:
            read_document(path)
        except ValueError as error:
            problems.append(str(error))
    return problems


def missing_translations(folder: Path) -> list[str]:
    langs: defaultdict[str, set[str]] = defaultdict(set)
    for document in read_documents(folder):
        langs[document.doc_id].add(document.lang)
    return [
        f"{doc_id}: only {sorted(found)}" for doc_id, found in langs.items() if found != set(LANGS)
    ]


def section_count_mismatches(folder: Path) -> list[str]:
    counts: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for document in read_documents(folder):
        sections = sum(line.startswith("## ") for line in document.body.splitlines())
        counts[document.doc_id][document.lang] = sections
    return [
        f"{doc_id}: {by_lang}"
        for doc_id, by_lang in counts.items()
        if len(set(by_lang.values())) > 1
    ]


def script_problems(folder: Path) -> list[str]:
    problems = []
    for document in read_documents(folder):
        letters = [char for char in document.body if char.isalpha()]
        bengali = sum(ord(char) in BENGALI_BLOCK for char in letters)
        if document.lang == "en" and bengali:
            problems.append(f"{document.path.name}: {bengali} Bengali letters")
        if document.lang == "bn" and bengali < 0.6 * len(letters):
            problems.append(
                f"{document.path.name}: only {bengali} of {len(letters)} letters Bengali"
            )
    return problems


def money_mentions(folder: Path) -> list[str]:
    return file_problems(folder, MONEY, frozenset())


def brand_mentions(folder: Path) -> list[str]:
    latin = re.compile(rf"\b(?:{'|'.join(BLOCKED_LATIN)})\b", re.I)
    problems = file_problems(folder, latin, frozenset())
    for document in read_documents(folder):
        problems += [
            f"{document.path.name}: {name}" for name in BLOCKED_BENGALI if name in document.body
        ]
    return problems


def unknown_package_codes(folder: Path) -> list[str]:
    return file_problems(folder, re.compile(r"\bN[TVC](?:-[A-Z0-9]+)+\b"), PACKAGE_CODES)


def unknown_short_codes(folder: Path) -> list[str]:
    ussd = file_problems(folder, re.compile(r"\*\d[\d*]*(?:<[^>]*>)?#"), USSD_CODES)
    return ussd + file_problems(folder, re.compile(r"`(\d{3,5})`"), SHORT_NUMBERS)


def unknown_ticket_numbers(folder: Path) -> list[str]:
    return [
        problem
        for problem in file_problems(folder, re.compile(r"TKT-[0-9০-৯]+"), frozenset())
        if problem.split(": ")[1].translate(TO_ASCII_DIGITS) not in EXAMPLE_TICKETS
    ]


def bad_phone_numbers(folder: Path) -> list[str]:
    phone = re.compile(r"(?<![0-9০-৯])[0০][1১][0-9০-৯]{9}(?![0-9০-৯])")
    return [
        problem
        for problem in file_problems(folder, phone, frozenset())
        if problem.split(": ")[1].translate(TO_ASCII_DIGITS) not in EXAMPLE_NUMBERS
    ]


CHECKS: list[Callable[[Path], list[str]]] = [
    unreadable_files,
    missing_translations,
    section_count_mismatches,
    script_problems,
    money_mentions,
    brand_mentions,
    unknown_package_codes,
    unknown_short_codes,
    unknown_ticket_numbers,
    bad_phone_numbers,
]


@pytest.mark.parametrize("check", CHECKS)
def test_the_knowledge_base_passes(check: Callable[[Path], list[str]]) -> None:
    assert check(KB_FOLDER) == []


def append(name: str, text: str) -> Callable[[Path], None]:
    def add(folder: Path) -> None:
        with (folder / name).open("a", encoding="utf-8") as kb_file:
            kb_file.write(text)

    return add


def write_bad_file(folder: Path) -> None:
    (folder / "broken.en.md").write_text("no front matter\n", encoding="utf-8")


def remove_bn_file(folder: Path) -> None:
    (folder / "validity.bn.md").unlink()


@pytest.mark.parametrize(
    ("check", "breakage"),
    [
        (unreadable_files, write_bad_file),
        (missing_translations, remove_bn_file),
        (section_count_mismatches, append("validity.en.md", "\n## Extra\n\nMore text.\n")),
        (script_problems, append("validity.en.md", "\nমেয়াদ\n")),
        (money_mentions, append("validity.bn.md", "\nদাম ৫০ টাকা।\n")),
        (brand_mentions, append("validity.en.md", "\nLike Grameenphone does.\n")),
        (unknown_package_codes, append("validity.en.md", "\nBuy NT-2GB.\n")),
        (unknown_short_codes, append("validity.en.md", "\nDial `*747*9#`.\n")),
        (unknown_ticket_numbers, append("validity.en.md", "\nYour ticket is TKT-123456.\n")),
        (bad_phone_numbers, append("validity.en.md", "\nCall 01711111111.\n")),
    ],
)
def test_each_check_catches_its_problem(
    tmp_path: Path,
    check: Callable[[Path], list[str]],
    breakage: Callable[[Path], None],
) -> None:
    folder = tmp_path / "kb"
    shutil.copytree(KB_FOLDER, folder)
    breakage(folder)
    assert check(folder)
