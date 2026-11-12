import pytest

from kotha.text.normalize import normalize

EXPECTED = [
    (
        "আমার mobile balance কত, আর 30GB internet package টা activate করতে চাই।",
        "আমার mobile balance কত আর 30gb internet package টা activate করতে চাই",
    ),
    ("০১২৩৪৫৬৭৮৯", "0123456789"),
    ("৩০GB", "30gb"),
    ("কর\u200cতে", "করতে"),
    ("\ufeffinter\u00adnet", "internet"),
    ("ক\u0964খ", "ক খ"),
    ("ক\u0965খ", "ক খ"),
    ("ক\u2014খ", "ক খ"),
    ("ক\u2026খ", "ক খ"),
    ("\u201cক\u201d \u2018খ\u2019", "ক খ"),
    ("ক?খ!", "ক খ"),
    ("NT-30GB", "nt 30gb"),
    ("50%", "50"),
    ("\u09f3299", "\u09f3299"),
    ("a+b", "a+b"),
    ("a\tb\nc\u00a0d\u2009e", "a b c d e"),
    ("  ক  খ  ", "ক খ"),
    ("", ""),
    (" \t\n\u00a0", ""),
    ("\u0964,!?\u2026", ""),
    ("Amar Mobile", "amar mobile"),
    ("आमार मुबायल", "आमार मुबायल"),
    ("३०", "३०"),
]

SAME_OUTPUT = [
    ("\u09c7\u09be", "\u09cb"),
    ("\u09df", "\u09af\u09bc"),
    ("\u09b0\u200d\u09cd\u09af\u09be\u09ac", "\u09b0\u09cd\u09af\u09be\u09ac"),
]

ALL_INPUTS = [text for text, _ in EXPECTED] + [text for pair in SAME_OUTPUT for text in pair]


@pytest.mark.parametrize(("text", "expected"), EXPECTED)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(("first", "second"), SAME_OUTPUT)
def test_equivalent_spellings_normalize_the_same(first: str, second: str) -> None:
    assert normalize(first) == normalize(second)


@pytest.mark.parametrize("text", ALL_INPUTS)
def test_normalize_is_idempotent(text: str) -> None:
    assert normalize(normalize(text)) == normalize(text)
