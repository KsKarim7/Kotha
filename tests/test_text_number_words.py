import unicodedata

import pytest

from kotha.text.number_words import BANGLA_0_TO_99, number_words_to_digits


def digits_for(text: str) -> str:
    return number_words_to_digits(unicodedata.normalize("NFKC", text).lower())


def test_the_bangla_table_has_one_distinct_word_per_number() -> None:
    assert len(BANGLA_0_TO_99) == 100
    assert len(set(BANGLA_0_TO_99)) == 100
    assert BANGLA_0_TO_99[25] == "পঁচিশ"
    assert BANGLA_0_TO_99[99] == "নিরানব্বই"


@pytest.mark.parametrize("number", [*range(2, 9), *range(10, 100)])
def test_every_unambiguous_bangla_number_word_is_read(number: int) -> None:
    assert digits_for(f"মোট {BANGLA_0_TO_99[number]} টি") == f"মোট {number} টি"


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("পাঁচশো", "500"),
        ("পাঁচ শত", "500"),
        ("দেড়শ", "150"),
        ("আড়াইশো", "250"),
        ("দেড় হাজার", "1500"),
        ("দুই হাজার পাঁচশো", "2500"),
        ("এক লক্ষ", "100000"),
        ("তিন কোটি", "30000000"),
        ("তিনটি", "3টি"),
        ("তিনবার", "3বার"),
        ("5 লক্ষ", "500000"),
        ("5হাজার", "5000"),
        ("30k", "30000"),
        ("2.3 lakh", "230000"),
        ("1.1k", "1100"),
        ("10:00 k", "10:00 k"),
        ("এক দিন", "1 দিন"),
        ("শূন্য টাকা", "0 টাকা"),
        ("দুটো", "2"),
        ("নয়টি", "9"),
        ("five", "5"),
        ("twenty-five", "25"),
        ("two hundred and fifty", "250"),
        ("two thousand five hundred", "2500"),
        ("thirty one", "31"),
        ("সাড়ে 5", "5.5"),
        ("one thousand", "1000"),
        ("three lakh", "300000"),
    ],
)
def test_number_phrases_become_one_value(text: str, digits: str) -> None:
    assert digits_for(text) == digits


@pytest.mark.parametrize(
    "text",
    [
        "one of them",
        "এক কথায়",
        "একটি প্যাক",
        "এটি নয়",
        "শূন্য",
        "someone",
        "fivefold",
        "4two",
        "লক্ষ করুন",
    ],
)
def test_ordinary_words_stay_as_they_are(text: str) -> None:
    assert digits_for(text) == unicodedata.normalize("NFKC", text).lower()
