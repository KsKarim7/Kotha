import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from kotha.dialogue.routing import Language
from kotha.text.number_words import BANGLA_0_TO_99
from kotha.tts.speakable import (
    VOICE_CHARACTERS,
    num_to_bn_words,
    num_to_en_words,
    speakable,
    speakable_sentences,
    split_sentences,
    spoken_text,
)

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
# the voices' vocab.json (ADR 0009) minus digits and punctuation; "6" is the Bangla blank token
BANGLA_VOICE_CHARACTERS = frozenset(" ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎ")
ENGLISH_VOICE_CHARACTERS = frozenset(" 'abcdefghijklmnopqrstuvwxyz")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@pytest.mark.parametrize("number", range(100))
def test_num_to_bn_words_uses_the_table_below_100(number: int) -> None:
    assert num_to_bn_words(number) == BANGLA_0_TO_99[number]


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (100, "একশো"),
        (105, "একশো পাঁচ"),
        (250, "দুইশো পঞ্চাশ"),
        (999, "নয়শো নিরানব্বই"),
        (1_000, "এক হাজার"),
        (1_001, "এক হাজার এক"),
        (1_500, "এক হাজার পাঁচশো"),
        (12_345, "বারো হাজার তিনশো পঁয়তাল্লিশ"),
        (99_999, "নিরানব্বই হাজার নয়শো নিরানব্বই"),
        (100_000, "এক লাখ"),
        (250_000, "দুই লাখ পঞ্চাশ হাজার"),
        (1_000_100, "দশ লাখ একশো"),
        (10_000_000, "এক কোটি"),
        (123_456_789, "বারো কোটি চৌত্রিশ লাখ ছাপ্পান্ন হাজার সাতশো উননব্বই"),
        (1_000_000_000, "একশো কোটি"),
        (2_500_000_007, "দুইশো পঞ্চাশ কোটি সাত"),
    ],
)
def test_num_to_bn_words_composes_hundreds_thousands_lakhs_and_crores(
    number: int, words: str
) -> None:
    assert nfc(num_to_bn_words(number)) == nfc(words)


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (20, "twenty"),
        (21, "twenty one"),
        (99, "ninety nine"),
        (100, "one hundred"),
        (105, "one hundred five"),
        (999, "nine hundred ninety nine"),
        (1_000, "one thousand"),
        (1_500, "one thousand five hundred"),
        (100_000, "one hundred thousand"),
        (1_000_001, "one million one"),
        (
            123_456_789,
            "one hundred twenty three million four hundred fifty six thousand seven"
            " hundred eighty nine",
        ),
        (2_000_000_000, "two billion"),
    ],
)
def test_num_to_en_words(number: int, words: str) -> None:
    assert num_to_en_words(number) == words


@pytest.mark.parametrize("to_words", [num_to_bn_words, num_to_en_words])
def test_negative_numbers_are_refused(to_words: Callable[[int], str]) -> None:
    with pytest.raises(ValueError, match="-1"):
        to_words(-1)


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("প্যাকের মেয়াদ ৩০ দিন।", "প্যাকের মেয়াদ ত্রিশ দিন"),
        ("প্যাকের মেয়াদ 30 দিন।", "প্যাকের মেয়াদ ত্রিশ দিন"),
        ("প্যাক NT-30GB সাত দিন চলবে।", "প্যাক এন টি ত্রিশ জিবি সাত দিন চলবে"),
        ("NT-30GB-তে ১.৫ GB বোনাস।", "এন টি ত্রিশ জিবিতে এক দশমিক পাঁচ জিবি বোনাস"),
        ("NT-NIGHT-5GB রাতে চলে।", "এন টি নাইট পাঁচ জিবি রাতে চলে"),
        ("৫০০ MB ডেটা", "পাঁচশো এমবি ডেটা"),
        ("সকাল ৮টা থেকে রাত ১১টা", "সকাল আটটা থেকে রাত এগারোটা"),
        ("৬টি ধাপ", "ছয়টি ধাপ"),
        ("২৪ ঘণ্টা আর ৯০ দিন", "চব্বিশ ঘণ্টা আর নব্বই দিন"),
        ("১,০০,০০০ মিনিট", "এক লাখ মিনিট"),
        ("৫০% বেশি", "পঞ্চাশ শতাংশ বেশি"),
    ],
)
def test_bangla_numbers_and_data_sizes_become_words(sentence: str, spoken: str) -> None:
    assert speakable(sentence, "bn") == nfc(spoken)


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("*747*1# ডায়াল করুন।", "স্টার সাত চার সাত স্টার এক হ্যাশ ডায়াল করুন"),
        ("`7470`-এ SMS করুন।", "সাত চার সাত শূন্য এ এস এম এস করুন"),
        ("হেল্পলাইন 01700000009।", "হেল্পলাইন শূন্য এক সাত শূন্য শূন্য শূন্য শূন্য শূন্য শূন্য শূন্য নয়"),
        ("747 নম্বরে", "সাত চার সাত নম্বরে"),
        ("কোড 05", "কোড শূন্য পাঁচ"),
    ],
)
def test_bangla_codes_and_phone_numbers_are_read_digit_by_digit(sentence: str, spoken: str) -> None:
    assert speakable(sentence, "bn") == nfc(spoken)


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("SIM বদলাতে NID কার্ড লাগবে।", "সিম বদলাতে এন আই ডি কার্ড লাগবে"),
        ("Nodi Telecom app-এ দেখুন", "নদী টেলিকম অ্যাপএ দেখুন"),
        ("AUTO ON লিখে পাঠান", "অটো অন লিখে পাঠান"),
        ("BAL লিখে পাঠান", "বি এ এল লিখে পাঠান"),
        ("PIN আর OTP", "পিন আর ও টি পি"),
    ],
)
def test_latin_words_in_bangla_become_bangla_script(sentence: str, spoken: str) -> None:
    assert speakable(sentence, "bn") == nfc(spoken)


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("The pack gives 8 GB for 7 days.", "the pack gives eight gigabytes for seven days"),
        ("You get 1 GB.", "you get one gigabyte"),
        ("You get 1.5 GB.", "you get one point five gigabytes"),
        ("Add 500MB.", "add five hundred megabytes"),
        ("It lasts 30 days.", "it lasts thirty days"),
        ("Open 24 hours.", "open twenty four hours"),
        ("About 20% more.", "about twenty percent more"),
        ("Up to 1,000 minutes.", "up to one thousand minutes"),
        ("From 9:30.", "from nine thirty"),
    ],
)
def test_english_numbers_and_data_sizes_become_words(sentence: str, spoken: str) -> None:
    assert speakable(sentence, "en") == spoken


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("Dial *747# to see the packs.", "dial star seven four seven hash to see the packs"),
        ("Send BAL to 7470 by SMS.", "send bee ay el to seven four seven zero by ess em ess"),
        ("Call 01700000009.", "call zero one seven zero zero zero zero zero zero zero nine"),
        ("Buy NT-30GB today.", "buy en tee thirty gigabytes today"),
        ("Keep your SIM and PIN safe.", "keep your sim and pin safe"),
        ("Don't share your OTP.", "don't share your oh tee pee"),
        ("Ask at any নদী service centre.", "ask at any nodi service centre"),
        ("I can help.", "i can help"),
    ],
)
def test_english_codes_acronyms_and_names(sentence: str, spoken: str) -> None:
    assert speakable(sentence, "en") == spoken


@pytest.mark.parametrize(
    ("sentence", "lang", "spoken"),
    [
        ("সিম বদলাতে কেন্দ্রে যান [sim-replacement.bn.0]।", "bn", "সিম বদলাতে কেন্দ্রে যান"),
        ("Visit a centre [sim-replacement.en.0].", "en", "visit a centre"),
        ("Visit [a.en.0] [b.en.1] now.", "en", "visit now"),
    ],
)
def test_citation_markers_are_removed(sentence: str, lang: Language, spoken: str) -> None:
    assert speakable(sentence, lang) == spoken


def test_precomposed_yya_and_format_characters_are_normalised() -> None:
    assert speakable("মেয়াদ", "bn") == nfc("মেয়াদ")
    assert speakable("র‍্যাব", "bn") == nfc("র্যাব")


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("## Service hours", "service hours"),
        ("This is **free** now.", "this is free now"),
        ("Dial *747# or #747*.", "dial star seven four seven hash or hash seven four seven star"),
        ("Press * then 5.", "press then five"),
    ],
)
def test_stars_and_hashes_are_spoken_only_inside_a_code_with_digits(
    sentence: str, spoken: str
) -> None:
    assert speakable(sentence, "en") == spoken


@pytest.mark.parametrize(
    ("sentence", "lang", "spoken"),
    [
        ("Open 10:00 to 18:00.", "en", "open ten o'clock to eighteen o'clock"),
        ("From 9:05 or 9:30.", "en", "from nine oh five or nine thirty"),
        ("সকাল ১০:০০ থেকে ১০:৩০", "bn", "সকাল দশটা থেকে দশটা ত্রিশ মিনিট"),
        ("Score 25:70.", "en", "score twenty five seventy"),
    ],
)
def test_clock_times_are_read_as_times(sentence: str, lang: Language, spoken: str) -> None:
    assert speakable(sentence, lang) == nfc(spoken)


@pytest.mark.parametrize(
    ("sentence", "lang", "spoken"),
    [
        ("**৩০ দিন** চলবে", "bn", "ত্রিশ দিন চলবে"),
        ("A **30 GB** pack.", "en", "a thirty gigabytes pack"),
        ("__7 days__ left", "en", "seven days left"),
    ],
)
def test_markdown_emphasis_around_numbers_is_not_a_code(
    sentence: str, lang: Language, spoken: str
) -> None:
    assert speakable(sentence, lang) == nfc(spoken)


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("5,10,15 days", "five ten fifteen days"),
        ("items 5,10", "items five ten"),
        ("1,000,000 SMS", "one million ess em ess"),
        ("1,00,000 minutes", "one hundred thousand minutes"),
        ("1,0000", "one zero zero zero zero"),
        (".5 and 1.2.3", "five and one point two three"),
    ],
)
def test_only_strict_grouping_makes_one_number_and_no_digit_is_left(
    sentence: str, spoken: str
) -> None:
    assert speakable(sentence, "en") == spoken


@pytest.mark.parametrize(
    ("sentence", "spoken"),
    [
        ("৩০,৬০ বা ৯০ দিন", "ত্রিশ ষাট বা নব্বই দিন"),
        ("১০০ টি প্যাক", "একশো টি প্যাক"),
        ("১০০০ এসএমএস", "এক হাজার এসএমএস"),
    ],
)
def test_bangla_lists_and_spaced_counts(sentence: str, spoken: str) -> None:
    assert speakable(sentence, "bn") == nfc(spoken)


@pytest.mark.parametrize(
    ("sentence", "lang", "spoken"),
    [
        ("A café visit.", "en", "a cafe visit"),
        ("café", "bn", "সি এ এফ ই"),
        ("Pay ৩০ দিন later.", "en", "pay thirty later"),
        ("１２ GB", "en", "twelve gigabytes"),
    ],
)
def test_accents_fullwidth_digits_and_bangla_in_english(
    sentence: str, lang: Language, spoken: str
) -> None:
    assert speakable(sentence, lang) == nfc(spoken)


def test_empty_and_punctuation_only_text_is_empty() -> None:
    assert speakable("", "bn") == ""
    assert speakable(" ।?! ", "en") == ""


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("এক। দুই? তিন! চার", ["এক।", "দুই?", "তিন!", "চার"]),
        ("Version 1.5 works. Next one", ["Version 1.5 works.", "Next one"]),
        ("One [a.en.0]. Two [b.en.1].", ["One  .", "Two  ."]),
        ("Line one\nLine two", ["Line one", "Line two"]),
        ("Wait... what", ["Wait.", ".", ".", "what"]),
        ("", []),
    ],
)
def test_split_sentences(text: str, sentences: list[str]) -> None:
    assert split_sentences(text) == sentences


def test_speakable_sentences_drops_empty_sentences() -> None:
    assert speakable_sentences("Wait... You get 1 GB [a.en.0]. ।", "en") == [
        "wait",
        "you get one gigabyte",
    ]
    assert speakable_sentences("৩০ দিন চলবে। শেষ?", "bn") == [nfc("ত্রিশ দিন চলবে"), "শেষ"]


@pytest.mark.parametrize(
    ("lang", "allowed"),
    [("bn", BANGLA_VOICE_CHARACTERS), ("en", ENGLISH_VOICE_CHARACTERS)],
)
def test_knowledge_base_sentences_lose_only_punctuation_to_the_vocabulary_filter(
    lang: Language, allowed: frozenset[str]
) -> None:
    assert VOICE_CHARACTERS[lang] == allowed
    files = sorted(KB_DIR.glob(f"*.{lang}.md"))
    assert files
    lost = {
        char
        for path in files
        for sentence in split_sentences(path.read_text(encoding="utf-8"))
        for char in spoken_text(sentence, lang)
        if char not in allowed and unicodedata.category(char)[0] not in "PSZ"
    }
    assert lost == set()


@pytest.mark.parametrize("lang", ["bn", "en"])
@pytest.mark.parametrize(
    "sentence",
    [
        ".5 1.2.3 1,0000 1,000,0 1,2 GB 5-10 -5 1/2 1st",
        "café naïve Ω İ ǅ ঌ ৠ ৗ ৱ ঽ ١٢٣ ３４ ½ 🙂 $ ₹ @ &",
        "Visit nodi.com.bd, e.g. today; ২০২৫ সালে ৳100 Tk 100",
    ],
)
def test_odd_input_never_leaves_the_voice_vocabulary(sentence: str, lang: Language) -> None:
    assert set(speakable(sentence, lang)) <= VOICE_CHARACTERS[lang]
