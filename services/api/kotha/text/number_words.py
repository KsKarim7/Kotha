import re
import unicodedata
from decimal import Decimal

BANGLA_0_TO_99 = tuple(
    """
    শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয়
    দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ
    বিশ একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ উনত্রিশ
    ত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ সাঁইত্রিশ আটত্রিশ উনচল্লিশ
    চল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ আটচল্লিশ উনপঞ্চাশ
    পঞ্চাশ একান্ন বায়ান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন উনষাট
    ষাট একষট্টি বাষট্টি তেষট্টি চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি উনসত্তর
    সত্তর একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর আটাত্তর উনআশি
    আশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি আটাশি উননব্বই
    নব্বই একানব্বই বিরানব্বই তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই
    """.split()
)
BANGLA_VARIANTS = {
    "এগার": 11,
    "তের": 13,
    "চোদ্দ": 14,
    "পনের": 15,
    "ষোল": 16,
    "সতের": 17,
    "আঠার": 18,
    "কুড়ি": 20,
    "তিরিশ": 30,
    "দু": 2,
    "দুটি": 2,
    "দুটো": 2,
    "তিনটে": 3,
    "নয়টি": 9,
    "নয়টা": 9,
    "দেড়": 1.5,
    "আড়াই": 2.5,
}
ENGLISH_NUMBERS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
MULTIPLIERS = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000,
    "lac": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000,
    "শো": 100,
    "শ": 100,
    "শত": 100,
    "হাজার": 1_000,
    "লাখ": 100_000,
    "লক্ষ": 100_000,
    "কোটি": 10_000_000,
}
BANGLA_HUNDRED_PREFIXES = {
    "এক": 1,
    "দুই": 2,
    "দু": 2,
    "তিন": 3,
    "চার": 4,
    "পাঁচ": 5,
    "ছয়": 6,
    "সাত": 7,
    "আট": 8,
    "নয়": 9,
    "দেড়": 1.5,
    "আড়াই": 2.5,
}
# these are mostly ordinary words ("নয়" also means "is not"), so they count only before an amount
AMBIGUOUS_WORDS = {"one": 1, "zero": 0, "এক": 1, "নয়": 9, "শূন্য": 0}
AMBIGUOUS_BANGLA = {"শূন্য", "এক", "নয়"}
COUNT_SUFFIXES = (
    "ঘণ্টা",
    "মিনিট",
    "টির",
    "টার",
    "টায়",
    "টাই",
    "দিন",
    "মাস",
    "বার",
    "টি",
    "টা",
    "টো",
    "টে",
    "জন",
    "গুণ",
    "ের",
    "এর",
)
AMOUNT_WORDS = (
    "taka",
    "tk",
    "bdt",
    "days",
    "day",
    "hours",
    "hour",
    "minutes",
    "minute",
    "months",
    "month",
    "weeks",
    "week",
    "years",
    "year",
    "gb",
    "mb",
    "sms",
    "টাকা",
    "দিন",
    "মাস",
    "ঘণ্টা",
    "মিনিট",
    "সপ্তাহ",
    "বছর",
    "জিবি",
    "এমবি",
)
TENS_WORDS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
LETTER = r"a-zঀ-ৣৰৱ"


def folded(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def number_word_values() -> dict[str, float]:
    bangla = {
        word: value for value, word in enumerate(BANGLA_0_TO_99) if word not in AMBIGUOUS_BANGLA
    }
    hundreds = {
        f"{prefix}{hundred}": value * 100
        for prefix, value in BANGLA_HUNDRED_PREFIXES.items()
        for hundred in ("শো", "শ", "শত")
    }
    values = {**bangla, **BANGLA_VARIANTS, **ENGLISH_NUMBERS, **MULTIPLIERS, **hundreds}
    return {folded(word): value for word, value in values.items()}


NUMBER_WORD_VALUES = number_word_values()
MULTIPLIER_WORDS = frozenset(folded(word) for word in MULTIPLIERS)
AMBIGUOUS_VALUES = {folded(word): value for word, value in AMBIGUOUS_WORDS.items()}
MULTIPLIER_ALTERNATIVES = "|".join(
    re.escape(word) for word in sorted(MULTIPLIER_WORDS, key=len, reverse=True)
)
WORD_ALTERNATIVES = "|".join(
    re.escape(word) for word in sorted(NUMBER_WORD_VALUES, key=len, reverse=True)
)
AMBIGUOUS_BEFORE_AMOUNT = (
    rf"(?:{'|'.join(AMBIGUOUS_VALUES)})"
    rf"(?=[\s-]+(?:{MULTIPLIER_ALTERNATIVES}|{'|'.join(folded(word) for word in AMOUNT_WORDS)})"
    rf"(?![{LETTER}]))"
)
AFTER_TENS = "|".join(rf"(?<={tens}[\s-])" for tens in TENS_WORDS)
ONE_AFTER_TENS = rf"(?:{AFTER_TENS})one"
SINGLE_WORD = rf"(?:{AMBIGUOUS_BEFORE_AMOUNT}|{ONE_AFTER_TENS}|{WORD_ALTERNATIVES})"
SUFFIX_PATTERN = "|".join(folded(suffix) for suffix in COUNT_SUFFIXES)
NUMBER_PHRASE = re.compile(
    rf"(?<![{LETTER}\d])({SINGLE_WORD}(?:[\s-]+(?:and[\s-]+)?{SINGLE_WORD})*)"
    rf"((?:{SUFFIX_PATTERN})?)(?![{LETTER}\d])"
)
DIGITS_WITH_MULTIPLIER = re.compile(
    rf"(?<![\d.:])(\d+(?:\.\d+)?)\s*({MULTIPLIER_ALTERNATIVES}|k)(?![{LETTER}\d])"
)
WORD_SEPARATOR = re.compile(r"[\s-]+(?:and[\s-]+)?")
# "লক্ষ" alone usually means "notice", as in "লক্ষ করুন"
BARE_LAKH = folded("লক্ষ")
FRACTION_OFFSETS = {folded("সাড়ে"): 0.5, folded("সোয়া"): 0.25, folded("পৌনে"): -0.25}
FRACTION_BEFORE_NUMBER = re.compile(rf"({'|'.join(FRACTION_OFFSETS)})\s*(\d+)")


def phrase_value(phrase: str) -> float:
    total: float = 0
    current: float = 0
    for word in WORD_SEPARATOR.split(phrase):
        value = AMBIGUOUS_VALUES.get(word, NUMBER_WORD_VALUES.get(word, 0))
        if word in MULTIPLIER_WORDS:
            current = (current or 1) * value
            if value >= 1_000:
                total += current
                current = 0
        else:
            current += value
    return total + current


def plain_number(value: float | Decimal) -> str:
    return str(int(value)) if value == int(value) else str(value)


def phrase_digits(phrase: re.Match[str]) -> str:
    if phrase[0] == BARE_LAKH:
        return phrase[0]
    return f"{plain_number(phrase_value(phrase[1]))}{phrase[2]}"


def multiplied_digits(number: re.Match[str]) -> str:
    multiplier = 1_000 if number[2] == "k" else NUMBER_WORD_VALUES[number[2]]
    return plain_number(Decimal(number[1]) * Decimal(multiplier))


def fraction_digits(fraction: re.Match[str]) -> str:
    return plain_number(int(fraction[2]) + FRACTION_OFFSETS[fraction[1]])


def number_words_to_digits(folded_lowercase_text: str) -> str:
    multiplied = DIGITS_WITH_MULTIPLIER.sub(multiplied_digits, folded_lowercase_text)
    digits = NUMBER_PHRASE.sub(phrase_digits, multiplied)
    return FRACTION_BEFORE_NUMBER.sub(fraction_digits, digits)
