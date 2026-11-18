import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from kotha.text.normalize import BANGLA_DIGITS, normalize
from kotha.text.number_words import LETTER, folded, number_words_to_digits

CITATION_PATTERN = r"\[[^\[\]\s]+\]"
CITATION = re.compile(r"\[([^\[\]\s]+)\]")
CITATION_MASK = ""
CLOSING_MARKS = "\"'”’»)*_"
LINE_BREAKS = "\n\r\v\f\x85  "
TERMINATORS = f"।॥?!؟|{LINE_BREAKS}"
SENTENCE_END = re.compile(
    rf"(?:[{TERMINATORS}]|(?<!\d)\.|\.(?!\d))+[{CLOSING_MARKS}\s]*"
    rf"(?:{CITATION_MASK}+[{CLOSING_MARKS}\s{TERMINATORS}.]*)*"
)
ENDS_WITH_CITATION = re.compile(rf"{CITATION_PATTERN}[{CLOSING_MARKS}\s{TERMINATORS}.]*$")
ENDS_AS_QUESTION = re.compile(rf"[?؟][?؟!{CLOSING_MARKS}\s]*$")
COURTESY_PHRASES = frozenset(
    normalize(phrase)
    for phrase in (
        "sure",
        "okay",
        "ok",
        "certainly",
        "thanks",
        "thank you",
        "thank you for calling",
        "you are welcome",
        "you're welcome",
        "জি",
        "জী",
        "আচ্ছা",
        "ঠিক আছে",
        "অবশ্যই",
        "ধন্যবাদ",
        "আপনাকে ধন্যবাদ",
    )
)

GROUPED_NUMBER = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{2,3})*,\d{3}(?![\d,])")
FREE_OF_CHARGE = re.compile(
    folded(
        r"(?<![a-z])(?:free|no charge|costs? nothing)(?![a-z])"
        rf"|(?<![{LETTER}])ফ্রি|বিনামূল্যে|বিনা মূল্যে|কোনো টাকা লাগবে না"
    )
)
NUMBER_PATTERN = r"\d{1,2}:\d{2}|\d+(?:\.\d+)?|(?<![\w.])\.\d+"
NUMBER = re.compile(NUMBER_PATTERN)
NON_MONEY_UNIT = (
    r"(?:(?:gb|mb|kb|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|weeks?|months?|years?"
    r"|sms|times?|kinds?|ways?|steps?|packs?|digits?|attempts?|working|percent)(?![a-z])|%"
    r"|জিবি|এমবি|দিন|ঘণ্টা|মিনিট|সেকেন্ড|সপ্তাহ|মাস|বছর|বার|ধরন|ধাপ|শতাংশ|কর্মদিবস|কার্যদিবস"
    r"|অঙ্ক|সংখ্যা|টি|টা(?![কঃ])|জন)"
)
MONEY_NUMBER = re.compile(
    folded(
        rf"({NUMBER_PATTERN})\s*(?:/-|/=|[৳$€₹£]|(?:bangladeshi\s+)?"
        r"(?:taka\w*|tk(?![a-z])|bdt(?![a-z])|rs(?![a-z])|rupees?|dollars?|bucks?|paisa|poisha)"
        r"|(?:বাংলাদেশি\s+)?টাকা|টাঃ|পয়সা)"
        rf"|(?:[৳$€₹£]|(?<![a-z])(?:tk|bdt|rs))[.:\-\s]*({NUMBER_PATTERN})"
        r"|(?:টাকা|(?<![a-z])taka)[.:\-\s]*"
        rf"({NUMBER_PATTERN})(?![\d.]*[\s-]*{NON_MONEY_UNIT})"
    )
)
BARE_AMOUNT = re.compile(
    rf"(?:^|(?<![a-z])(?:is|for|only)\s+)({NUMBER_PATTERN})[\s.।!{CLOSING_MARKS}]*$"
)
PRICE_TERM = re.compile(
    folded(
        r"(?<![a-z])(?:pay|pays|paid|paying|prices?|priced|costs?|fees?|charges?|charged|charging"
        r"|rates?|tariffs?)(?![a-z])|দাম|(?<!বিনা)(?<!বিনা )মূল্য"
    )
)
ACCOUNT_MONEY_TERM = re.compile(
    folded(
        r"(?<![a-z])(?:balance|deduct\w*|back)(?![a-z])"
        r"|ব্যালেন্স|ব্যালান্স|খরচ|কেটে|কাটা|ফেরত|দিতে হবে|লাগবে"
    )
)
NOT_AN_AMOUNT = re.compile(
    folded(
        r"(?<![a-z0-9])n[tvc]-[a-z0-9-]+|\*?747[\d*#]*|(?<!\d)0?1700000009(?!\d)|\d{1,2}:\d{2}"
        rf"|(?:{NUMBER_PATTERN})[\s-]*{NON_MONEY_UNIT}"
    )
)


class ProblemKind(StrEnum):
    EMPTY_ANSWER = "empty_answer"
    MISSING_CITATION = "missing_citation"
    UNKNOWN_CITATION = "unknown_citation"
    UNSUPPORTED_NUMBER = "unsupported_number"


@dataclass(frozen=True)
class CitationProblem:
    kind: ProblemKind
    detail: str


def visible_text(text: str) -> str:
    return folded("".join(char for char in text if unicodedata.category(char) != "Cf"))


def strip_citations(text: str) -> str:
    return " ".join(CITATION.sub(" ", text).split())


def cited_chunk_ids(answer: str) -> list[str]:
    return list(dict.fromkeys(CITATION.findall(answer)))


def has_words(text: str) -> bool:
    return any(char.isalnum() for char in strip_citations(text))


def split_sentences(answer: str) -> list[str]:
    masked = CITATION.sub(lambda citation: CITATION_MASK * len(citation[0]), answer)
    sentences: list[str] = []
    start = 0
    for boundary in SENTENCE_END.finditer(masked):
        sentences.append(answer[start : boundary.end()].strip())
        start = boundary.end()
    sentences.append(answer[start:].strip())
    return [sentence for sentence in sentences if has_words(sentence)]


def digit_text(text: str) -> str:
    lowered = visible_text(text).translate(BANGLA_DIGITS).lower()
    return GROUPED_NUMBER.sub(lambda number: number[0].replace(",", ""), lowered)


def comparable_text(text: str) -> str:
    return number_words_to_digits(FREE_OF_CHARGE.sub(" 0 ", digit_text(text)))


def canonical_number(token: str) -> str:
    if ":" in token:
        hour, minute = token.split(":")
        return f"{int(hour)}:{minute}"
    whole, _, fraction = token.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def numbers_in(text: str) -> list[str]:
    return [canonical_number(token) for token in NUMBER.findall(comparable_text(text))]


def source_numbers(texts: Iterable[str]) -> set[str]:
    return {number for text in texts for number in numbers_in(text)}


def money_numbers(sentence: str) -> set[str]:
    spoken = strip_citations(sentence)
    digits_only = digit_text(spoken)
    with_words = comparable_text(spoken)
    tokens = ["".join(groups) for groups in MONEY_NUMBER.findall(with_words)]
    tokens += BARE_AMOUNT.findall(NOT_AN_AMOUNT.sub(" ", digits_only))
    if PRICE_TERM.search(with_words):
        tokens += NUMBER.findall(NOT_AN_AMOUNT.sub(" ", with_words))
    elif ACCOUNT_MONEY_TERM.search(digits_only):
        tokens += NUMBER.findall(NOT_AN_AMOUNT.sub(" ", digits_only))
    return {canonical_number(token) for token in tokens}


def needs_citation(sentence: str) -> bool:
    spoken = strip_citations(sentence)
    return not (ENDS_AS_QUESTION.search(spoken) or normalize(spoken) in COURTESY_PHRASES)


def verify_answer(
    answer: str,
    retrieved: Mapping[str, str],
    tool_results: Sequence[str] = (),
    money_amounts: Sequence[str] = (),
) -> list[CitationProblem]:
    if isinstance(tool_results, str) or isinstance(money_amounts, str):
        raise TypeError("tool_results and money_amounts must be sequences of strings")
    clean_answer = visible_text(answer)
    spoken_text = strip_citations(clean_answer)
    if not has_words(spoken_text):
        return [CitationProblem(ProblemKind.EMPTY_ANSWER, "")]
    sentences = split_sentences(clean_answer)
    cited_ids = cited_chunk_ids(clean_answer)
    allowed_money = source_numbers(money_amounts)
    tool_numbers = source_numbers(tool_results) | allowed_money
    cited_numbers = source_numbers(
        retrieved[chunk_id] for chunk_id in cited_ids if chunk_id in retrieved
    )
    stated_money = {number for sentence in sentences for number in money_numbers(sentence)}
    uncited = [
        sentence
        for sentence in sentences
        if needs_citation(sentence) and not ENDS_WITH_CITATION.search(sentence)
    ]
    unknown_ids = [chunk_id for chunk_id in cited_ids if chunk_id not in retrieved]
    unsupported = [
        number
        for number in dict.fromkeys(numbers_in(spoken_text))
        if (number in stated_money and number not in allowed_money)
        or (number not in cited_numbers and number not in tool_numbers)
    ]
    return [
        *(CitationProblem(ProblemKind.MISSING_CITATION, sentence) for sentence in uncited),
        *(CitationProblem(ProblemKind.UNKNOWN_CITATION, chunk_id) for chunk_id in unknown_ids),
        *(CitationProblem(ProblemKind.UNSUPPORTED_NUMBER, number) for number in unsupported),
    ]
