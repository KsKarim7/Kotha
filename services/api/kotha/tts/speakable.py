import re
import unicodedata

from kotha.dialogue.routing import Language
from kotha.rag.citations import CITATION
from kotha.text.lexicon import BENGALI_BLOCK
from kotha.text.normalize import BANGLA_DIGITS
from kotha.text.number_words import BANGLA_0_TO_99

BANGLA_LETTER = r"ঀ-৿"
ENGLISH_0_TO_19 = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    " fifteen sixteen seventeen eighteen nineteen"
).split()
ENGLISH_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
ENGLISH_SCALES = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))
DIGIT_NAMES = {"bn": BANGLA_0_TO_99[:10], "en": tuple(ENGLISH_0_TO_19[:10])}
DECIMAL_POINT = {"bn": "দশমিক", "en": "point"}
SYMBOL_WORDS = {
    "bn": {"*": "স্টার", "#": "হ্যাশ", "%": "শতাংশ"},
    "en": {"*": "star", "#": "hash", "%": "percent"},
}
DATA_UNITS = {
    "bn": {"gb": "জিবি", "mb": "এমবি", "kb": "কেবি"},
    "en": {"gb": "gigabyte", "mb": "megabyte", "kb": "kilobyte"},
}
LETTER_NAMES = {
    "bn": dict(
        zip(
            "abcdefghijklmnopqrstuvwxyz",
            "এ বি সি ডি ই এফ জি এইচ আই জে কে এল এম এন ও পি কিউ আর এস টি ইউ ভি"
            " ডাবলিউ এক্স ওয়াই জেড".split(),
            strict=True,
        )
    ),
    "en": dict(
        zip(
            "abcdefghijklmnopqrstuvwxyz",
            "ay bee see dee ee ef jee aitch eye jay kay el em en oh pee cue ar ess tee you vee"
            " double-you ex why zed".split(),
            strict=True,
        )
    ),
}
BANGLA_LOANWORDS = {
    "app": "অ্যাপ",
    "auto": "অটো",
    "balance": "ব্যালেন্স",
    "bonus": "বোনাস",
    "call": "কল",
    "card": "কার্ড",
    "code": "কোড",
    "data": "ডেটা",
    "internet": "ইন্টারনেট",
    "kotha": "কথা",
    "minute": "মিনিট",
    "mobile": "মোবাইল",
    "night": "নাইট",
    "nodi": "নদী",
    "off": "অফ",
    "offer": "অফার",
    "on": "অন",
    "online": "অনলাইন",
    "pack": "প্যাক",
    "package": "প্যাকেজ",
    "pin": "পিন",
    "recharge": "রিচার্জ",
    "roaming": "রোমিং",
    "sim": "সিম",
    "stop": "স্টপ",
    "telecom": "টেলিকম",
    "ticket": "টিকিট",
}
ENGLISH_ACRONYMS_SAID_AS_WORDS = frozenset({"sim", "pin", "nodi", "kotha"})
ENGLISH_FOR_BANGLA_NAMES = {"নদী": "nodi", "কথা": "kotha"}
# the voices' vocab.json (ADR 0009) without digits, punctuation and the Bangla blank token "6"
VOICE_CHARACTERS = {
    "bn": frozenset(" ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎ"),
    "en": frozenset(" 'abcdefghijklmnopqrstuvwxyz"),
}
UNIT_AFTER_NUMBER = re.compile(
    r"\s*(?:(?:gb|mb|kb|days?|hours?|minutes?|mins?|months?|weeks?|years?|sms|%)(?![a-z])"
    r"|দিন|ঘণ্টা|মিনিট|মাস|সপ্তাহ|বছর|জিবি|এমবি|এসএমএস|(?<=\d)(?:টা|টি|টে|টো)"
    r"|(?:টা|টি|টে|টো)(?:য়|র)?(?![ঀ-৿]))",
    re.IGNORECASE,
)
PERCENT = re.compile("%")
DATA_UNIT_AFTER_NUMBER = re.compile(r"\s*(gb|mb|kb)(?![a-z])", re.IGNORECASE)
GROUPED_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3})(?!,?\d)"
NUMBER = re.compile(rf"(?<![\d.])({GROUPED_NUMBER}|\d+)(?:\.(\d+))?(?!\d)")
CLOCK_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
DIGITS = re.compile(r"\d+")
EMPHASIS = re.compile(r"\*{2,}|_{2,}")
CODE = re.compile(r"(?<![*#\d])(?=[*#\d]*\d)[*#\d]*[*#][*#\d]*")
HYPHEN_BEFORE_BANGLA_SUFFIX = re.compile(rf"(?<=[A-Za-z\d])-(?=[{BANGLA_LETTER}])")
LATIN_WORD = re.compile(r"[A-Za-z]+")
BANGLA_WORD = re.compile(rf"[{BANGLA_LETTER}]+")
SENTENCE_BREAK = re.compile(r"(?<=[।॥?!\n])|(?<=\.)(?!\d)")


def num_to_bn_words(number: int) -> str:
    if number < 0:
        raise ValueError(f"expected a number of zero or more, got {number}")
    if number < 100:
        return BANGLA_0_TO_99[number]
    words: list[str] = []
    crores, rest = divmod(number, 10_000_000)
    if crores:
        words += [num_to_bn_words(crores), "কোটি"]
    for scale, name in ((100_000, "লাখ"), (1_000, "হাজার")):
        count, rest = divmod(rest, scale)
        if count:
            words += [BANGLA_0_TO_99[count], name]
    hundreds, rest = divmod(rest, 100)
    if hundreds:
        words.append(f"{BANGLA_0_TO_99[hundreds]}শো")
    if rest:
        words.append(BANGLA_0_TO_99[rest])
    return " ".join(words)


def num_to_en_words(number: int) -> str:
    if number < 0:
        raise ValueError(f"expected a number of zero or more, got {number}")
    if number < 20:
        return ENGLISH_0_TO_19[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        tens_word = ENGLISH_TENS[tens - 2]
        return f"{tens_word} {ENGLISH_0_TO_19[ones]}" if ones else tens_word
    if number < 1_000:
        hundreds, rest = divmod(number, 100)
        head = f"{ENGLISH_0_TO_19[hundreds]} hundred"
    else:
        scale, name = next((scale, name) for scale, name in ENGLISH_SCALES if number >= scale)
        count, rest = divmod(number, scale)
        head = f"{num_to_en_words(count)} {name}"
    return f"{head} {num_to_en_words(rest)}" if rest else head


def number_words(number: int, lang: Language) -> str:
    return num_to_bn_words(number) if lang == "bn" else num_to_en_words(number)


def digit_by_digit(digits: str, lang: Language) -> str:
    return " ".join(DIGIT_NAMES[lang][int(digit)] for digit in digits)


def spoken_code(code: str, lang: Language) -> str:
    words = (
        DIGIT_NAMES[lang][int(char)] if char.isdigit() else SYMBOL_WORDS[lang][char]
        for char in code
    )
    return f" {' '.join(words)} "


def data_unit(unit: str, singular: bool, lang: Language) -> str:
    word = DATA_UNITS[lang][unit.lower()]
    return word if lang == "bn" or singular else f"{word}s"


def spoken_number(match: re.Match[str], text: str, lang: Language) -> tuple[str, int]:
    grouped, fraction = match.group(1), match.group(2)
    whole = grouped.replace(",", "")
    after = match.end()
    is_amount = "," in grouped or UNIT_AFTER_NUMBER.match(text, after) is not None
    leading_zero = whole[0] == "0" and len(whole) > 1
    if fraction is None and (leading_zero or (len(whole) > 2 and not is_amount)):
        return f" {digit_by_digit(whole, lang)} ", after
    words = number_words(int(whole), lang)
    if fraction is not None:
        words += f" {DECIMAL_POINT[lang]} {digit_by_digit(fraction, lang)}"
    unit = DATA_UNIT_AFTER_NUMBER.match(text, after)
    if unit:
        singular = fraction is None and int(whole) == 1
        return f" {words} {data_unit(unit.group(1), singular, lang)}", unit.end()
    if BANGLA_WORD.match(text, after):
        return f" {words}", after
    return f" {words} ", after


def spoken_time(match: re.Match[str], lang: Language) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)
    if lang == "bn":
        minutes = f" {num_to_bn_words(minute)} মিনিট" if minute else ""
        return f" {num_to_bn_words(hour)}টা{minutes} "
    if minute == 0:
        return f" {num_to_en_words(hour)} o'clock "
    minutes = f"oh {num_to_en_words(minute)}" if minute < 10 else num_to_en_words(minute)
    return f" {num_to_en_words(hour)} {minutes} "


def spoken_numbers(text: str, lang: Language) -> str:
    pieces: list[str] = []
    position = 0
    for match in NUMBER.finditer(text):
        if match.start() < position:
            continue
        pieces.append(text[position : match.start()])
        words, position = spoken_number(match, text, lang)
        pieces.append(words)
    pieces.append(text[position:])
    return "".join(pieces)


def spoken_latin_word(word: str, lang: Language) -> str:
    lowered = word.lower()
    if lang == "bn":
        if lowered in BANGLA_LOANWORDS:
            return BANGLA_LOANWORDS[lowered]
        if lowered in DATA_UNITS["bn"]:
            return DATA_UNITS["bn"][lowered]
        return " ".join(LETTER_NAMES["bn"][letter] for letter in lowered)
    if lowered in DATA_UNITS["en"]:
        return f"{DATA_UNITS['en'][lowered]}s"
    if word.isupper() and len(word) > 1 and lowered not in ENGLISH_ACRONYMS_SAID_AS_WORDS:
        return " ".join(LETTER_NAMES["en"][letter] for letter in lowered)
    return word


def without_accents(text: str) -> str:
    return "".join(char if ord(char) in BENGALI_BLOCK else plain_character(char) for char in text)


def plain_character(char: str) -> str:
    decomposed = unicodedata.normalize("NFKD", char)
    return "".join(part for part in decomposed if not unicodedata.combining(part))


def speakable(sentence: str, lang: Language) -> str:
    text = spoken_text(sentence, lang)
    return " ".join(
        "".join(char if char in VOICE_CHARACTERS[lang] else " " for char in text).split()
    )


def spoken_text(sentence: str, lang: Language) -> str:
    text = CITATION.sub(" ", sentence)
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = without_accents(unicodedata.normalize("NFC", text)).translate(BANGLA_DIGITS)
    text = EMPHASIS.sub(" ", text)
    text = HYPHEN_BEFORE_BANGLA_SUFFIX.sub("", text)
    text = CODE.sub(lambda match: spoken_code(match.group(0), lang), text)
    text = CLOCK_TIME.sub(lambda match: spoken_time(match, lang), text)
    text = spoken_numbers(text, lang)
    text = DIGITS.sub(lambda match: f" {digit_by_digit(match.group(0), lang)} ", text)
    text = PERCENT.sub(f" {SYMBOL_WORDS[lang]['%']} ", text)
    text = LATIN_WORD.sub(lambda match: spoken_latin_word(match.group(0), lang), text)
    if lang == "en":
        text = BANGLA_WORD.sub(lambda match: english_name(match.group(0)), text).lower()
    return unicodedata.normalize("NFC", text)


def english_name(bangla_word: str) -> str:
    return ENGLISH_FOR_BANGLA_NAMES.get(bangla_word, bangla_word)


def split_sentences(text: str) -> list[str]:
    without_citations = CITATION.sub(" ", text)
    return [part.strip() for part in SENTENCE_BREAK.split(without_citations) if part.strip()]


def speakable_sentences(text: str, lang: Language) -> list[str]:
    sentences = (speakable(sentence, lang) for sentence in split_sentences(text))
    return [sentence for sentence in sentences if any(char.isalpha() for char in sentence)]
