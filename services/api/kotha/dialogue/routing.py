import re
import unicodedata
from typing import Literal

from kotha.text.lexicon import BENGALI_BLOCK
from kotha.text.normalize import normalize

Language = Literal["bn", "en"]


def normalized_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(unicodedata.normalize("NFC", pattern))


CLAUSE_END = re.compile(r"[.!?।॥;,:\n]+")
HOW_TO = normalized_pattern(
    r"(?<!\S)(?:how(?! much| many| about)|where|when|why|can i|could i|should i|what do i|is there"
    r"|if|possible"
    r"|dial|code|ussd|app|কীভাবে|কিভাবে|কেমনে|কোথায়|কখন|কেন|উপায়|নিয়ম|কোড|ডায়াল"
    r"|কী করতে|কি করতে|কী করব|কি করব|কী করলে|কি করলে|যায়)(?!\S)"
)
HOLDING = (
    r"(?:balance|credit|data|gb|mb|talk ?time|minutes?|sms)"
    r"(?! (?:packs?|validity|history|list|options|usage|plans?)(?!\S))"
)
STILL_HELD = r"(?:do i have|have i got|is left|are left|left|remaining)(?! (?:after|when|at)(?!\S))"
OWN_BALANCE = normalized_pattern(
    rf"(?<!\S)(?:(?:what is|what s|whats|tell me|show me|check|give me) my (?:\S+ )?{HOLDING}"
    rf"|how (?:much|many) (?:\S+ )?{HOLDING} {STILL_HELD}|my (?:remaining|leftover) {HOLDING}"
    r"|(?:আমার|my) (?:\S+ ){0,2}(?:ব্যালেন্স|ব্যালান্স|balance|টাকা|ডেটা|ডাটা|data|মিনিট|minute)"
    r"\S* (?:\S+ ){0,2}(?:কত(?! দিন)|কতো(?! দিন)|কতটুকু|বলুন|বলেন|বলো|জানান|দেখুন|দেখেন"
    r"|জানতে চাই|চেক)"
    r"|(?:ব্যালেন্স|ব্যালান্স|balance)\S* (?:\S+ )?(?:কত|কতো)(?! দিন))(?!\S)"
)
ACTION_VERB = (
    r"(?:activate|buy|purchase|subscribe(?: to)?|recharge|top up|cancel|deactivate|renew"
    r"|turn (?:on|off)|switch (?:on|off)|block|unblock)"
)
SPECIFIC_OBJECT = r"(?:the|my|this|that|it|nt|nv|nc|\d\S*)"
ENGLISH_REQUEST = normalized_pattern(
    rf"^(?:just |now )?{ACTION_VERB} {SPECIFIC_OBJECT}(?!\S)"
    rf"|(?<!\S)(?:please|pls|just|i want you to|go ahead and) {ACTION_VERB}(?!\S)"
    rf"|(?<!\S){ACTION_VERB} .*(?<!\S)for me(?!\S)"
)
ENGLISH_INDIRECT_REQUEST = normalized_pattern(
    rf"(?<!\S)(?:(?:and|then|now) {ACTION_VERB}"
    rf"|(?:can|could|will|would)? ?you {ACTION_VERB} (?:\S+ )?{SPECIFIC_OBJECT}"
    r"|(?:give me|sign me up for|i need|i want) (?:\S+ )?(?:nt|nv|nc)"
    r"|(?:add|put|transfer|send) .*(?<!\S)(?:to|into|on) my (?:balance|account|number))(?!\S)"
)
BANGLA_REQUEST = normalized_pattern(
    r"(?<!\S)(?:চালু|বন্ধ|অফ|অন|কিনে|রিচার্জ|অ্যাক্টিভেট|এক্টিভেট|ক্যানসেল|ক্যান্সেল|বাতিল|নবায়ন"
    r"|রিনিউ|ব্লক|activate|cancel|recharge|buy|renew|block|on|off) (?:করে )?"
    r"(?:দিন|দেন|দাও|দে|দিবেন|দেবেন|করুন|করেন|করো|কর)(?!\S)"
)
BANGLA_INDIRECT_REQUEST = normalized_pattern(
    r"(?<!\S)(?:nt|nv|nc|প্যাক|pack)\S* (?:\S+ ){0,3}(?<!বলে )(?:দিয়ে )?(?:দিন|দেন|দাও)(?!\S)"
)


def caller_language(question: str) -> Language:
    bangla = any(char.isalpha() and ord(char) in BENGALI_BLOCK for char in question)
    return "bn" if bangla else "en"


def is_account_clause(clause: str) -> bool:
    text = normalize(clause)
    if ENGLISH_REQUEST.search(text) or BANGLA_REQUEST.search(text):
        return True
    if HOW_TO.search(text):
        return False
    indirect = (OWN_BALANCE, ENGLISH_INDIRECT_REQUEST, BANGLA_INDIRECT_REQUEST)
    return any(rule.search(text) for rule in indirect)


def is_account_request(question: str) -> bool:
    return any(is_account_clause(clause) for clause in CLAUSE_END.split(question))
