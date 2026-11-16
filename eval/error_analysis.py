from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jiwer

BENGALI_BLOCK = range(0x0980, 0x0A00)
CATEGORIES = ("loanword-script", "script", "number", "other")


@dataclass(frozen=True)
class Substitution:
    ref: str
    hyp: str
    count: int
    category: str


def substitution_counts(pairs: Sequence[tuple[str, str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    if not pairs:
        return counts
    output = jiwer.process_words(
        [reference for reference, _ in pairs], [hypothesis for _, hypothesis in pairs]
    )
    for ref_words, hyp_words, chunks in zip(
        output.references, output.hypotheses, output.alignments, strict=True
    ):
        for chunk in chunks:
            if chunk.type == "substitute":
                substituted = zip(
                    ref_words[chunk.ref_start_idx : chunk.ref_end_idx],
                    hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx],
                    strict=True,
                )
                counts.update(substituted)
    return counts


def has_bengali(word: str) -> bool:
    return any(ord(char) in BENGALI_BLOCK for char in word)


def has_latin(word: str) -> bool:
    return any(char.isascii() and char.isalpha() for char in word)


def is_latin_word(word: str) -> bool:
    return has_latin(word) and not has_bengali(word)


def is_bengali_word(word: str) -> bool:
    return has_bengali(word) and not has_latin(word)


def categorize(ref_word: str, hyp_word: str, lexicon: Mapping[str, str]) -> str:
    if lexicon.get(hyp_word) == ref_word:
        return "loanword-script"
    if any(char.isascii() and char.isdigit() for char in ref_word + hyp_word):
        return "number"
    if (is_latin_word(ref_word) and is_bengali_word(hyp_word)) or (
        is_bengali_word(ref_word) and is_latin_word(hyp_word)
    ):
        return "script"
    return "other"


def top_substitutions(
    pairs: Sequence[tuple[str, str]], lexicon: Mapping[str, str], limit: int = 20
) -> list[Substitution]:
    ranked = sorted(
        substitution_counts(pairs).items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )
    return [
        Substitution(ref, hyp, count, categorize(ref, hyp, lexicon))
        for (ref, hyp), count in ranked[:limit]
    ]


def category_counts(substitutions: Sequence[Substitution]) -> dict[str, int]:
    counts = Counter(substitution.category for substitution in substitutions)
    return {category: counts[category] for category in CATEGORIES}
